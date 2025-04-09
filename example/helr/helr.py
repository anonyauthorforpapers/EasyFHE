import os, sys, time
sys.path.append("/".join(os.getcwd().split("/")[:-2]))

import torch
import torch.fhe as fhe
import openfhe as openfhe
import numpy as np
import random, math

from torch.fhe.ciphertext import Cipher #todo: to be removed

DATA_DIR = os.environ["DATA_DIR"]

class Params:
    def __init__(self, factor_num, sample_num, iter_num, alpha, num_thread, slots):
        self.factor_num = 1 << int(np.ceil(np.log2(factor_num)))
        self.iter_num = iter_num
        self.num_thread = num_thread
        self.alpha = alpha

        self.w_bits = 59
        self.p_bits = 59
        self.l_bits = 5
        self.degree3 = [0.5, -0.0843, 0.0, 0.0002] 
        self.iter_per_boot = 5

        self.kdeg = 3
        
        self.encode_slots = slots

        self.f_num_bits = int(np.ceil(np.log2(factor_num)))
        self.s_num_bits = int(np.ceil(np.log2(sample_num)))

        self.cnum = num_thread
        self.batch = self.factor_num // self.cnum
        self.slots = slots
        self.block_size = self.slots // self.batch
        self.sample_num = (sample_num // self.block_size) * self.block_size
        if sample_num % self.block_size != 0:
            self.sample_num += self.block_size

        self.b_bits = int(np.ceil(np.log2(self.batch)))
        self.s_bits = int(np.ceil(np.log2(self.slots)))

        self.depth = 0

        self.is_first = True
        self.path_to_file = DATA_DIR + "MNIST_test.txt"
        self.path_to_test_file = DATA_DIR + "MNIST_train.txt"


        self.start_time_val = None
        self.end_time_val = None

        cryptoContext = None
        openfhe_context = None
        logBsSlots_list=None


        print("***********************************")
        print("Secure Machine Learning Parameters")
        print("***********************************")
        print(f"- factorNum = {self.factor_num}, sampleNum = {self.sample_num}, approximate degree = {self.kdeg}")
        print(f"- Iteration Number = {self.iter_num}, iterPerBoot = {self.iter_per_boot}, Learning Rate = {self.alpha}")
        print(f"- cnum = {self.cnum}, batch = {self.batch}, mini-batch Block Size = {self.block_size}")
        print()

    def start_time(self):
        self.start_time_val = time.perf_counter()

    def end_time(self):
        self.end_time_val = time.perf_counter()
        elapsed_time = self.end_time_val - self.start_time_val
        return elapsed_time

    def print_time(self, msg, elapsed_time):
        print(f"{msg} time = {elapsed_time:.4f} seconds")

class SecureML:
    def __init__(self, params, cryptoContext):
        self.params = params
        pvals = np.zeros(self.params.slots, dtype=complex) 
        for i in range(0, self.params.slots, self.params.batch):
            pvals[i] = 1.0
        pvals = torch.tensor(pvals, dtype=torch.float64).cuda()
        self.dummy = cryptoContext.openfhe_context.encode(pvals, 1, 0, self.params.encode_slots)


    def inner_product(self, cryptoContext, enc_z_data, enc_v_data):
        enc_ip_vec = []
        for i in range(self.params.cnum):
            enc_ip = fhe.homo_mul(enc_z_data[i], enc_v_data[i], cryptoContext)
            # enc_ip = fhe.homo_rescale(enc_ip, 1, cryptoContext) 

            for j in range(self.params.b_bits):
                enc_rot = fhe.homo_rotate(enc_ip, 1 << j, cryptoContext) 
                enc_ip = fhe.homo_add(enc_ip, enc_rot, cryptoContext) 
            enc_ip_vec.append(enc_ip)
        
        enc_ip = enc_ip_vec[0]
        for i in range(1, self.params.cnum):
            enc_ip = fhe.homo_add(enc_ip, enc_ip_vec[i], cryptoContext)

        enc_ip = fhe.homo_mul_pt(enc_ip, self.dummy, cryptoContext)
        # enc_ip = fhe.homo_rescale(enc_ip, 1, cryptoContext) 
        
        for i in range(self.params.b_bits):
            tmp = fhe.homo_rotate(enc_ip, -(1 << i), cryptoContext)
            enc_ip = fhe.homo_add(enc_ip, tmp, cryptoContext)


        return enc_ip

    def sigmoid(self, cryptoContext, enc_grad, enc_z_data, enc_ip, gamma):
        enc_ip_sqr = fhe.homo_square(enc_ip, cryptoContext)
        enc_ip_sqr = fhe.homo_add_scalar_double(enc_ip_sqr, self.params.degree3[1] / self.params.degree3[3], cryptoContext)

        for i in range(self.params.cnum): 
            enc_grad[i] = fhe.homo_mul_scalar_double(enc_z_data[i], gamma * self.params.degree3[3], cryptoContext)
            # enc_grad[i] = fhe.homo_rescale(enc_grad[i], 1, cryptoContext)
            enc_grad[i] = fhe.homo_mul(enc_grad[i], enc_ip, cryptoContext)
            # enc_grad[i] = fhe.homo_rescale(enc_grad[i], 1, cryptoContext) 
            enc_grad[i] = fhe.homo_mul(enc_grad[i], enc_ip_sqr, cryptoContext)
            # enc_grad[i] = fhe.homo_rescale(enc_grad[i], 1, cryptoContext) 

            tmp = fhe.homo_mul_scalar_double(enc_z_data[i], gamma * self.params.degree3[0], cryptoContext)
            # tmp = fhe.homo_rescale(tmp, 1, cryptoContext)
            enc_grad[i] = fhe.homo_add(enc_grad[i], tmp, cryptoContext)


            for l in range(self.params.b_bits, self.params.s_bits):
                tmp = fhe.homo_rotate(enc_grad[i], 1 << l, cryptoContext)
                enc_grad[i] = fhe.homo_add(enc_grad[i], tmp, cryptoContext)


    def innerproduct(self, vec1, vec2, size):
        ip = 0.0
        for i in range(size):
            ip += vec1[i] * vec2[i]
        return ip
    
    def plain_inner_product(self, ip, z_data, v_data, factor_num, block_size):
        for i in range(block_size):
            ip[i] = self.innerproduct(v_data, z_data[i][:factor_num], factor_num) 

    
    def plain_sigmoid(self, grad, z_data, ip, gamma, factor_num, sample_num):
        for i in range(sample_num):
            tmp = self.params.degree3[0] + self.params.degree3[1] * ip[i] + self.params.degree3[3] * ip[i] ** 3
            tmp *= gamma 
            for j in range(len(z_data[0])):
                grad[j] += tmp * z_data[i][j]

    def encrypt_z_data(self, cryptoContext, z_data, block_array, num_iter):
        for i in range(num_iter):
            block_id = block_array[i]
            for j in range(self.params.cnum):
                pz_data = np.zeros(self.params.slots, dtype=np.complex128)
                
                for k in range(self.params.block_size):
                    for l in range(self.params.batch):
                        if (self.params.block_size * block_id + k) < len(z_data) and (self.params.batch * j + l) < len(z_data[0]):
                            pz_data[self.params.batch * k + l] = z_data[self.params.block_size * block_id + k][self.params.batch * j + l]
                ptxt = self.params.openfhe_context.cc.MakeCKKSPackedPlaintext(pz_data.tolist(), 1, 0, None, self.params.encode_slots)
                ptxt.SetLength(self.params.slots)
                enc_z_data = self.params.openfhe_context.cc.Encrypt(self.params.openfhe_context.publicKey, ptxt)

                file_name = DATA_DIR + f"/helr/encData/{block_id + 1}_{j + 1}_cipher.txt"
                
                if not openfhe.SerializeToFile(file_name, enc_z_data, openfhe.BINARY):
                    raise IOError(f"Error writing serialization of ciphertext to {file_name}")

                del pz_data

    def update(self, cryptoContext, enc_w_data, enc_v_data, gamma, eta, block_id):
        enc_data = []

        for i in range(self.params.cnum):
            file_name = DATA_DIR + f"/helr/encData/{block_id + 1}_{i + 1}_cipher.txt"
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"File {file_name} not found.")
            
            try:
                enc_ciphertext, res = openfhe.DeserializeCiphertext(file_name, openfhe.BINARY)
                if not res:
                    raise IOError(f"Could not read the ciphertext from {file_name}")
                data = enc_ciphertext.GetVectorOfData()
                cv = [torch.tensor(elem, device="cuda", dtype=torch.uint64) for elem in data]
                enc_ciphertext = Cipher(cv, cv[0].shape[0], enc_ciphertext.GetNoiseScaleDeg(), enc_ciphertext.GetSlots(), is_ext=False)
                enc_data.append(enc_ciphertext)
            except Exception as e:
                raise IOError(f"Error deserializing {file_name}: {e}")
            

        self.params.start_time()

        enc_ip = self.inner_product(cryptoContext, enc_data, enc_v_data)

        enc_grad = [None] * self.params.cnum
        self.sigmoid(cryptoContext, enc_grad, enc_data, enc_ip, gamma)

        for i in range(self.params.cnum):
            enc_w_data[i] = enc_w_data[i].deep_copy()

            enc_w_data[i] = fhe.homo_add(enc_w_data[i], enc_grad[i], cryptoContext)
            tmp2 = fhe.homo_mul_scalar_double(enc_w_data[i], eta, cryptoContext)
            # tmp2 = fhe.homo_rescale(tmp2, 1, cryptoContext) 

            enc_v_data[i] = fhe.homo_mul_scalar_double(enc_w_data[i], 1.0 - eta, cryptoContext)
            # enc_v_data[i] = fhe.homo_rescale(enc_v_data[i], 1, cryptoContext) 

            # tmp2 = fhe.homo_rescale(tmp2, 1, cryptoContext)
            enc_v_data[i] = fhe.homo_add(enc_v_data[i], tmp2, cryptoContext)

            # enc_w_data[i] = fhe.homo_rescale(enc_w_data[i], 1, cryptoContext)

        elapsed_time = self.params.end_time()
        self.params.print_time("Encrypted Update", elapsed_time)
        print("Update finished!")

    def training(self, cryptoContext, enc_w_data, factor_num, sample_num, w_data, z_data, block_array):
        enc_v_data = [None] * self.params.cnum

        zero_vec = np.zeros(self.params.slots, dtype=complex)
        input_vec = zero_vec


        for i in range(self.params.cnum):
            enc_w_data[i] = self.params.openfhe_context.encrypt(input_vec, 1, 0, self.params.encode_slots)
            enc_v_data[i] = self.params.openfhe_context.encrypt(input_vec, 1, 0, self.params.encode_slots)

        v_data = np.zeros(self.params.factor_num)
        w_data = np.zeros(self.params.factor_num)

        alpha0 = 0.01
        alpha1 = (1.0 + np.sqrt(1.0 + 4.0 * alpha0 * alpha0)) / 2.0
        gamma = self.params.alpha / self.params.block_size

        z_data_test, factor_num_test, sample_num_test = self.z_data_from_file(self.params.path_to_test_file, True)
        z_data_test = self.normalize_z_data(z_data_test, factor_num_test, sample_num_test)

        for iter in range(self.params.iter_num):
            print(f"\n{iter + 1}-th iteration started !!!")

            eta = (1 - alpha0) / alpha1
            block_id = block_array[iter]
            print(f"blockid: {block_id}")
            print("** un-encrypted")
            
            self.plain_update(w_data, v_data, z_data, gamma, eta, factor_num, sample_num, block_id)
            self.test_auroc(self, z_data_test, factor_num_test, sample_num_test, w_data, self.params.is_first)

            self.update(cryptoContext, enc_w_data, enc_v_data, gamma, eta, block_id)

            dw_data = self.decrypt_w_data(cryptoContext, enc_w_data, factor_num)
            self.test_auroc(self, z_data_test, factor_num_test, sample_num_test, dw_data, self.params.is_first)

            alpha0 = alpha1
            alpha1 = (1.0 + np.sqrt(1.0 + 4.0 * alpha0 * alpha0)) / 2.0

            if iter % self.params.iter_per_boot == self.params.iter_per_boot - 1 and iter < self.params.iter_num - 1:
                print("\nBootstrapping START!!!")

                self.params.start_time()

                for i in range(self.params.cnum):
                    enc_w_data[i] = fhe.homo_bootstrap(enc_w_data[i], cryptoContext.L, self.params.logBsSlots_list[0], cryptoContext)
                    enc_v_data[i] = fhe.homo_bootstrap(enc_v_data[i], cryptoContext.L, self.params.logBsSlots_list[0], cryptoContext)

                elapsed_time = self.params.end_time()
                self.params.print_time("bootstrapping", elapsed_time)
                print("Bootstrapping END!!!")

                dwdata = self.decrypt_w_data(cryptoContext,  enc_w_data, factor_num)

                self.test_auroc(self, z_data_test, factor_num_test, sample_num_test, dwdata, self.params.is_first)


    def plain_training(self, cryptoContext, w_data, z_data, factor_num, sample_num):
        gamma = 0
        eta = 0
        alpha0 = 0.01
        alpha1 = (1.0 + math.sqrt(1.0 + 4.0 * alpha0 ** 2)) / 2.0

        v_data = np.zeros(self.params.factor_num)
        w_data.fill(0.0)
        gamma = self.params.alpha / self.params.block_size
        block_num = self.params.sample_num // self.params.block_size
        auc, accuracy = 0, 0


        z_data_test, factor_num_test, sample_num_test = self.z_data_from_file(self.params.path_to_test_file,self.params.isfirst)
        self.normalize_z_data(z_data_test,factor_num_test,sample_num_test)

        random.seed(1)

        for iter in range(self.params.iter_num):
            print(f"\n{iter + 1}-th iteration started (plain)!!!")

            eta = (1 - alpha0) / alpha1

            self.params.start_time()
            block_id = random.randint(0, block_num - 1)
            self.plain_update(w_data, v_data, z_data, gamma, eta, factor_num, sample_num, block_id)
            elapsed_time = self.params.end_time()
            self.params.print_time("Plain Update", elapsed_time)

            self.test_auroc(self, z_data_test, factor_num_test, sample_num_test, w_data, self.params.is_first)

            alpha0 = alpha1
            alpha1 = (1.0 + math.sqrt(1.0 + 4.0 * alpha0 ** 2)) / 2.0

    def plain_update(self, w_data, v_data, z_data, gamma, eta, factor_num, sample_num, block_id):
        z_block_data = np.zeros((self.params.block_size, self.params.factor_num))
        for i in range(self.params.block_size):
            for j in range(self.params.factor_num):
                index = block_id * self.params.block_size + i
                if index < sample_num and j < len(z_data[0]):
                    z_block_data[i, j] = z_data[index, j]
                else:
                    z_block_data[i, j] = 0.0

        grad = np.zeros(self.params.factor_num)
        ip = np.zeros(self.params.block_size)

        self.plain_inner_product(ip, z_block_data, v_data, self.params.factor_num, self.params.block_size)
        self.plain_sigmoid(grad, z_block_data, ip, gamma, self.params.factor_num, self.params.block_size)

        for i in range(self.params.factor_num):
            tmp1 = v_data[i] + grad[i]
            
            tmp2 = eta * w_data[i]
            v_data[i] = tmp1 * (1.0 - eta) + tmp2
            w_data[i] = tmp1

    def decrypt_w_data(self, cryptoContext, enc_w_data, factor_num):
        w_data = np.zeros(factor_num)
        for i in range(self.params.cnum):
            result= self.params.openfhe_context.decrypt(enc_w_data[i])
            for j in range(self.params.batch):
                if self.params.batch * i + j < factor_num:
                    w_data[self.params.batch * i + j] = result[j]
        return w_data
    
    def decrypt_w_data_and_save(self, cryptoContext,  enc_w_data, factor_num, file_name):
        w_data = np.zeros(factor_num)
        for i in range(self.params.cnum):
            result = self.params.openfhe_context.decrypt(enc_w_data[i])


            for j in range(self.params.batch):
                if self.params.batch * i + j < factor_num:
                    w_data[self.params.batch* i + j] = result[j]

        with open(file_name, 'w') as file:
            for i in range(factor_num):
                file.write(f"{i + 1}, {w_data[i]}\n")

    def decrypt_and_print(self, cryptoContext, msg, cipher):
        result = self.params.openfhe_context.decrypt(cipher)
        print(f"{msg} = [{', '.join(map(str, result[:10]))}]")

    @staticmethod
    def z_data_from_file(path, is_first):
        data = []
        with open(path, "r") as file:
            lines = file.readlines()

        factor_dim = len(lines[0].strip().split(","))
        sample_dim = len(lines) - 1

        for line in lines[1:]:
            values = list(map(float, line.strip().split(",")))
            data.append(values)

        z_data = np.zeros((sample_dim, factor_dim))

        if is_first:
            for j in range(sample_dim):
                z_data[j, 0] = 2 * data[j][0] - 1
                for i in range(1, factor_dim):
                    z_data[j, i] = z_data[j, 0] * data[j][i]
        else:
            for j in range(sample_dim):
                z_data[j, 0] = 2 * data[j][factor_dim - 1] - 1
                for i in range(1, factor_dim):
                    z_data[j, i] = z_data[j, 0] * data[j][i - 1]
        return z_data, factor_dim, sample_dim
    
    @staticmethod
    def normalize_z_data(z_data, factor_dim, sample_dim):
        z_data = np.array(z_data)
        for i in range(factor_dim):
            m = np.max(np.abs(z_data[:, i]))
            if m > 1e-10:
                z_data[:, i] /= m
        return z_data
    
    @staticmethod
    def shuffle_z_data(z_data, factor_dim, sample_dim):
        np.random.seed(1)
        tmp = np.zeros(factor_dim)
        
        for i in range(sample_dim):
            idx = i + np.random.randint(0, sample_dim - i)

            tmp[:] = z_data[i]
            z_data[i] = z_data[idx]
            z_data[idx] = tmp
        
        return z_data
    
    @staticmethod
    def test_auroc(self, z_data, factor_dim, sample_dim, w_data, is_first):
        print("\t - wData = [", end="")
        for i in range(min(10, len(w_data))):
            print(w_data[i], end=", ")
            if i == 9:
                print(w_data[i], "]")
        
        TN = 0
        FP = 0
        theta_TN = []
        theta_FP = []

        for i in range(sample_dim):
            if z_data[i][0] > 0:
                if self.innerproduct(z_data[i], w_data,factor_dim) < 0:
                    TN += 1
                theta_TN.append(z_data[i][0] * self.innerproduct(z_data[i][1:], w_data[1:],factor_dim-1))
            else:
                if self.innerproduct(z_data[i], w_data,factor_dim) < 0:
                    FP += 1
                theta_FP.append(z_data[i][0] * self.innerproduct(z_data[i][1:], w_data[1:],factor_dim-1))

        accuracy = (sample_dim - TN - FP) / sample_dim
        print("\t - Accuracy:", accuracy)
        print( "\t - TN:", TN, ", FP:", FP)

        auc = 0.0
        if not theta_FP or not theta_TN:
            print("\t - n_test_yi = 0 : cannot compute AUC")
        else:
            for theta_tn in theta_TN:
                for theta_fp in theta_FP:
                    if theta_fp <= theta_tn:
                        auc += 1
            auc /= len(theta_TN) * len(theta_FP)
            print("\t - AUC:", auc)
        
        return auc, accuracy


def test(cryptoContext, openfhe_context, encode_slots, logBsSlots_list, file, file_test, is_first, num_iter, learning_rate, num_thread, is_encrypted):
    z_data, factor_num, sample_num = SecureML.z_data_from_file(file, is_first)
    z_data = SecureML.shuffle_z_data(z_data, factor_num, sample_num)
    z_data = SecureML.normalize_z_data(z_data, factor_num, sample_num)

    params = Params(z_data.shape[1] - 1, z_data.shape[0], num_iter, learning_rate, num_thread, encode_slots)
    params.cryptoContext = cryptoContext
    params.openfhe_context = openfhe_context
    params.logBsSlots_list = logBsSlots_list
    depth = cryptoContext.L - 1
    params.depth = depth

    params.openfhe_context = openfhe_context #todo: bad assignment, to be removed
    
    params.path_to_file = file
    params.path_to_test_file = file_test
    params.isfirst = is_first

    print("Setting up crypto context...")
    params.start_time()
    secure_ml = SecureML(params, cryptoContext)

    block_num = params.sample_num // params.block_size
    block_array = [random.randint(0, block_num - 1) for _ in range(num_iter)]

    start_time = time.time()
    if is_encrypted:
        print("Encrypting data...")
        secure_ml.encrypt_z_data(cryptoContext, z_data, block_array, num_iter)
        print("Encryption finished!")
    elapsed_time = time.time() - start_time
    print(f"Encryption time: {elapsed_time:.4f} seconds")

    enc_w_data = [None] * params.cnum
    w_data = np.zeros(params.factor_num)

    start_time = time.time()
    if is_encrypted:
        secure_ml.training(cryptoContext, enc_w_data, factor_num, sample_num, w_data, z_data, block_array)
    else:
        secure_ml.plain_training(cryptoContext, w_data, z_data, factor_num, sample_num)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.4f} seconds")

    save_dwData=False
    if save_dwData:
        secure_ml.decrypt_w_data_and_save(cryptoContext, enc_w_data, factor_num, DATA_DIR + "/helr/dwData.csv")


def main():
    file1 = "./data/MNIST_train.txt"
    file2 = "./data/MNIST_test.txt"
    is_first = True
    is_encrypted = True
    num_iter = 30
    learning_rate = 1.0
    num_thread = 1

    type = "HELR" if is_encrypted else "LR"
    print(f"{type} Test with thread {num_thread}, "
          f"the var thread is aligned with the one in AAAI'19, currently is set to 1 in most case")
    print(f"Training Data = {file1}")
    print(f"Testing Data = {file2}")

    maxLevelsRemaining = 30
    logBsSlots_list = [8] 
    logN = 16
    levelBudget_list = [[4, 4]]

    encode_slots = (1 << (logN - 1))
    appRotIndex_list = []
    i = 1
    while i < encode_slots:
        appRotIndex_list.append(i)
        i *= 2

    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    cryptoContext, openfhe_context = fhe.try_load_context(
        maxLevelsRemaining,
        appRotIndex_list,
        logBsSlots_list,
        logN,
        levelBudget_list,
        save_dir=DATA_DIR,
        autoLoadAndSetConfig=True,
    )
    cryptoContext.openfhe_context = openfhe_context
        
    test(cryptoContext, openfhe_context, encode_slots, logBsSlots_list, file1, file2, is_first, num_iter, learning_rate, num_thread, is_encrypted)

if __name__ == "__main__":
    main()

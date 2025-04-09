import os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch.fhe as fhe
import torch
import math, time
import numpy as np
from torch.fhe.approx import eval_chebyshev_coefficients, eval_chebyshev_series_ps

DATA_DIR = os.environ["DATA_DIR"]
DEBUG = True
polyDeg_in_compare_and_swap = 119


def eval_chebyshev_function(function, ciphertext, lowerBound, upperBound, poly_degree, cryptoContext):
    coefficients = eval_chebyshev_coefficients(
        function, lowerBound, upperBound, poly_degree)
    result = eval_chebyshev_series_ps(
        ciphertext, coefficients, lowerBound, upperBound, cryptoContext
    )
    return result

def compare_and_swap(a1, a2, a3, a4, cryptoContext):
    a1_sub_a2 = fhe.homo_sub(a1, a2,cryptoContext)
    a2_sub_a1 = fhe.homo_sub(a2, a1,cryptoContext)
    lowerBound = -5
    upperBound = 5

    a1_gt_a2 = eval_chebyshev_function(lambda x: 1 if x>=0 else 0,
                                              a1_sub_a2,
                                              lowerBound, upperBound, polyDeg_in_compare_and_swap, cryptoContext )
    a2_gt_a1 = eval_chebyshev_function(lambda x: 1 if x>0 else 0,
                                              a2_sub_a1,
                                              lowerBound, upperBound, polyDeg_in_compare_and_swap, cryptoContext)
    return fhe.homo_add(fhe.homo_mul(a1_gt_a2,a3, cryptoContext), fhe.homo_mul(a2_gt_a1,a4, cryptoContext), cryptoContext)

def Sort(input_length=8):
    print("--------------------------------- Sorting ---------------------------------")

    maxLevelsRemaining = 34
    rotate_index_list = []
    i = 1
    while i < input_length:
        rotate_index_list.append(i)
        rotate_index_list.append(-i)
        i <<= 1 
    logBsSlots_list = [int(math.log2(input_length))]
    logN = 14
    levelBudget_list = [[2,2]]

    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Directory {DATA_DIR} does not exist!")

    cryptoContext, openfhe_context = fhe.try_load_context(
        maxLevelsRemaining,
        rotate_index_list,
        logBsSlots_list,
        logN,
        levelBudget_list,
        save_dir=DATA_DIR,
        autoLoadAndSetConfig=True,
    )
    cryptoContext.openfhe_context = openfhe_context


    total_steps = int((1+math.log2(input_length))*math.log2(input_length)/2)
    print("Total steps: {}".format(total_steps))

    input_msg = np.random.uniform(3, 4, input_length)
    input_ct = openfhe_context.encrypt(input_msg, 1, 0, input_length)
    print("Initial number of mult depth remaining: ", input_ct.cur_limbs-1)

    total_time = 0

    # Sorting
    n = input_length
    k = 2
    step = 1
    sorted_input_msg = sorted(input_msg)
    while k<=n:
        j = int(math.floor(k/2))
        while j>0:
            cur_lRemain = input_ct.cur_limbs - (input_ct.noise_deg - 1)
            if cur_lRemain <= 10:
                print("lRemain before bootstrapping: ", cur_lRemain)
                input_ct = fhe.homo_bootstrap(input_ct, cryptoContext.L, logBsSlots_list[0], cryptoContext)
                print("lRemain after bootstrapping: ", cur_lRemain)

            print("step: {}".format(step))
            print(f"[APP TRACE] step: {step} (k={k}, j={j})", file=sys.stderr)
            step += 1

            mask1 = [0.0] * n
            mask2 = [0.0] * n
            mask3 = [0.0] * n
            mask4 = [0.0] * n

            for i in range(n):
                l = i ^ j
                if i < l:
                    if (i & k) == 0:
                        mask1[i] = 1
                        mask2[l] = 1
                    else:
                        mask3[i] = 1
                        mask4[l] = 1
            mask1 = torch.tensor(mask1, dtype=torch.float64).cuda()
            mask2 = torch.tensor(mask2, dtype=torch.float64).cuda()
            mask3 = torch.tensor(mask3, dtype=torch.float64).cuda()
            mask4 = torch.tensor(mask4, dtype=torch.float64).cuda()

            mask1 = cryptoContext.openfhe_context.encode(mask1, 1, 0, input_length)
            mask2 = cryptoContext.openfhe_context.encode(mask2, 1, 0, input_length)
            mask3 = cryptoContext.openfhe_context.encode(mask3, 1, 0, input_length)
            mask4 = cryptoContext.openfhe_context.encode(mask4, 1, 0, input_length)

            start_time = time.time()
            arr1 = fhe.homo_mul_pt(input_ct, mask1, cryptoContext)
            arr2 = fhe.homo_mul_pt(input_ct, mask2, cryptoContext)
            arr3 = fhe.homo_mul_pt(input_ct, mask3, cryptoContext)
            arr4 = fhe.homo_mul_pt(input_ct, mask4, cryptoContext)
            arr5_1 = fhe.homo_rotate(arr1,-j,cryptoContext)
            arr5_2 = fhe.homo_rotate(arr3,-j,cryptoContext)
            arr6_1 = fhe.homo_rotate(arr2,j,cryptoContext)
            arr6_2 = fhe.homo_rotate(arr4,j,cryptoContext)
            arr7 = fhe.homo_add(fhe.homo_add(arr5_1,arr5_2,cryptoContext),fhe.homo_add(arr6_1,arr6_2,cryptoContext), cryptoContext)
            arr8 = input_ct.deep_copy()
            arr9 = fhe.homo_add(fhe.homo_add(arr5_1,arr1,cryptoContext), fhe.homo_add(arr6_2,arr4,cryptoContext), cryptoContext)
            arr10 = fhe.homo_add(fhe.homo_add(arr5_2,arr3, cryptoContext),fhe.homo_add(arr6_1,arr2,cryptoContext), cryptoContext)
            input_ct = compare_and_swap(arr7, arr8, arr9, arr10, cryptoContext)
            total_time += (time.time() - start_time)

            j =int(j/2)

            print("remaining level: ", input_ct.cur_limbs - (input_ct.noise_deg - 1))

            if DEBUG:
                clear_result = openfhe_context.decrypt(input_ct)
                clear_result = clear_result.cpu().numpy().reshape(-1)
                total_error = 0.0
                for i in range(n):
                    total_error += (clear_result[i]-sorted_input_msg[i])**2
                print("Avg error: ", total_error/n)

        k *= 2

    print("[APP TRACE] finish")

    print("Expected output: ", sorted_input_msg[:10])
    clear_result = openfhe_context.decrypt(input_ct)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("Actual output: ", clear_result[:10])

    print("Total time is ", total_time)

if __name__ == "__main__":
    Sort(1<<3)
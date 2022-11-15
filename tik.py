from tbe import tik
import numpy as np
from tbe.common.platform import set_current_compile_soc_info


def element_add_test():
    tik_instance = tik.Tik()
    set_current_compile_soc_info("Ascend310")
    data_A = tik_instance.Tensor("float16", (3, 5, 5, 1024), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float16", (3, 3, 3, 16), name="data_B", scope=tik.scope_gm)
    data_C = tik_instance.Tensor("float16", (3, 3, 3, 1024), name="data_C", scope=tik.scope_gm)

    data_a_ub = tik_instance.Tensor("float16", (1024,), name="data_a_ub", scope=tik.scope_ubuf)
    data_b_ub = tik_instance.Tensor("float16", (3, 3, 3, 16), name="data_b_ub", scope=tik.scope_ubuf)
    data_c_ub = tik_instance.Tensor("float16", (1024,), name="data_c_ub", scope=tik.scope_ubuf)

    # define other scope_ubuf Tensors

    tik_instance.data_move(data_b_ub, data_B, 0, 1, 3 * 3 * 3 * 16 * 2 // 32, 0, 0)

    with tik_instance.for_range(0, 3) as i0:
        with tik_instance.for_range(0, 25) as i1:
            # move data from out to UB
            move_offset_a = i0 * 25 * 1024 + i1 * 1024
            tik_instance.data_move(data_a_ub, data_A[move_offset_a], 0, 1, 1024 * 2 // 32, 0, 0)
            pass

            # calculate with TIK API
            tik_instance.vec_mul(128, data_c_ub, data_a_ub, data_b_ub, 8, 8, 8, 0)
            pass

            # move data from UB to OUT

            pass

    tik_instance.BuildCCE(kernel_name="element_add_test", inputs=[data_A, data_B], outputs=[data_C])

    return tik_instance




def compareData(src, exp, print_n):
    # pass
    totolNum = src.reshape(-1).shape[0]
    errorCnt = 0
    for i in range(totolNum):
        if (abs(src[i] - exp[i])) / abs(exp[i]) > 0.01:
            if i < print_n or print_n == 0:
                print("loc:", i, "src:", str(src[i]), "exp:", str(exp[i]))
            errorCnt = errorCnt + 1
        elif i < print_n:
            print("loc:", i, "src:", str(src[i]), "exp:", str(exp[i]))
    print("Is allclose:", (str(np.allclose(src.reshape(-1), exp.reshape(-1), atol=0.1, rtol=0.1))))
    print("Total Num:", totolNum, "error cnt:", errorCnt, "error percent:", float(errorCnt) / float(totolNum))
    if errorCnt > 0:
        print("compare falied")
    else:
        print("compare success")


if __name__ == "__main__":
    tik_instance = element_add_test()

    dataA = np.random.uniform(1, 10, (16, 16, 16, 16, 16)).astype(np.float16)
    dataB = np.random.uniform(1, 10, (16, 16, 16, 16, 16)).astype(np.float16)
    # dataA = np.ones((16, 16, 16, 16, 16), dtype=np.float16)
    # dataB = np.ones((16, 16, 16, 16, 16), dtype=np.float16)
    dataC = dataA + dataB
    feed_dict = {"data_A": dataA, "data_B": dataB}
    data_C, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

    dataC = dataC.reshape(-1)
    data_C = data_C.reshape(-1)

    compareData(data_C, dataC, 5)

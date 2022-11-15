from tbe import tik
import numpy as np
from tbe.common.platform import set_current_compile_soc_info


def element_add_test():
    tik_instance = tik.Tik()
    set_current_compile_soc_info("Ascend310")
    data_A = tik_instance.Tensor("float16", (10, 10, 256), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float16", (5, 5, 16), name="data_B", scope=tik.scope_gm)
    data_C = tik_instance.Tensor("float16", (6, 6, 256), name="data_C", scope=tik.scope_gm)

    with tik_instance.for_range(0, 2, block_num=2) as i0:
        # define other scope_ubuf Tensors
        data_a_ub = tik_instance.Tensor("float16", (10, 10, 128), name="data_a_ub", scope=tik.scope_ubuf)
        data_b_ub = tik_instance.Tensor("float16", (5, 5, 16), name="data_b_ub", scope=tik.scope_ubuf)
        data_c_ub = tik_instance.Tensor("float16", (1, 1, 128), name="data_c_ub", scope=tik.scope_ubuf)
        data_zero_ub = tik_instance.Tensor("float16", (1, 1, 128), name="data_zero_ub", scope=tik.scope_ubuf)

        # move data from out to UB
        tik_instance.data_move(data_b_ub, data_B, 0, 1, 25, 0, 0)
        tik_instance.data_move(data_a_ub, data_A[i0 * 128], 0, 100, 8, 8, 0)

        for h in range(6):
            for w in range(6):
                tik_instance.vec_dup(128, data_zero_ub, 0, 1, 8)

                for i in range(5):
                    for j in range(5):
                        tik_instance.vec_mul(16, data_c_ub, data_a_ub[(h + i) * 10 * 128 + (w + j) * 128],
                                             data_b_ub[i * 5 * 16 + j * 16], 8, 1, 1, 0)
                        tik_instance.vec_add(128, data_zero_ub, data_c_ub, data_zero_ub, 1, 8, 8, 8)
                tik_instance.data_move(data_C[h * 6 * 256 + w * 256 + i0 * 128], data_zero_ub, 0, 1, 8, 0, 0)

    tik_instance.BuildCCE(kernel_name="element_add_test", inputs=[data_A, data_B], outputs=[data_C])

    return tik_instance


def np_compute(inputs, weight):
    output = np.zeros((256, 6, 6))
    inputs = inputs.transpose((2, 0, 1))
    for n in range(256):
        for h in range(6):
            for w in range(6):
                temp = 0
                for i in range(5):
                    for j in range(5):
                        temp += inputs[n][h + i][w + j] * weight[i][j]
                output[n][h][w] = temp
    output = output.transpose((1, 2, 0))
    return output


def compareData(src, exp, print_n):
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

    dataA = np.random.uniform(1, 10, (10, 10, 256)).astype(np.float16)
    dataB_ori = np.random.uniform(1, 10, (5, 5)).astype(np.float16)
    dataB = np.expand_dims(dataB_ori, -1)
    dataB = np.tile(dataB, 16)

    dataC = np_compute(dataA, dataB_ori)

    feed_dict = {"data_A": dataA, "data_B": dataB}
    data_C, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

    dataC = dataC.reshape(-1)
    data_C = data_C.reshape(-1)

    compareData(data_C, dataC, 5)

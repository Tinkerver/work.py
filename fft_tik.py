from tbe import tik
import numpy as np
from tbe.common.platform import set_current_compile_soc_info


def element_fft_test():
    tik_instance = tik.Tik()
    set_current_compile_soc_info("Ascend310")
    data_A = tik_instance.Tensor("float32", (1024,), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float32", (1024,), name="data_B", scope=tik.scope_gm)

    # define other scope_ubuf Tensors
    data_a_ub = tik_instance.Tensor("float32", (1024,), name="data_a_ub", scope=tik.scope_ubuf)  # 输入序列
    data_b_ub = tik_instance.Tensor("float32", (1024,), name="data_b_ub", scope=tik.scope_ubuf)  # w系数因子实部
    data_b_i_ub = tik_instance.Tensor("float32", (1024,), name="data_b_i_ub", scope=tik.scope_ubuf)  # w系数因子虚部
    data_c_ub = tik_instance.Tensor("float32", (1024,), name="data_c_ub", scope=tik.scope_ubuf)  # 输出实部
    data_c_i_ub = tik_instance.Tensor("float32", (1024,), name="data_c_i_ub", scope=tik.scope_ubuf)  # 输出虚部
    tmp = tik_instance.Tensor("float32", (1024,), name="tmp_ub", scope=tik.scope_ubuf)  # 暂存实部
    tmp_i = tik_instance.Tensor("float32", (1024,), name="tmp_i_ub", scope=tik.scope_ubuf)  # 暂存虚部
    tmp_2 = tik_instance.Tensor("float32", (1024,), name="tmp_2_ub", scope=tik.scope_ubuf)  # 暂存实部
    tmp_2_i = tik_instance.Tensor("float32", (1024,), name="tmp_2_i_ub", scope=tik.scope_ubuf)  # 暂存虚部

    # move data from out to UB
    tik_instance.data_move(data_b_ub, data_B, 0, 1, 128, 0, 0)
    tik_instance.data_move(data_b_i_ub, data_B, 0, 1, 128, 0, 0)
    tik_instance.data_move(data_a_ub, data_A, 0, 1, 128, 0, 0)
    tik_instance.data_move(data_c_ub, data_A, 0, 1, 128, 0, 0)
    tik_instance.data_move(data_c_i_ub, data_A, 0, 1, 128, 0, 0)

    N = 1024  # 数据总数
    total_m = 10  # 序列总层数

    for m in range(total_m):
        _split = N // 2 ** m  # 计算蝶形组数
        num_each = N // _split  # 一组鲽形中有多少条
        data_b_l_ub = tik_instance.Tensor("float32", (512,), name="data_b_l_ub", scope=tik.scope_ubuf)  # w系数列表实部
        data_b_i_l_ub = tik_instance.Tensor("float32", (512,), name="data_b_i_l_ub", scope=tik.scope_ubuf)  # w系数列表虚部
        for i in range(512):
            data_b_l_ub[i] = data_b_ub[i % (2 ** m)]
            data_b_i_l_ub[i] = data_b_i_ub[i % (2 ** m)]
        tik_instance.vec_mul(64, tmp, data_c_ub[512], data_b_l_ub, 8, 8, 8, 8)
        tik_instance.vec_mul(64, tmp_2, data_c_i_ub[512], data_b_i_l_ub, 8, 8, 8, 8)
        tik_instance.vec_mul(64, tmp_i, data_c_i_ub[512], data_b_l_ub, 8, 8, 8, 8)
        tik_instance.vec_mul(64, tmp_2_i, data_c_ub[512], data_b_i_l_ub, 8, 8, 8, 8)

        tik_instance.vec_add(64, data_c_i_ub[512], tmp_i, tmp_2_i, 8, 8, 8, 8)
        tik_instance.vec_sub(64, data_c_ub[512], tmp, tmp_2, 8, 8, 8, 8)

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

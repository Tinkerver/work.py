from tbe import tik
import numpy as np
from tbe.common.platform import set_current_compile_soc_info


def element_add_test():
    tik_instance = tik.Tik()
    set_current_compile_soc_info("Ascend310")
    data_A = tik_instance.Tensor("float16", (20, 20, 512), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float16", (5, 5, 16), name="data_B", scope=tik.scope_gm)
    data_C = tik_instance.Tensor("float16", (16, 16, 512), name="data_C", scope=tik.scope_gm)

    data_a_ub = tik_instance.Tensor("float16", (20, 20, 256), name="data_a_ub", scope=tik.scope_ubuf)
    data_b_ub = tik_instance.Tensor("float16", (5, 5, 16), name="data_b_ub", scope=tik.scope_ubuf)
    data_c_ub = tik_instance.Tensor("float16", (1, 1, 256), name="data_c_ub", scope=tik.scope_ubuf)

    # define other scope_ubuf Tensors

    tik_instance.data_move(data_b_ub, data_B, 0, 1, 25, 0, 0)
    data_zero_ub = tik_instance.Tensor("float16", (1, 1, 256), name="data_zero_ub", scope=tik.scope_ubuf)
    with tik_instance.for_range(0, 2, block_num=2) as i0:
        tik_instance.data_move(data_a_ub, data_A[i0 * 20 * 20 * 256], 0, 1, 8, 0, 0)

        for h in range(16):
            for w in range(16):
                # 赋值0
                for i in range(5):
                    for j in range(5):
                        print(i, j)
                        # tik_instance.vec_mul(128,data_c_ub, data_a_ub[h+i,w+j,:], data_b_ub[i,j:], 2, 8, 8, 0)
                        tik_instance.vec_mul(16, data_c_ub, data_a_ub[(h + i) * 20 * 256 + (w + j) * 256],
                                             data_b_ub[i * 5 * 16 + j * 16], 16, 1, 1, 0)
                        tik_instance.vec_add(128, data_zero_ub, data_c_ub, data_zero_ub, 2, 8, 8, 8)
                tik_instance.data_move(data_C[h, w, i0 * 256:], data_zero_ub, 0, 1, 16, 0, 0)

    tik_instance.BuildCCE(kernel_name="element_add_test", inputs=[data_A, data_B], outputs=[data_C])

    return tik_instance


def compareData(src, exp, print_n):
    # pass
    totolNum = src.reshape(-1).shape[0]
    errorCnt = 0

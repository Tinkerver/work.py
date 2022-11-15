import math
from te import tik
import numpy as np
from te import platform as tbe_platform
from tbe.common.utils import shape_util as tbe_shape_util
from tbe.dsl.instrinsic import cce_intrin as tbe_cce_intrin


class bit_reverse():
    def __init__(self, input_r, input_i, input_w, kernel_name="bit_reverse"):
        self.shape = input_r.get("shape")  # (n, 1)
        self.dtype = input_r.get("dtype")  # fp16
        block_bite_size = 32  # tik中一个block的大小为32B
        dtype_bytes_size = tbe_cce_intrin.get_bit_len(self.dtype) // 8  # 查看一个数据占多少字节。因为是float16，所以变量为2
        self.data_each_block = block_bite_size // dtype_bytes_size  # 一个block中可以容纳16个float16
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.input_num = tbe_shape_util.get_shape_size(self.shape)  # 数据总数，即为n
        self.input_r_gm = self.tik_instance.Tensor(  # global_memory上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="input_r_gm", scope=tik.scope_gm)
        self.input_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_i_gm", scope=tik.scope_gm)
        self.input_matrix_gm = self.tik_instance.Tensor(  # 转移矩阵
            self.dtype, (256, 256,), name="input_matrix_gm", scope=tik.scope_gm)

        self.reverse_unit = 16
        self.reverse_size = 256

        self.output_r_gm = self.tik_instance.Tensor(  # global_memory上声明输输出数据,因为数据由fixpipe得到，原始格式不一致，需转存
            self.dtype, (self.reverse_size,), name="output_r_gm", scope=tik.scope_gm)
        self.output_i_gm = self.tik_instance.Tensor(
            self.dtype, (self.reverse_size,), name="output_i_gm", scope=tik.scope_gm)

        self.output_r_ub = self.tik_instance.Tensor(  # unified_buffer上声明输输出数据,out不能直接输出out，需转存
            self.dtype, (self.reverse_size,), name="output_r_ub", scope=tik.scope_ubuf)
        self.output_i_ub = self.tik_instance.Tensor(
            self.dtype, (self.reverse_size,), name="output_i_ub", scope=tik.scope_ubuf)

    def reverse_compute(self):
        loop_time = self.input_num // self.reverse_size
        burst_len = self.reverse_unit // self.data_each_block
        stride = (self.input_num // self.reverse_unit - self.reverse_unit) // self.data_each_block
        with self.tik_instance.for_range(0, loop_time) as loop_index:
            self.input_r_cb = self.tik_instance.Tensor(  # 实部
                self.dtype, (self.reverse_size // 16, 16, 16),
                name="input_r_cb",
                scope=tik.scope_cbuf)
            self.input_i_cb = self.tik_instance.Tensor(  # 虚部
                self.dtype, (self.reverse_size // 16, 16, 16),
                name="input_i_cb",
                scope=tik.scope_cbuf)
            self.matrix_cb = self.tik_instance.Tensor(  # 转移矩阵
                self.dtype, (self.reverse_size // 16, self.reverse_size, 16),
                name="matrix_cb",
                scope=tik.scope_cbuf)
            self.dst_r_cb = self.tik_instance.Tensor(  # 实部
                "float32", (self.reverse_size // 16, 16, 16),
                name="dst_r_cb_out",
                scope=tik.scope_cbuf)
            self.dst_i_cb = self.tik_instance.Tensor(  # 虚部
                "float32", (self.reverse_size // 16, 16, 16),
                name="dst_i_cb_out",
                scope=tik.scope_cbuf)

            self.tik_instance.data_move(self.input_r_cb, self.input_r_gm[loop_index * self.reverse_unit],
                                        0, self.reverse_unit, burst_len, stride, 0)  # 将需要反转的数据实部输入
            self.tik_instance.data_move(self.input_i_cb, self.input_i_gm[loop_index * self.reverse_unit],
                                        0, self.reverse_unit, burst_len, stride, 0)  # 将需要反转的数据虚部输入
            self.tik_instance.data_move(self.matrix_cb, self.input_matrix_gm,  # 将转移矩阵输入
                                        0, 1, self.reverse_size // self.data_each_block, 0, 0)
            # 进行矩阵乘法，完成顺序转换
            self.tik_instance.matmul(self.dst_r_cb, self.input_r_cb, self.matrix_cb, 1, 256, 256)
            self.tik_instance.matmul(self.dst_i_cb, self.input_i_cb, self.matrix_cb, 1, 256, 256)

            # 将结果输出到gm，同时从32量化至16
            self.tik_instance.fixpipe(self.output_r_gm, self.dst_r_cb, 16, 2, 0, 0,
                                      extend_params={"bias": None,
                                                     "quantize_params": {"mode": "fp322fp16", "mode_param": None}})
            self.tik_instance.fixpipe(self.output_i_gm, self.dst_r_cb, 16, 2, 0, 0,
                                      extend_params={"bias": None,
                                                     "quantize_params": {"mode": "fp322fp16", "mode_param": None}})

            # 输出回对应位置
            self.tik_instance.data_move(self.output_r_ub, self.output_r_gm,
                                        0, 1, self.reverse_size // self.data_each_block, 0, 0)
            self.tik_instance.data_move(self.output_i_ub, self.output_i_gm,
                                        0, 1, self.reverse_size // self.data_each_block, 0, 0)

            self.tik_instance.data_move(self.input_r_gm[loop_index * self.reverse_unit], self.output_r_ub,
                                        0, self.reverse_unit, burst_len, 0, stride)
            self.tik_instance.data_move(self.input_i_gm[loop_index * self.reverse_unit], self.output_i_ub,
                                        0, self.reverse_unit, burst_len, 0, stride)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_r_gm, self.input_i_gm, self.input_matrix_gm],
                                   outputs=[self.input_r_gm, self.input_i_gm])
        return self.tik_instance


if __name__ == '__main__':
    data_x = np.ones((512,)).astype("float16")
    feed_dict = {'src_gm': data_x}
    model_data, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    m = np.array([[0 for i in range(64)] for j in range(64)])  # 用于得到位反转矩阵
    reverse_list = [0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240, 8, 136, 72, 200, 40, 168,
                    104, 232, 24, 152, 88, 216, 56, 184, 120, 248, 4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212,
                    52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252, 2, 130,
                    66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 10, 138, 74, 202, 42, 170, 106, 234,
                    26, 154, 90, 218, 58, 186, 122, 250, 6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182,
                    118, 246, 14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254, 1, 129, 65, 193,
                    33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241, 9, 137, 73, 201, 41, 169, 105, 233, 25, 153,
                    89, 217, 57, 185, 121, 249, 5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
                    13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253, 3, 131, 67, 195, 35, 163,
                    99, 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219,
                    59, 187, 123, 251, 7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247, 15, 143,
                    79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255]
    for i in range(256):
        m[i][reverse_list[i]] = 1
    print(model_data)

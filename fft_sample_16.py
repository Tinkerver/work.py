import math
from te import tik
import numpy as np
from te import platform as tbe_platform
from tbe.common.utils import shape_util as tbe_shape_util
from tbe.dsl.instrinsic import cce_intrin as tbe_cce_intrin
from tbe.common.platform import set_current_compile_soc_info


class fft_c():
    def __init__(self, input_r, input_i, input_w, kernel_name="fft"):
        self.shape = input_r.get("shape")  # (n, 1)
        self.dtype = input_r.get("dtype")  # fp16
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()

        self.input_num = tbe_shape_util.get_shape_size(self.shape)  # 数据总数，即为n
        self.level = int(math.log2(self.input_num))  # 鲽形展开后得到的计算层数
        block_bite_size = 32  # tik中一个block的大小为32B
        dtype_bytes_size = tbe_cce_intrin.get_bit_len(self.dtype) // 8  # 查看一个数据占多少字节。因为是float16，所以变量为2
        self.data_each_block = block_bite_size // dtype_bytes_size  # 一个block中可以容纳16个float16
        # self.ub_tensor_size = 8 * 1024  # 将数据总数分割为8K个数，16KB大小，因为分实部虚部且需要若干辅助空间，ub总容量仅248KB
        self.vector_mask_max = 8 * self.data_each_block  # 用作向量计算的最大掩码，最高128

        self.input_r_gm = self.tik_instance.Tensor(  # global_memory上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="input_r_gm", scope=tik.scope_gm)
        self.input_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_i_gm", scope=tik.scope_gm)
        self.output_r_gm = self.tik_instance.Tensor(  # global_memory上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="output_r_gm", scope=tik.scope_gm)
        self.output_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="output_i_gm", scope=tik.scope_gm)
        self.input_w_gm = self.tik_instance.Tensor(  # global_memory上声明w权重，前一半为实部，后一半为虚部
            self.dtype, self.shape, name="input_w_gm", scope=tik.scope_gm)

    def fft_compute(self):
        input_r_ub = self.tik_instance.Tensor(  # unified_buffer上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="input_r_ub", scope=tik.scope_ubuf)
        input_i_ub = self.tik_instance.Tensor(  # unified_buffer上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="input_i_ub", scope=tik.scope_ubuf)
        input_w_ub = self.tik_instance.Tensor(  # unified_buffer上声明w权重，前一半为实部，后一半为虚部
            self.dtype, self.shape, name="input_w_ub", scope=tik.scope_ubuf)

        tmp_a_ub = self.tik_instance.Tensor(  # unified_buffer上声明中间变量
            self.dtype, self.shape, name="tmp_a_ub", scope=tik.scope_ubuf)
        tmp_b_ub = self.tik_instance.Tensor(  # unified_buffer上声明中间变量
            self.dtype, self.shape, name="tmp_b_ub", scope=tik.scope_ubuf)

        burst_len = self.input_num // self.data_each_block

        self.tik_instance.data_move(input_w_ub, self.input_w_gm, 0, 1, burst_len, 0, 0)  # 读取相应权重
        self.tik_instance.data_move(input_r_ub, self.input_r_gm, 0, 1, burst_len, 0, 0)  # 读取数据实部
        self.tik_instance.data_move(input_i_ub, self.input_i_gm, 0, 1, burst_len, 0, 0)  # 读取数据虚部

        with self.tik_instance.for_range(0, 4) as loop_index:
            self.tik_instance.vec_add(self.data_each_block, input_r_ub, input_r_ub,  # 鲽形操作，得到上半蝶形单元实部结果
                                      input_r_ub, 1, 8, 8, 8)
            self.tik_instance.vec_add(self.data_each_block, input_i_ub, input_i_ub,  # 鲽形操作，得到上半蝶形单元虚部结果
                                      input_i_ub, 1, 8, 8, 8)

            self.tik_instance.vec_sub(self.data_each_block, tmp_a_ub, input_r_ub,
                                      input_r_ub, 1, 8, 8, 8)
            self.tik_instance.vec_sub(self.data_each_block, tmp_b_ub, input_i_ub,
                                      input_i_ub, 1, 8, 8, 8)

            self.tik_instance.vec_mul(self.data_each_block, input_r_ub, tmp_a_ub,  # 复数乘法实部相乘
                                      input_w_ub, 1, 8, 8, 8)
            self.tik_instance.vec_mul(self.data_each_block, input_i_ub, tmp_b_ub,  # 复数乘法虚部相乘
                                      input_w_ub, 1, 8, 8, 8)

            self.tik_instance.vec_mul(self.data_each_block, tmp_a_ub, tmp_a_ub,  # 复数乘法数据实部权重虚部相乘
                                      input_w_ub, 1, 8, 8, 8)
            self.tik_instance.vec_mul(self.data_each_block, tmp_b_ub, tmp_b_ub,  # 复数乘法数据虚部权重实部相乘
                                      input_w_ub, 1, 8, 8, 8)

            self.tik_instance.vec_sub(self.data_each_block, input_r_ub, input_r_ub,
                                      input_i_ub, 1, 8, 8, 8)
            self.tik_instance.vec_add(self.data_each_block, input_i_ub, tmp_a_ub,
                                      tmp_b_ub, 1, 8, 8, 8)

        self.tik_instance.data_move(self.output_r_gm, input_r_ub, 0, 1, burst_len, 0, 0)  # 读取数据实部
        self.tik_instance.data_move(self.output_i_gm, input_i_ub, 0, 1, burst_len, 0, 0)  # 读取数据虚部

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_r_gm, self.input_i_gm, self.input_w_gm],
                                   outputs=[self.output_i_gm, self.output_r_gm])
        return self.tik_instance


def fft(input_r, input_i, input_w, output_r, output_i, kernel_name="fft"):
    fft_instance = fft_c(input_r, input_i, input_w, kernel_name)
    tik_instance = fft_instance.fft_compute()
    return tik_instance


if __name__ == '__main__':
    set_current_compile_soc_info('Ascend910B1', 'AiCore')

    fft_size = 16
    input_r_data = np.random.rand(fft_size, )
    input_i_data = np.random.rand(fft_size, )
    input_r_data = input_r_data.astype("float16")
    input_i_data = input_i_data.astype("float16")
    input_w_data = np.empty([fft_size, ], dtype="float16")

    output_w = output_r = output_i = input_i = input_r = input_w = temp_uw = {
        "dtype": "float16",
        "shape": (fft_size,)}

    for i in range(fft_size // 2):
        input_w_data[i] = math.cos(i * math.pi / fft_size)
        input_w_data[i + fft_size // 2] = math.sin(i * math.pi / fft_size)

    kernel_name = "fft_sample"
    fft_instance = fft_c(input_r, input_i, input_w, kernel_name)
    tik_instance = fft_instance.fft_compute()
    feed_dict = {'input_r_gm': input_r_data, 'input_i_gm': input_i_data, 'input_w_gm': input_w_data}

    tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

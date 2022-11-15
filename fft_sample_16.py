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

    def fft_compute(self):
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

    fft_size = 1024 * 1024
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

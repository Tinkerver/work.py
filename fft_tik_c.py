import math
from te import tik
import numpy as np
from te import platform as tbe_platform
from tbe.common.utils import shape_util as tbe_shape_util
from tbe.dsl.instrinsic import cce_intrin as tbe_cce_intrin


class fft():
    def __init__(self, input_r, input_i, input_w, kernel_name="fft"):
        self.shape = input_r.get("shape")  # (n, 1)
        self.dtype = input_r.get("dtype")  # fp16
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.aicore_num = 2  # 分为双核运行
        self.input_num = tbe_shape_util.get_shape_size(self.shape)  # 数据总数，即为n
        self.level = int(math.log2(self.input_num))  # 鲽形展开后得到的计算层数
        block_bite_size = 32  # tik中一个block的大小为32B
        dtype_bytes_size = tbe_cce_intrin.get_bit_len(self.dtype) // 8  # 查看一个数据占多少字节。因为是float16，所以变量为2
        self.data_each_block = block_bite_size // dtype_bytes_size  # 一个block中可以容纳16个float16
        self.ub_tensor_size = 8 * 1024  # 将数据总数分割为8K个数，16KB大小，因为分实部虚部且需要若干辅助空间，ub总容量仅248KB
        self.vector_mask_max = 8 * self.data_each_block  # 用作向量计算的最大掩码，最高128
        self.data_num_each_core = self.input_num // self.aicore_num  # 分核操作数据
        self.input_r_gm = self.tik_instance.Tensor(  # global_memory上声明输入数据和输出数据，分实部虚部
            self.dtype, self.shape, name="input_r_gm", scope=tik.scope_gm)
        self.input_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_i_gm", scope=tik.scope_gm)
        self.input_w_gm = self.tik_instance.Tensor(  # global_memory上声明w权重，前一半为实部，后一半为虚部
            self.dtype, self.shape, name="input_w_gm", scope=tik.scope_gm)
        self.output_w_gm = self.tik_instance.Tensor(
            self.dtype, (16, 1), name="output_w_gm", scope=tik.scope_gm)

    def fft_compute(self):
        endLevel = self.level - int(math.log2(self.ub_tensor_size))  # 处理大于一次ub能够完成的模块
        for level in range(0, endLevel):
            wnum = pow(2, level)  # 此层下有几个蝶形模块
            rnum = self.input_num // wnum  # 碟形模块规模,有多少个float16
            loop_time = self.input_num // self.ub_tensor_size // wnum // 2  # 在这一层下，处理一个鲽形单元需要几次进出ub

            with self.tik_instance.for_range(0, loop_time) as loop_index:  # 这里的循环为在一层下，取上半部分蝶形单元
                # Unified Buffer中定义tensor
                self.input_r_ub = self.tik_instance.Tensor(  # 实部 需要两倍空间，存放需要用于计算的两部分数
                    self.dtype, (self.ub_tensor_size * 2,),
                    name="input_r_ub",
                    scope=tik.scope_ubuf)
                self.input_i_ub = self.tik_instance.Tensor(  # 虚部 需要两倍空间，存放需要计算的两部分数
                    self.dtype, (self.ub_tensor_size * 2,),
                    name="input_i_ub",
                    scope=tik.scope_ubuf)
                self.input_w_ub = self.tik_instance.Tensor(  # 权重 需要两倍空间，分别存放权重的实部虚部
                    self.dtype, (self.ub_tensor_size * 2,),
                    name="input_w_ub",
                    scope=tik.scope_ubuf)
                self.temp_a_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size,),
                    name="temp_a_ub",
                    scope=tik.scope_ubuf)
                self.temp_b_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size,),
                    name="temp_b_ub",
                    scope=tik.scope_ubuf)
                self.temp_c_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size,),
                    name="temp_c_ub",
                    scope=tik.scope_ubuf)
                self.temp_d_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size,),
                    name="temp_d_ub",
                    scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, wnum) as wnum_index:  # 分别处理每一个蝶形模块
                    burst_len = self.ub_tensor_size // self.data_each_block  # 一次进出UB能处理多少block，应该是0.5k
                    if level > 0:
                        self.tik_instance.data_move(self.input_w_ub, self.input_w_gm[loop_index * self.ub_tensor_size],
                                                    0, 1, burst_len, 0, 0)  # 在不是第一层时，分别搬运一次进出的w
                        self.tik_instance.data_move(self.input_w_ub[self.ub_tensor_size],
                                                    self.input_w_gm[
                                                        loop_index * self.ub_tensor_size + self.input_num // 2],
                                                    0, 1, burst_len, 0, 0)
                    # for aicore_index in range(self.aicore_num):
                    mask = self.vector_mask_max  # 取出最大的掩码值，对于float16应该是128
                    st = wnum_index * rnum + loop_index * self.ub_tensor_size  # 定位到模块下一次进出所对应的数据
                    move_offset = rnum // 2  # 设定偏移为蝶形模块规模的一半

                    self.tik_instance.data_move(self.input_r_ub, self.input_r_gm[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_ub[self.ub_tensor_size],
                                                self.input_r_gm[st + move_offset], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub, self.input_i_gm[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub[self.ub_tensor_size],
                                                self.input_i_gm[st + move_offset], 0, 1, burst_len, 0, 0)

                    # 蝶形运算，完成对一组鲽形单元的计算。
                    self.tik_instance.vsub(mask, self.temp_a_ub, self.input_r_ub,
                                           self.input_r_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.temp_b_ub, self.input_i_ub,
                                           self.input_i_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub, self.input_r_ub,
                                           self.input_r_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_i_ub, self.input_i_ub,
                                           self.input_i_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_a_ub,
                                           self.input_w_ub, self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_b_ub,
                                           self.input_w_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub[self.ub_tensor_size], self.temp_c_ub,
                                           self.temp_d_ub, self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_b_ub,
                                           self.input_w_ub, self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_a_ub,
                                           self.input_w_ub[self.ub_tensor_size],
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.input_i_ub[self.ub_tensor_size],
                                           self.temp_c_ub, self.temp_d_ub,
                                           self.ub_tensor_size // mask, 1, 1, 1, 8, 8, 8)

                    self.tik_instance.data_move(self.input_r_gm[st], self.input_r_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_gm[st + move_offset // 2],
                                                self.input_r_ub[self.ub_tensor_size], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_gm[st], self.input_i_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_gm[st + move_offset // 2],
                                                self.input_i_ub[self.ub_tensor_size], 0, 1, burst_len, 0, 0)

        # 以上，按层数完成了一次进出ub无法完成计算的碟形单元的蝶形计算
        # 以下，直接按块将计算推进到最终结果。

        endLevel2 = int(math.log2(self.ub_tensor_size)) - int(math.log2(self.vector_mask_max))  # 处理mask取最大即可完成操作的层数
        parts = self.input_num // self.ub_tensor_size

        for part_num in range(parts):
            st = part_num * self.ub_tensor_size
            burst_len = self.ub_tensor_size // self.data_each_block  # 一次进出UB能处理多少block，应该是0.5k
            self.input_r_ub_all = self.tik_instance.Tensor(  # 实部 需要两倍空间，存放需要用于计算的两部分数
                self.dtype, (self.ub_tensor_size,),
                name="input_r_ub",
                scope=tik.scope_ubuf)
            self.input_i_ub_all = self.tik_instance.Tensor(  # 虚部 需要两倍空间，存放需要计算的两部分数
                self.dtype, (self.ub_tensor_size,),
                name="input_i_ub",
                scope=tik.scope_ubuf)
            self.input_w_ub_all = self.tik_instance.Tensor(  # 权重 需要两倍空间，分别存放权重的实部虚部
                self.dtype, (self.ub_tensor_size,),
                name="input_w_ub",
                scope=tik.scope_ubuf)

            self.tik_instance.data_move(self.input_r_ub_all, self.input_r_gm[st], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.input_i_ub_all, self.input_i_gm[st], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.input_w_ub_all, self.input_w_gm[st], 0, 1, burst_len, 0, 0)
            # 以上，完成数据搬运。

            for level in range(endLevel2):
                wnum = pow(2, level)  # 处理几个碟形单元
                rnum = self.ub_tensor_size // wnum  # 碟形单元有多大
                block_size = self.ub_tensor_size // wnum // 2

                self.input_r_ub = self.tik_instance.Tensor(  # 实部 需要两倍空间，存放需要用于计算的两部分数(上部和下部）
                    self.dtype, (block_size * 2,),
                    name="input_r_ub",
                    scope=tik.scope_ubuf)
                self.input_i_ub = self.tik_instance.Tensor(  # 虚部 需要两倍空间，存放需要计算的两部分数
                    self.dtype, (block_size * 2,),
                    name="input_i_ub",
                    scope=tik.scope_ubuf)
                self.input_w_ub = self.tik_instance.Tensor(  # 权重 需要两倍空间，分别对应权重？
                    self.dtype, (block_size * 2,),
                    name="input_w_ub",
                    scope=tik.scope_ubuf)
                self.temp_a_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_a_ub",
                    scope=tik.scope_ubuf)
                self.temp_b_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_b_ub",
                    scope=tik.scope_ubuf)
                self.temp_c_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_c_ub",
                    scope=tik.scope_ubuf)
                self.temp_d_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_d_ub",
                    scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, wnum) as wnum_index:  # 对逐个蝶形单元进行处理
                    mask = self.vector_mask_max  # 取出最大的掩码值，对于float16应该是128
                    st = wnum_index * rnum
                    move_offset = rnum // 2
                    burst_len = block_size // self.data_each_block

                    self.tik_instance.data_move(self.input_r_ub, self.input_r_ub_all[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_ub[block_size],
                                                self.input_r_ub_all[st + move_offset], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub, self.input_i_ub_all[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub[block_size],
                                                self.input_i_ub_all[st + move_offset], 0, 1, burst_len, 0, 0)

                    # 蝶形运算，完成对一组鲽形单元的计算。
                    self.tik_instance.vsub(mask, self.temp_a_ub, self.input_r_ub,
                                           self.input_r_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.temp_b_ub, self.input_i_ub,
                                           self.input_i_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub, self.input_r_ub,
                                           self.input_r_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_i_ub, self.input_i_ub,
                                           self.input_i_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_a_ub,
                                           self.input_w_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_b_ub,
                                           self.input_w_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub[block_size], self.temp_c_ub,
                                           self.temp_d_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_b_ub,
                                           self.input_w_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_a_ub,
                                           self.input_w_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.input_i_ub[block_size],
                                           self.temp_c_ub, self.temp_d_ub,
                                           block_size // mask, 1, 1, 1, 8, 8, 8)

                    self.tik_instance.data_move(self.input_r_ub_all[st], self.input_r_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_ub_all[st + move_offset // 2],
                                                self.input_r_ub[block_size], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub_all[st], self.input_i_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub_all[st + move_offset // 2],
                                                self.input_i_ub[block_size], 0, 1, burst_len, 0, 0)

            # 以上完成了对于大于128*2=256长度的蝶形单元的层数，这些层数上不需要操作mask
            # 以下对256-16长度的蝶形单元进行鲽形操作

            endLevel3 = int(math.log2(self.vector_mask_max)) - int(math.log2(16))
            for level in range(endLevel3):
                wnum = pow(2, level + endLevel2)  # 处理几个碟形单元
                rnum = self.ub_tensor_size // wnum  # 碟形单元有多少float16
                block_size = self.ub_tensor_size // wnum // 2  # 单次计算应用空间

                self.input_r_ub = self.tik_instance.Tensor(  # 实部 需要两倍空间，存放需要用于计算的两部分数(上部和下部）
                    self.dtype, (block_size * 2,),
                    name="input_r_ub",
                    scope=tik.scope_ubuf)
                self.input_i_ub = self.tik_instance.Tensor(  # 虚部 需要两倍空间，存放需要计算的两部分数
                    self.dtype, (block_size * 2,),
                    name="input_i_ub",
                    scope=tik.scope_ubuf)
                self.input_w_ub = self.tik_instance.Tensor(  # 权重 需要两倍空间，分别对应权重？
                    self.dtype, (block_size * 2,),
                    name="input_w_ub",
                    scope=tik.scope_ubuf)
                self.temp_a_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_a_ub",
                    scope=tik.scope_ubuf)
                self.temp_b_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_b_ub",
                    scope=tik.scope_ubuf)
                self.temp_c_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_c_ub",
                    scope=tik.scope_ubuf)
                self.temp_d_ub = self.tik_instance.Tensor(
                    self.dtype, (block_size,),
                    name="temp_d_ub",
                    scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, wnum) as wnum_index:  # 对逐个蝶形单元进行处理
                    mask = rnum // 2
                    st = wnum_index * rnum
                    move_offset = rnum // 2
                    burst_len = block_size // self.data_each_block

                    self.tik_instance.data_move(self.input_r_ub, self.input_r_ub_all[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_ub[block_size],
                                                self.input_r_ub_all[st + move_offset], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub, self.input_i_ub_all[st], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub[block_size],
                                                self.input_i_ub_all[st + move_offset], 0, 1, burst_len, 0, 0)

                    # 蝶形运算，完成对一组鲽形单元的计算。
                    self.tik_instance.vsub(mask, self.temp_a_ub, self.input_r_ub,
                                           self.input_r_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.temp_b_ub, self.input_i_ub,
                                           self.input_i_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub, self.input_r_ub,
                                           self.input_r_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_i_ub, self.input_i_ub,
                                           self.input_i_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_a_ub,
                                           self.input_w_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_b_ub,
                                           self.input_w_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(mask, self.input_r_ub[block_size], self.temp_c_ub,
                                           self.temp_d_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_c_ub, self.temp_b_ub,
                                           self.input_w_ub, block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vmul(mask, self.temp_d_ub, self.temp_a_ub,
                                           self.input_w_ub[block_size],
                                           block_size // mask, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsub(mask, self.input_i_ub[block_size],
                                           self.temp_c_ub, self.temp_d_ub,
                                           block_size // mask, 1, 1, 1, 8, 8, 8)

                    self.tik_instance.data_move(self.input_r_ub_all[st], self.input_r_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_r_ub_all[st + move_offset // 2],
                                                self.input_r_ub[block_size], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub_all[st], self.input_i_ub, 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.input_i_ub_all[st + move_offset // 2],
                                                self.input_i_ub[block_size], 0, 1, burst_len, 0, 0)

            # 以上完成了十六长度以上的碟形计算
            # 以下完成16点DFT
            self.input_r_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="input_r_ub",
                scope=tik.scope_ubuf)
            self.input_i_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="input_i_ub",
                scope=tik.scope_ubuf)
            self.temp_wr_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="tempt_wr_ub",
                scope=tik.scope_ubuf)
            self.temp_wi_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="tempt_wi_ub",
                scope=tik.scope_ubuf)
            self.temp_a_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="temp_a_ub",
                scope=tik.scope_ubuf)
            self.temp_b_ub = self.tik_instance.Tensor(
                self.dtype, (self.ub_tensor_size * 2,),
                name="temp_b_ub",
                scope=tik.scope_ubuf)


            for level in range(self.data_each_block):  # 对于一个block里的数据
                for i in range(2 * self.ub_tensor_size // self.data_each_block):  # 一次进出ub所能处理的数据量的block数,w对应装载
                    self.tik_instance.data_move(self.temp_wr_ub[i * self.data_each_block],
                                                self.input_w_ub_all[level * self.data_each_block],
                                                0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.temp_wi_ub[i * self.data_each_block],
                                                self.input_w_ub_all[self.data_each_block * self.data_each_block + level * self.data_each_block],
                                                0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.input_r_ub, self.input_r_ub_all, 0, 1,
                                            2 * self.ub_tensor_size // self.data_each_block, 0, 0)
                self.tik_instance.data_move(self.input_i_ub, self.input_i_ub_all, 0, 1,
                                            2 * self.ub_tensor_size // self.data_each_block, 0, 0)
                mask = self.data_each_block * 8  # mask取满，一次128
                self.tik_instance.vmul(mask, self.temp_a_ub, self.input_r_ub, self.temp_wr_ub,
                                       self.ub_tensor_size * 2 // mask, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(mask, self.temp_b_ub, self.input_i_ub, self.temp_wi_ub,
                                       self.ub_tensor_size * 2 // mask, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(mask, self.temp_a_ub, self.temp_a_ub, self.temp_b_ub,
                                       self.ub_tensor_size * 2 // mask, 1, 1, 1, 8, 8, 8)  # 得到实部值
                for i in range(int(math.log2(self.data_each_block))):
                    self.tik_instance.vec_cpadd(mask, self.temp_a_ub, self.temp_a_ub,  # 若干次相邻相加，得到整个的和
                                                self.ub_tensor_size * 2 // mask // (2 ** i), 1, 8)

                self.tik_instance.vmul(mask, self.temp_a_ub, self.input_i_ub, self.temp_wr_ub,
                                       self.ub_tensor_size * 2 // mask // self.data_each_block, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(mask, self.temp_b_ub, self.input_r_ub, self.temp_wi_ub,
                                       self.ub_tensor_size * 2 // mask, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vsub(mask, self.temp_b_ub, self.temp_a_ub, self.temp_b_ub,
                                       self.ub_tensor_size * 2 // mask, 1, 1, 1, 8, 8, 8)
                for i in range(int(math.log2(self.data_each_block))):
                    self.tik_instance.vec_cpadd(mask, self.temp_b_ub, self.temp_b_ub,
                                                self.ub_tensor_size * 2 // mask // (2 ** i), 1, 8)

                self.tik_instance.data_move(self.input_r_ub_all, self.input_r_ub, 0, 1,
                                            2 * self.ub_tensor_size // self.data_each_block, 0, 0)
                self.tik_instance.data_move(self.input_i_ub_all, self.input_i_ub, 0, 1,
                                            2 * self.ub_tensor_size // self.data_each_block, 0, 0)
















def fft_sample(input_r, input_i, input_w, output_r, output_i, output_w, kernel_name="fft_sample"):
    fft_instance = fft(input_r, input_i, input_w, kernel_name)
    tik_instance = fft_instance.fft_compute()
    return tik_instance

if __name__ == '__main__':
    fft_size = 1024*64
    input_r_data = np.random.rand(fft_size,)
    input_i_data = np.random.rand(fft_size,)
    input_r_data = input_r_data.astype("float16")
    input_i_data = input_i_data.astype("float16")
    input_w_data = np.empty([fft_size, ], dtype="float16")

    output_w = output_r = output_i = input_i = input_r = input_w = temp_uw = {
        "dtype": "float16",
        "shape": (fft_size, )}

    for i in range(fft_size // 2):
        input_w_data[i] = math.cos(i * math.pi / fft_size)
        input_w_data[i+fft_size//2] = math.sin(i * math.pi / fft_size)

    kernel_name = "fft_sample"
    tik_instance = fft_sample(input_r, input_i, input_w, output_r, output_i, output_w, kernel_name)
    feed_dict = {'input_r_gm': input_r_data, 'input_i_gm': input_i_data, 'input_w_gm': input_w_data}

    tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
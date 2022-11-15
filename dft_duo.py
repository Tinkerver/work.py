import math
from te import tik
import numpy as np
from te import platform as tbe_platform
from tbe.common.utils import shape_util as tbe_shape_util
from tbe.dsl.instrinsic import cce_intrin as tbe_cce_intrin


class fft():
    def __init__(self, input_r, input_i, input_w, kernel_name="fft"):
        self.shape = input_r.get("shape")  # (n, 1)
        self.dtype = input_r.get("dtype")  # fp16/fp32
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.aicore_num = 2
        self.thread_num = 2  # thread_num -> enable double buffer
        # input_num : 输入数据的长度（规定为2的幂次方）
        # level : 蝶形运算的层数（log2n)
        # data_each_block: 每个block存放的元素数量
        # self.ub_tenor_size: ai core 单次计算的tensor长度。
        # self.data_num_each_core: 每个ai core 计算的总任务量
        self.input_num = tbe_shape_util.get_shape_size(self.shape)
        # Unified Buffer上数据读取和写入必须32B对齐，此参数用来计算tensor划分和数据搬运指令参数
        block_bite_size = 32
        # 获取Unified Buffer空间大小，单位为bytes，Ascend310为240KB(256KB-16KB),其中16KB为Scalar Buffer
        # ub_size_bytes = tbe_platform.get_soc_spec("UB_SIZE")
        # 根据输入的数据类型计算一个block可以存放多少个对应的元素
        dtype_bytes_size = tbe_cce_intrin.get_bit_len(self.dtype) // 8      # ==2 for fp16
        self.data_each_block = block_bite_size // dtype_bytes_size          # 16 for fp16
        self.dft_basis = self.data_each_block
        self.ub_tensor_size = 8 * 1024 // self.thread_num
        if self.dtype != "float16":
            self.ub_tensor_size //= 2
        self.vector_mask_max = 8 * self.data_each_block

        self.max_level = int(math.log2(self.input_num))
        # self.break_level = 1
        # self.end_level = 0
        self.break_level = int(math.log2(self.input_num)-math.log2(self.ub_tensor_size))
        self.end_level = int(math.log2(self.input_num)-math.log2(self.dft_basis))

        self.block_num = self.input_num // (2 * self.ub_tensor_size) // self.thread_num

        self.input_r_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_r_gm", scope=tik.scope_gm)
        self.output_r_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="output_r_gm", scope=tik.scope_gm)
        self.input_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_i_gm", scope=tik.scope_gm)
        self.output_i_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="output_i_gm", scope=tik.scope_gm)
        self.input_w_gm = self.tik_instance.Tensor(
            self.dtype, self.shape, name="input_w_gm", scope=tik.scope_gm)


    def fft_compute(self):

        # FFT_0

        for level_index in range(0, self.break_level):
            w_num = pow(2, level_index)
            r_num = self.input_num // w_num // 2

            with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_index:
                with self.tik_instance.for_range(0, self.thread_num, thread_num=self.thread_num) as thread_index:
                    self.input_r_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_r_ub",
                        scope=tik.scope_ubuf)
                    self.input_i_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_i_ub",
                        scope=tik.scope_ubuf)
                    self.input_w_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_w_ub",
                        scope=tik.scope_ubuf)
                    self.temp_a_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_a_ub",
                        scope=tik.scope_ubuf)
                    self.temp_b_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_b_ub",
                        scope=tik.scope_ubuf)
                    self.temp_c_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_c_ub",
                        scope=tik.scope_ubuf)
                    self.temp_d_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_d_ub",
                        scope=tik.scope_ubuf)

                    move_burst = self.ub_tensor_size // self.data_each_block
                    st_0 = block_index * self.thread_num * self.ub_tensor_size + thread_index * self.ub_tensor_size
                    st_1 = st_0 + self.input_num // 2

                    if level_index % 2 == 0:
                        self.tik_instance.data_move(self.input_r_ub, self.input_r_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub, self.input_i_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_r_ub[self.ub_tensor_size],
                                                    self.input_r_gm[st_1], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub[self.ub_tensor_size],
                                                    self.input_i_gm[st_1], 0, 1, move_burst, 0, 0)
                    else:
                        self.tik_instance.data_move(self.input_r_ub, self.output_r_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub, self.output_i_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_r_ub[self.ub_tensor_size],
                                                    self.output_r_gm[st_1], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub[self.ub_tensor_size],
                                                    self.output_i_gm[st_1], 0, 1, move_burst, 0, 0)

                    self.tik_instance.data_move(self.input_w_ub, self.input_w_gm[st_0], 0, 1, move_burst, 0, 0)
                    self.tik_instance.data_move(self.input_w_ub[self.ub_tensor_size],
                                                self.input_w_gm[st_1], 0, 1, move_burst, 0, 0)
                    # test_repeat = 2 * 2 * 2 * 2 * 2 * 2
                    repeat = self.ub_tensor_size // self.vector_mask_max
                    mask = self.vector_mask_max
                    # test_repeat = 255
                    self.butterfly_compute(mask, repeat, self.ub_tensor_size)

                    # ed = test_block_index * self.thread_num * test_block_size // 2 + test_thread_index * test_block_size//2
                    ed_0 = (st_0 // r_num) * (r_num // 2) + (st_0 % r_num) // (r_num // 2) * (self.input_num // 2)
                    ed_1 = (st_1 // r_num) * (r_num // 2) + (st_1 % r_num) // (r_num // 2) * (self.input_num // 2)

                    if level_index % 2 == 0:
                        self.tik_instance.data_move(self.output_r_gm[ed_0], self.input_r_ub, 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.output_i_gm[ed_0], self.input_i_ub, 0, 1, move_burst, 0, 0)

                        self.tik_instance.data_move(self.output_r_gm[ed_1], self.input_r_ub[self.ub_tensor_size],
                                                    0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.output_i_gm[ed_1], self.input_i_ub[self.ub_tensor_size],
                                                    0, 1, move_burst, 0, 0)
                    else:
                        self.tik_instance.data_move(self.input_r_gm[ed_0], self.input_r_ub, 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_gm[ed_0], self.input_i_ub, 0, 1, move_burst, 0, 0)

                        self.tik_instance.data_move(self.input_r_gm[ed_1], self.input_r_ub[self.ub_tensor_size],
                                                    0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_gm[ed_1], self.input_i_ub[self.ub_tensor_size],
                                                    0, 1, move_burst, 0, 0)



        # FFT_1
        for level_index in range(self.break_level, self.end_level):
            w_num = pow(2, level_index)
            r_num = self.input_num // w_num // 2

            with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_index:
                with self.tik_instance.for_range(0, self.thread_num, thread_num=self.thread_num) as thread_index:
                    self.input_r_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_r_ub",
                        scope=tik.scope_ubuf)
                    self.input_i_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_i_ub",
                        scope=tik.scope_ubuf)
                    self.input_w_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size*2, ),
                        name="input_w_ub",
                        scope=tik.scope_ubuf)
                    self.temp_a_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_a_ub",
                        scope=tik.scope_ubuf)
                    self.temp_b_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_b_ub",
                        scope=tik.scope_ubuf)
                    self.temp_c_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_c_ub",
                        scope=tik.scope_ubuf)
                    self.temp_d_ub = self.tik_instance.Tensor(
                        self.dtype, (self.ub_tensor_size, ),
                        name="temp_d_ub",
                        scope=tik.scope_ubuf)

                    move_burst = self.ub_tensor_size // self.data_each_block
                    st_0 = block_index * self.thread_num * self.ub_tensor_size + thread_index * self.ub_tensor_size
                    st_1 = st_0 + self.input_num // 2

                    if level_index % 2 == 0:
                        self.tik_instance.data_move(self.input_r_ub, self.input_r_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub, self.input_i_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_r_ub[self.ub_tensor_size],
                                                    self.input_r_gm[st_1], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub[self.ub_tensor_size],
                                                    self.input_i_gm[st_1], 0, 1, move_burst, 0, 0)
                    else:
                        self.tik_instance.data_move(self.input_r_ub, self.output_r_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub, self.output_i_gm[st_0], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_r_ub[self.ub_tensor_size],
                                                    self.output_r_gm[st_1], 0, 1, move_burst, 0, 0)
                        self.tik_instance.data_move(self.input_i_ub[self.ub_tensor_size],
                                                    self.output_i_gm[st_1], 0, 1, move_burst, 0, 0)

                    self.tik_instance.data_move(self.input_w_ub, self.input_w_gm[st_0], 0, 1, move_burst, 0, 0)
                    self.tik_instance.data_move(self.input_w_ub[self.ub_tensor_size],
                                                self.input_w_gm[st_1], 0, 1, move_burst, 0, 0)
                    # test_repeat = 2 * 2 * 2 * 2 * 2 * 2
                    repeat = self.ub_tensor_size // self.vector_mask_max
                    mask = self.vector_mask_max
                    # test_repeat = 255
                    self.butterfly_compute(mask, repeat, self.ub_tensor_size)

                    for w_index in range(self.ub_tensor_size//r_num):
                        split_burst = r_num // self.data_each_block
                        split_st_0 = st_0 + w_index * r_num
                        split_st_1 = st_1 + w_index * r_num

                        ed_0 = (split_st_0//r_num) * (r_num//2) + (split_st_0 % r_num) // (r_num//2) * (self.input_num//2)
                        ed_1 = (split_st_1//r_num) * (r_num//2) + (split_st_1 % r_num) // (r_num//2) * (self.input_num//2)

                        if level_index % 2 == 0:
                            self.tik_instance.data_move(self.output_r_gm[ed_0],
                                                        self.input_r_ub[w_index*r_num], 0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.output_i_gm[ed_0],
                                                        self.input_i_ub[w_index*r_num], 0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.output_r_gm[ed_1],
                                                        self.input_r_ub[w_index*r_num+self.ub_tensor_size],
                                                        0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.output_i_gm[ed_1],
                                                        self.input_i_ub[w_index*r_num+self.ub_tensor_size],
                                                        0, 1, split_burst, 0, 0)

                        else:
                            self.tik_instance.data_move(self.input_r_gm[ed_0],
                                                        self.input_r_ub[w_index*r_num], 0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.input_i_gm[ed_0],
                                                        self.input_i_ub[w_index*r_num], 0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.input_r_gm[ed_1],
                                                        self.input_r_ub[w_index*r_num+self.ub_tensor_size],
                                                        0, 1, split_burst, 0, 0)
                            self.tik_instance.data_move(self.input_i_gm[ed_1],
                                                        self.input_i_ub[w_index*r_num+self.ub_tensor_size],
                                                        0, 1, split_burst, 0, 0)

        # 若最后一层为偶数层，计算数据写回的是output_data_gm, 需要将数据通过OUT->UB->OUT搬运至input_data_gm，此时方可进行DFT运算
        # data_transfer()
        # DFT

        """
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_index:
            with self.tik_instance.for_range(0, self.thread_num, thread_num=self.thread_num) as thread_index:
                self.input_r_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size*2, ),
                    name="input_r_ub",
                    scope=tik.scope_ubuf)
                self.input_i_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size*2, ),
                    name="input_i_ub",
                    scope=tik.scope_ubuf)
                self.temp_wr_ub = self.tik_instance.Tensor(
                    self.dtype, (self.vector_mask_max*self.dft_basis, ),
                    name="tempt_wr_ub",
                    scope=tik.scope_ubuf)
                self.temp_wi_ub = self.tik_instance.Tensor(
                    self.dtype, (self.vector_mask_max*self.dft_basis, ),
                    name="tempt_wi_ub",
                    scope=tik.scope_ubuf)
                self.temp_a_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size*2, ),
                    name="temp_a_ub",
                    scope=tik.scope_ubuf)
                self.temp_b_ub = self.tik_instance.Tensor(
                    self.dtype, (self.ub_tensor_size*2, ),
                    name="temp_b_ub",
                    scope=tik.scope_ubuf)

                tensor_index = block_index * self.thread_num + thread_index
                st = tensor_index * self.ub_tensor_size * 2

                # w_st = self.level_end * self.input_num
                w_st = 0
                data_burst = self.ub_tensor_size * 2 // self.data_each_block
                w_burst = self.vector_mask_max * 16 // self.data_each_block
                mask = self.vector_mask_max
                repeat = self.ub_tensor_size * 2 // mask

                # data_move
                self.tik_instance.data_move(self.input_r_ub, self.input_r_gm[st], 0, 1,
                                            data_burst, 0, 0)
                self.tik_instance.data_move(self.input_i_ub, self.input_i_gm[st], 0, 1,
                                            data_burst, 0, 0)
                self.tik_instance.data_move(self.temp_wr_ub, self.input_w_gm[w_st], 0, 1,
                                            w_burst, 0, 0)
                self.tik_instance.data_move(self.temp_wi_ub, self.input_w_gm[w_st+self.vector_mask_max*16], 0, 1,
                                            w_burst, 0, 0)
                # dft以及数据写回
                self.dft(mask, repeat, tensor_index)
    
        """

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_r_gm, self.input_i_gm, self.input_w_gm],
                                   outputs=[self.output_r_gm, self.output_i_gm])
        return self.tik_instance

    def butterfly_compute(self, mask, repeat, ub_tensor_size):
        self.tik_instance.vec_sub(mask, self.temp_a_ub,
                                  self.input_r_ub, self.input_r_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_sub(mask, self.temp_b_ub,
                                  self.input_i_ub, self.input_i_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_add(mask, self.input_r_ub,
                                  self.input_r_ub, self.input_r_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_add(mask, self.input_i_ub,
                                  self.input_i_ub, self.input_i_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_mul(mask, self.temp_c_ub,
                                  self.temp_a_ub, self.input_w_ub,
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_mul(mask, self.temp_d_ub,
                                  self.temp_b_ub, self.input_w_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_add(mask, self.input_r_ub[ub_tensor_size],
                                  self.temp_c_ub, self.temp_d_ub,
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_mul(mask, self.temp_c_ub,
                                  self.temp_b_ub, self.input_w_ub,
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_mul(mask, self.temp_d_ub,
                                  self.temp_a_ub, self.input_w_ub[ub_tensor_size],
                                  repeat, 8, 8, 8)
        self.tik_instance.vec_sub(mask, self.input_i_ub[ub_tensor_size],
                                  self.temp_c_ub, self.temp_d_ub,
                                  repeat, 8, 8, 8)


    def dft(self, mask, repeat, tensor_index):

        for level_index in range(self.dft_basis):
            # 计算实数部分
            w_st = level_index*self.dft_basis
            self.tik_instance.vmul(mask, self.temp_a_ub, self.input_r_ub, self.temp_wr_ub[w_st],
                                   repeat, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vmul(mask, self.temp_b_ub, self.input_i_ub, self.temp_wi_ub[w_st],
                                   repeat, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vadd(mask, self.temp_a_ub, self.temp_a_ub, self.temp_b_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)
            for i in range(int(math.log2(self.dft_basis))):
                if i % 2 == 0:
                    self.tik_instance.vec_cpadd(mask, self.temp_b_ub, self.temp_a_ub,
                                                self.ub_tensor_size*2//mask//(2**i), 1, 8)
                else:
                    self.tik_instance.vec_cpadd(mask, self.temp_a_ub, self.temp_b_ub,
                                                self.ub_tensor_size*2//mask//(2**i), 1, 8)
            # 写回实数部分
            split_st = level_index * (self.ub_tensor_size * 2 // self.data_each_block)
            split_burst = self.ub_tensor_size * 2 // self.data_each_block // self.data_each_block
            ed = split_st % (self.ub_tensor_size * 2 // 16) \
                 + split_st // (self.ub_tensor_size * 2 // 16) * (self.input_num // 16)\
                 + tensor_index * (self.ub_tensor_size * 2 // 16)

            if int(math.log2(self.dft_basis)) % 2 == 0:
                self.tik_instance.data_move(self.output_r_gm[ed],
                                            self.temp_a_ub, 0, 1,
                                            split_burst, 0, 0)
            else:
                self.tik_instance.data_move(self.output_r_gm[ed],
                                            self.temp_b_ub, 0, 1,
                                            split_burst, 0, 0)
            # 计算虚数部分
            self.tik_instance.vmul(mask, self.temp_a_ub, self.input_i_ub, self.temp_wr_ub[w_st],
                                   repeat, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vmul(mask, self.temp_b_ub, self.input_r_ub, self.temp_wi_ub[w_st],
                                   repeat, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vsub(mask, self.temp_a_ub, self.temp_b_ub, self.temp_b_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)
            for i in range(int(math.log2(self.dft_basis))):
                if i % 2 == 0:
                    self.tik_instance.vec_cpadd(mask, self.temp_b_ub, self.temp_a_ub,
                                                self.ub_tensor_size*2//mask//(2**i), 1, 8)
                else:
                    self.tik_instance.vec_cpadd(mask, self.temp_a_ub, self.temp_b_ub,
                                                self.ub_tensor_size*2//mask//(2**i), 1, 8)
            # 写回虚数部分
            if int(math.log2(self.dft_basis)) % 2 == 0:
                self.tik_instance.data_move(self.output_r_gm[ed],
                                            self.temp_a_ub, 0, 1,
                                            split_burst, 0, 0)
            else:
                self.tik_instance.data_move(self.output_i_gm[ed],
                                            self.temp_b_ub, 0, 1,
                                            split_burst, 0, 0)

def fft_sample(input_r, input_i, input_w, output_r, output_i, kernel_name="fft_sample"):
    """
    calculating data

    Parameters
    ----------
    input_r : dict
        shape and dtype of input
    input_i : dict
        shape and dtype of input
    output_r : dict
        shape and dtype of output, should be same shape and type as input
    output_i : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "fft_sample"

    Returns : tik_instance
    -------
    None
    """
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
    tik_instance = fft_sample(input_r, input_i, input_w, output_r, output_i, kernel_name)
    feed_dict = {'input_r_gm': input_r_data, 'input_i_gm': input_i_data, 'input_w_gm': input_w_data}

    tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=False)
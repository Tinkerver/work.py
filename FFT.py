from cmath import sin, cos, pi
from builtins import list


class FFT_pack():
    def __init__(self, _list=None, N=0):  # _list为传入的离散序列x(0)-x(k),N是采样点，N为2^n
        if _list is None:
            _list = []
        self.list = _list
        self.N = N
        self.total_m = 0  # 序列总层数
        self._reverse_list = []  # 位倒序列表
        self.output = []  # 计算结果
        self._W = []  # 系数因子
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        print(self.output)
        for _ in range(self.N):
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  # 计算出W_N^_值

    def _reverse_pos(self, num) -> int:  # 得到数位反转后的数值，用于做蝶型计算
        out = 0
        bits = 0
        _i = self.N
        data = num
        while _i != 0:  # 得到N是2的几次方，是二进制表示数据的位数，也是序列一共有几层 bits=m+1
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):  # 对数据的每一位进行反转
            out = out << 1
            out |= (data >> i) & 1  # 0与数据的相应位进行或运算得到该数，以此转向
        self.total_m = bits - 1
        return out

    def DFT(self, _list, N) -> list:  # 计算给定序列的离散傅里叶变换结果，算法复杂度较大，返回一个列表，结果没有经过归一化处理
        self.__init__(_list, N)
        origin = self.list.copy()
        for i in range(self.N):  # 一次求解一个输出
            temp = 0
            for j in range(self.N):
                temp += origin[j] * (((cos(2 * pi / self.N) - sin(2 * pi / self.N) * 1j)) ** (i * j))
            self.output[i] = temp.__abs__()
        return self.output

    def FFT(self, _list, N, abs=True):  # 计算给定序列的傅里叶变换结果，返回列表
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)  # 计算蝶形组数
            num_each = self.N // _split  # 一组鲽形中有多少条
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]  # 左操作数 x_m(p)
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[
                        __ * 2 ** (self.total_m - m - 1)]  # 右操作数x_m(q)
                    self.output[_ * num_each + __] = temp + temp2  # 结果x_m+1(p)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)  # 结果x_m+1(q)
        if abs:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

    def stockham_FFT(self, _list, N, abs=True):  # 计算给定序列的傅里叶变幻，使用stockman方法
        self.__init__(_list, N)
        self.output = self.list
        for m in range(self.total_m):
            _split = self.N // 2 ** m  # 计算蝶形组数
            num_each = self.N // _split  # 一组鲽形中有多少条
            s_temp = [i + 1j for i in range(N)]
            for x in range(N // 2):
                k = x % num_each
                temp = self.output[x]
                temp2 = self.output[x + N // 2] * self._W[k % num_each * 2 ** (self.total_m - m - 1)]
                s_temp[(x - k) * 2 + k] = temp + temp2
                s_temp[(x - k) * 2 + k + num_each] = temp - temp2
            self.output = s_temp
        if abs:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

    def IFFT(self, _list, N) -> list:  # 计算给定序列的傅里叶逆变换结果，返回一个列表
        self.__init__(_list, N)
        for _ in range(self.N):
            self._W[_] = (cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** (-_)  # 这里的系数要重新计算，算法不同
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        for _ in range(self.N):  # 根据IFFT计算公式对所有计算列表中的元素进行*1/N的操作
            self.output[_] /= self.N
            self.output[_] = self.output[_].__abs__()
        return self.output

import time

if __name__ == '__main__':
    list = [i for i in range(32)]
    starttime = time.time()
    a = FFT_pack().stockham_FFT(list, 32, False)
    endtime = time.time()
    print(a)
    print('总共的时间为:', round(endtime - starttime, 2),'secs')

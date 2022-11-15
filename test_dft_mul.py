import numpy as np

m4_4 = np.ones((4, 4), dtype=complex)
for i in range(4):
    for j in range(4):
        m4_4[i][j] = np.exp(-2 * i * j * np.pi * 1j / 4)
print("m4_4")
print(m4_4)

m2_2 = np.ones((2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        m2_2[i][j] = np.exp(-2 * i * j * np.pi * 1j / 2)
print("m2_2")
print(m2_2)

t4_4 = np.ones((4, 4), dtype=complex)
for i in range(4):
    for j in range(4):
        t4_4[i][j] = np.exp(-2 * i * j * np.pi * 1j / 16)
print("t4_4")
print(t4_4)

t2_4 = np.ones((2, 4), dtype=complex)
for i in range(2):
    for j in range(4):
        t2_4[i][j] = np.exp(-2 * i * j * np.pi * 1j / 8)
print("t2_4")
print(t2_4)

t2_8 = np.ones((2, 8), dtype=complex)
for i in range(2):
    for j in range(8):
        t2_8[i][j] = np.exp(-2 * i * j * np.pi * 1j / 16)
print("t2_8")
print(t2_8)

n = np.ones((16), dtype=complex)
mat = [0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15]

for i in range(16):
    n[i] = i

n = n.reshape(4, 4)
print(n)
n=n.T
n = np.matmul(n,m4_4)
print("四个fft序列")
print(n)

n = n * t4_4
print(n)
r = np.matmul(m4_4,n)
print(r)


#
# n1 = n[[0, 1], :]
# n2 = n[[2, 3], :]
#
# n1 = n1 * t2_4
# n2 = n2 * t2_4
#
# n1 = np.matmul(m2_2, n1)
# n2 = np.matmul(m2_2, n2)
#
# n1 = n1.reshape(1, 8)
# n2 = n2.reshape(1, 8)
#
# n = np.append(n1, n2, 0)
# n = n * t2_8
#
# n = np.matmul(m2_2, n)
# print(n)


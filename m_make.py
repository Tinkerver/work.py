import numpy as np

# m = np.array([[0 for i in range(64)] for j in range(64)])
# reverse_list = [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31]
# print(m.shape)
#
# for i in range(64):
#     m[i][reverse_list[i]] = 1


# 生成用于32大小矩阵的位反转
# m = np.array([[0 for i in range(32)] for j in range(32)])
# reverse_list = [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27,
#                 7, 23, 15, 31]
# for i in range(32):
#     m[i][reverse_list[i]] = 1
# np.set_printoptions(threshold=np.inf)
# a = np.array([c[:16] for c in m])
# b = np.array([c[16:] for c in m])
# m = np.append(a, b, axis=0)
# m = m.reshape((32, 32))
# print(m)


m = np.array([[0 for i in range(32)] for j in range(32)])
reverse_list = [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27,
                7, 23, 15, 31]
for i in range(32):
    m[i][reverse_list[i]] = 1
np.set_printoptions(threshold=np.inf)
tmp=np.split(m,16,axis=-1)
print(m)
r=np.array(tmp[0])
for i in tmp[1:]:
    r=np.append(r,i,axis=0)
print(r)
# a = np.array([c[:16] for c in m])
# b = np.array([c[16:] for c in m])
# m = np.append(a, b, axis=0)
# m = m.reshape((32, 32))
#print(m)
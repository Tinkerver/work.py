import numpy as np

m = np.array([[0 for i in range(8)] for j in range(8)])
reverse_list = [0, 4, 2, 6, 1, 5, 3, 7]
for i in range(8):
    m[i][reverse_list[i]] = 1

x = np.array(range(64)).reshape((8, 8))
print(x)
x = np.dot(x, m)
print(x)

x = np.dot(m, x)

print(x)

x = x.transpose()
print(x.reshape(1,64))

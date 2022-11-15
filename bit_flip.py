import numpy as np

a = np.mat([i for i in range(32)])
a = a.reshape((32))
a1 = a[:16]
a2 = a[16:]

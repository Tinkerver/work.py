import numpy as np
import torch
import torch.nn as nn

input = np.random.rand(10, 10, 256)
weight = np.random.rand(5, 5)
output = np.zeros((256, 6, 6))
input = input.transpose((2, 0, 1))
print(input.shape)
for n in range(256):
    for h in range(6):
        for w in range(6):
            temp = 0
            for i in range(5):
                for j in range(5):
                    temp += input[n][h + i][w + j] * weight[i][j]
            output[n][h][w] = temp
output = output.transpose((1, 2, 0))
print(output.shape)

# kernel_tensor = torch.tensor(weight)
# im_tensor = torch.tensor(input)
#
# conv = nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=(5, 5), bias=False)
# conv.weight = nn.Parameter(kernel_tensor.unsqueeze(0).unsqueeze(0))
# output2 = conv(im_tensor.unsqueeze(1))
# print(output2[0][0][0])

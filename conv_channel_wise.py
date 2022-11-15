import numpy as np
import torch
import torch.nn as nn

n, c_in, h, w = 1, 3, 16, 16
c_out = 1

im = np.random.rand(n, c_in, h, w)
kernel = np.random.rand(c_in, 3, 3)
im_col = np.zeros((c_in, n * h * w, kernel.shape[-1] * kernel.shape[-2]))

im_pad = np.pad(im, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant')
padded_input_shape = im_pad.shape

k = 0
for idx_im in range(n):
    for idx_c in range(c_in):
        for i in range(1, padded_input_shape[-2] - 1):
            for j in range(1, padded_input_shape[-1] - 1):
                im_col[idx_c, k, :] = im_pad[idx_im, idx_c, i - 1:i + 2, j - 1:j + 2].reshape(-1)
                k += 1
        k = 0

print(kernel[0].shape)
output_mat = np.array([])
for idx_c in range(c_in):
    output_mat_slice = np.matmul(im_col[idx_c], kernel.reshape((c_in, 9))[idx_c])
    output_mat = np.append(output_mat, output_mat_slice)

output_mat = output_mat.reshape((c_in, n, h, w))

output_mat = output_mat.transpose((1, 0, 2, 3))
print(output_mat)
print(output_mat.shape)
# kernel_tensor = torch.tensor(kernel)
# im_tensor = torch.tensor(im)
# conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=1,
#                  padding_mode='zeros', bias=False)
# conv.weight = nn.Parameter(kernel_tensor)
# output = conv(im_tensor)
# print(output)

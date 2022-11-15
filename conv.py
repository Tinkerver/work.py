import numpy as np
import torch
import torch.nn as nn

n, c_in, h, w = 1, 3, 16, 16
c_out = 2

im = np.random.rand(n, c_in, h, w)
kernel = np.random.rand(c_out, c_in, 3, 3)
im_col = np.zeros((n * h * w, c_in * kernel.shape[-1] * kernel.shape[-2]))

kernel_tensor = torch.tensor(kernel)
im_tensor = torch.tensor(im)

im_pad = np.pad(im, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant')
padded_input_shape = im_pad.shape

k = 0
for idx_im in range(n):
    for i in range(1, padded_input_shape[-2] - 1):
        for j in range(1, padded_input_shape[-1] - 1):
            im_col[k, :] = im_pad[idx_im, :, i - 1:i + 2, j - 1:j + 2].reshape(-1)
            k += 1
print(im_col.shape)

output_mat = np.matmul(im_col, kernel.reshape(c_out, 3 * 3 * c_in).transpose((1, 0)))
print(output_mat.shape)

output_mat = output_mat.transpose((1, 0))
output_mat = output_mat.reshape((c_out, n, h, w))
output_mat = output_mat.transpose((1, 0, 2, 3))
print(output_mat)

conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=1,
                 padding_mode='zeros', bias=False)
conv.weight = nn.Parameter(kernel_tensor)
output = conv(im_tensor)
print(output)


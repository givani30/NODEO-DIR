import math

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from torch import nn

from NeuralODE import ODEF


# NOT USED
class GaussianKernel(torch.nn.Module):
    def __init__(self, win=11, nsig=0.1):
        super(GaussianKernel, self).__init__()
        self.win = win
        self.nsig = nsig
        kernel_x, kernel_y, kernel_z = self.gkern1D_xyz(self.win, self.nsig)
        kernel = kernel_x * kernel_y * kernel_z
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.register_buffer("kernel", kernel)

    def gkern1D(self, kernlen=None, nsig=None):
        '''
        :param nsig: large nsig gives more freedom(pixels as agents), small nsig is more fluid.
        :return: Returns a 1D Gaussian kernel.
        '''
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern1d = kern1d / kern1d.sum()
        return torch.tensor(kern1d, requires_grad=False).float()

    def gkern1D_xy(self, kernlen=None, nsig=None):
        """Returns 2 1D Gaussian kernel on xy direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
        return kernel_x, kernel_y

    def forward(self, x):
        pad = int((self.win - 1) / 2)
        # Apply Gaussian by 3D kernel
        x = F.conv3d(x, self.kernel, padding=pad)
        return x


class AveragingKernel(torch.nn.Module):
    def __init__(self, win=11):
        super(AveragingKernel, self).__init__()
        self.win = win

    def window_averaging(self, v):
        win_size = self.win
        v = v.double()

        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 2

        v_padded = F.pad(v, pad=pad, mode='constant', value=0)  # [x+pad, y+pad]

        # Run the cumulative sum across both dimensions
        v_cs_x = torch.cumsum(v_padded, dim=2)
        v_cs_xy = torch.cumsum(v_cs_x, dim=3)

        x, y = v.shape[2:]

        # Use subtraction to calculate the window sum
        v_win = v_cs_xy[:, :, win_size:, win_size:] \
                - v_cs_xy[:, :, win_size:, :y] \
                - v_cs_xy[:, :, :x, win_size:] \
                + v_cs_xy[:, :, :x, :y]

        # Normalize by number of elements
        v_win = v_win / (win_size ** 2)
        v_win = v_win.float()
        return v_win

    def forward(self, v):
        return self.window_averaging(v)


#TODO: Convert to 2D nn
class BrainNet(ODEF):
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet, self).__init__()
        padding_mode = 'replicate'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_pass = smoothing_pass
        # self.enc_conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.bottleneck_sz = int(
            math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds))) #TODO: convert to 2d
        self.lin1 = nn.Linear(288, self.bs, bias=bias)  #TODO: verify dimension, conolution theory
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * 2, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels
        if self.smoothing_kernel == 'AK':
            self.sk = AveragingKernel(win=smoothing_win)
        else:
            self.sk = GaussianKernel(win=smoothing_win, nsig=0.1) #TODO: implement for 2d

    def forward(self, x):
        imgx = self.img_sz[0]
        imgy = self.img_sz[1]
        # x = self.relu(self.enc_conv1(x))
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')  # Optional to downsample the image
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.enc_conv6(x)
        x = x.view(-1)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.view(1, 2, int(math.ceil(imgx / pow(2, self.ds))), int(math.ceil(imgy / pow(2, self.ds)))) #TODO: convert to 2d
        for _ in range(self.ds):
            x = F.upsample(x, scale_factor=2, mode='bilinear')
        # Apply Gaussian/Averaging smoothing
        for _ in range(self.smoothing_pass):
            if self.smoothing_kernel == 'AK':
                x = self.sk(x)
            else:
                x_x = self.sk(x[:, 0, :, :].unsqueeze(1))
                x_y = self.sk(x[:, 1, :, :].unsqueeze(1))
                x = torch.cat([x_x, x_y], 1)
        return x

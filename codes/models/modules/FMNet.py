import functools
from warnings import filters
import torch.nn as nn
import torch
import models.modules.arch_util as arch_util
import torch.nn.functional as F
from models.modules.arch_util import initialize_weights
from utils.gpu_memory_log import gpu_memory_log
import math

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class FMBlock(nn.Module):
    def __init__(self, nf=64, opt=None):
        super(FMBlock, self).__init__()
        self.opt = opt

        self.down = nn.Conv2d(nf, opt['FM_channelNumber'], 1, 1, 0, bias=True)
        self.conv1_f = nn.Conv2d(opt['FM_channelNumber'], 2 * opt['FM_kernelNumber'], 3, 1, 1, bias=True)
        self.conv1_w = nn.Conv2d(opt['FM_channelNumber'], opt['FM_kernelNumber'], 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(opt['FM_channelNumber'], opt['FM_channelNumber'], 3, 1, 1, bias=True)
        self.up = nn.Conv2d(opt['FM_channelNumber'], nf, 1, 1, 0, bias=True)
        
        self.pi = 3.1415927410125732
        p = torch.arange(0, opt['FM_kernelSize'], 1)
        q = torch.arange(0, opt['FM_kernelSize'], 1)
        p, q = torch.meshgrid(p, q)
        self.p = ((p + 0.5)* self.pi / opt['FM_kernelSize']).cuda().view(1, 1, 1, opt['FM_kernelSize'], opt['FM_kernelSize'])
        self.q = ((q + 0.5)* self.pi / opt['FM_kernelSize']).cuda().view(1, 1, 1, opt['FM_kernelSize'], opt['FM_kernelSize'])

        # initialization
        initialize_weights([self.down, self.conv1_f, self.conv1_w, self.conv2, self.up], 0.1)

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.opt['FM_kernelNumber']

        identity = x
        
        x = self.down(x)
        frequency = self.conv1_f(x) # N C H W -> N 2*K H W 
        weight = self.conv1_w(x) # N C H W -> N K H W 
        frequency = torch.sigmoid(frequency)
        weight = torch.nn.functional.softmax(weight, dim=1)
        frequency = frequency * (self.opt['FM_kernelSize'] - 1)
        frequency = frequency.permute(0, 2, 3, 1).contiguous().view(N, H * W, 2, K) # N 2*K H W -> N H*W 2 K
        weight = weight.permute(0, 2, 3, 1).contiguous().view(N, H * W, 1, K) # N K H W -> N H*W 1 K
        hFrequency = frequency[:, :, 0, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, self.opt['FM_kernelSize'], self.opt['FM_kernelSize']])
        wFrequency = frequency[:, :, 1, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, self.opt['FM_kernelSize'], self.opt['FM_kernelSize']])
        p = self.p.expand([N, H * W, K, -1, -1])
        q = self.q.expand([N, H * W, K, -1, -1])
        
        kernel = torch.cos(hFrequency * p) * torch.cos(wFrequency * q)
        kernel = kernel.view(N, H * W, K, self.opt['FM_kernelSize'] ** 2)
        kernel = torch.matmul(weight, kernel)
        kernel = kernel.view(N, H * W, self.opt['FM_kernelSize'] ** 2, 1)
        # N H*W K**2 1

        v = torch.nn.functional.unfold(x, kernel_size=self.opt['FM_kernelSize'], padding=int((self.opt['FM_kernelSize'] - 1) / 2), stride=1) # N C H W -> N C*(K**2) H*W
        v = v.view(N, self.opt['FM_channelNumber'], self.opt['FM_kernelSize'] ** 2, H * W) # N C K**2 H*W
        v = v.permute(0, 3, 1, 2).contiguous() # N H*W C K**2

        z = torch.matmul(v, kernel) # N H*W C 1
        z = z.squeeze(-1).view(N, H, W, self.opt['FM_channelNumber']).permute(0, 3, 1, 2) # N H*W C -> N C H W
        z = self.conv2(z)
        out = self.up(z)
        
        return identity + out

    def build_filter(self, kernelSize):
        filters = torch.zeros((kernelSize, kernelSize, kernelSize, kernelSize))
        for i in range(kernelSize):
            for j in range(kernelSize):
                for h in range(kernelSize):
                    for w in range(kernelSize):
                        filters[i, j, h, w] = math.cos(math.pi * i * (h + 0.5) / kernelSize) * math.cos(math.pi * j * (w + 0.5) / kernelSize)
        return filters.view(kernelSize ** 2, kernelSize, kernelSize).cuda()

class FMNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu', opt=None):
        super(FMNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        fm_block = functools.partial(FMBlock, nf=nf, opt=opt)
        if opt['FM_blockNumber'] == 0:
            self.recon_trunk_fm = nn.Identity()
        else:
            self.recon_trunk_fm = arch_util.make_layer(fm_block, opt['FM_blockNumber'])

        res_block = functools.partial(ResidualBlock_noBN, nf=nf)
        if nb - opt['FM_blockNumber'] == 0:
            self.recon_trunk_res = nn.Identity()
        else:
            self.recon_trunk_res = arch_util.make_layer(res_block, nb - opt['FM_blockNumber'])

        self.upconv = nn.Conv2d(nf, nf*4, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # initialization
        initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last], 0.1)

    def forward(self, x):
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk_res(fea)
        out = self.recon_trunk_fm(out)
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return out
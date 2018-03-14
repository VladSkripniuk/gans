import math

from torch.autograd import Variable

from torch import nn
from torch.nn import functional as F

from .max_sv import *

class SNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, spec_norm=True, detach=False):
        super(SNConv2d, self).__init__()

        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.spec_norm = spec_norm
        self.detach = detach

        self.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel_size[0], kernel_size[1]).normal_(0.0, 0.02))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels,).zero_())

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.u = None
        self.sigma_approx = 1
        self.sigma_true = 1



    def forward(self, input):
        cuda = input.is_cuda

        sigma, u, _ = max_singular_value(self.weight.view(self.weight.size()[0], -1), u=self.u, cuda=cuda)

        self.u = Variable(u.data)

        if self.spec_norm:
          if self.detach:
              W_SN = self.weight / sigma.detach()
          else:
              W_SN = self.weight / sigma
        else:
          W_SN = self.weight

        self.sigma_approx = sigma

        # _, S, _ = torch.svd(self.W.view(self.W.size()[0], -1))
        # self.sigma_true = S[0]


        out = F.conv2d(input, W_SN, self.bias, self.stride, self.padding)

        return out
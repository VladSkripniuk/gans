import math

from torch.autograd import Variable

from torch import nn
from torch.nn import functional as F

from .max_sv import *

class SNLinear(nn.Linear):
    def __init__(self, *args, spec_norm=True, detach=False, **kwargs):
        super(SNLinear, self).__init__(*args, **kwargs)

        self.spec_norm = spec_norm
        self.detach = detach


        self.u = None
        self.sigma_approx = 1
        self.sigma_true = 1



    def forward(self, input):
        cuda = input.is_cuda

        sigma, u, _ = max_singular_value(self.weight, u=self.u, cuda=cuda)

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


        out = F.linear(input, W_SN, self.bias)

        return out
import torch

from torch.autograd import Variable, Function

from torch import nn
from torch.nn import functional as F

import numpy as np
from numpy.random import choice

class channel_drop(Function):
    @staticmethod
    def _make_mask(batch_size, in_channels, protected_channels, out_nonzero_channels, groupby):
        assert in_channels % groupby == 0

        mask = torch.FloatTensor(batch_size, in_channels // groupby).zero_()
        
        if protected_channels is not None:
            protected_channels_tensor = torch.LongTensor(protected_channels).expand(batch_size,1)
            mask.scatter_(1, protected_channels_tensor, 1)


        if protected_channels is None:
            protected_channels = []
        else:
            protected_channels = protected_channels

        not_protected_channels = [i for i in range(in_channels // groupby) if i not in protected_channels]
        num_channels_to_add = out_nonzero_channels - len(protected_channels)
        
        if num_channels_to_add:
            chosen_channels = torch.from_numpy(np.stack([choice(not_protected_channels, size=num_channels_to_add, replace=False) for _ in range(batch_size)], axis=0))

            mask.scatter_(1, chosen_channels, 1)
            
        mask = mask.view(batch_size, in_channels // groupby, 1)
        mask = mask.expand(batch_size, in_channels // groupby, groupby)

        mask = mask.contiguous()
        mask = mask.view(batch_size, in_channels)

        return mask.view(batch_size, in_channels, 1, 1)

    @staticmethod
    def forward(ctx, input, in_channels, protected_channels=None, out_nonzero_channels=1, groupby=1, train=True):

        ctx.train = train

        if not ctx.train:
            return input

        output = input.clone()

        ctx.mask = channel_drop._make_mask(input.size(0), in_channels, protected_channels, out_nonzero_channels, groupby)
        
        if input.is_cuda:
            ctx.mask = ctx.mask.cuda()

        ctx.mask = ctx.mask.expand_as(input)
        
        output.mul_(ctx.mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * Variable(ctx.mask), None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None


class ChannelDrop(nn.Module):
    
    def __init__(self, in_channels, protected_channels=None, out_nonzero_channels=1, groupby=1):
        super(ChannelDrop, self).__init__()
        
        self.in_channels = in_channels
        self.protected_channels = protected_channels
        self.out_nonzero_channels = out_nonzero_channels
        self.groupby = groupby

    def forward(self, input):

        return channel_drop.apply(input, self.in_channels, self.protected_channels, self.out_nonzero_channels, self.groupby, self.training)

if __name__ == '__main__':

    layer = ChannelDrop(3, protected_channels=[0], out_nonzero_channels=2)

    x = Variable(torch.ones(3, 3, 2, 2).float(), requires_grad=True)
    print(x.norm())
    # y = channel_drop.apply(x, 3, [0], 2)
    y = layer(x)
    print(y.norm())

    y.sum().backward()

    print(x.grad.norm())




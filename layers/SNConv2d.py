from torch.autograd import Variable

from torch import nn
from torch.nn import functional as F

from .max_sv import *

class SNConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(SNConv2d, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		if type(kernel_size) is int:
			kernel_size = (kernel_size, kernel_size)

		self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel_size[0], kernel_size[1]).normal_())
		self.b = nn.Parameter(torch.FloatTensor(out_channels,).normal_())

		self.u = None


	def forward(self, input):
		cuda = input.is_cuda

		sigma, u, _ = max_singular_value(self.W.view(self.W.size()[0], -1), u=self.u, cuda=cuda)

		self.u = Variable(u.data)

		W_SN = self.W / sigma#.detach()

		out = F.conv2d(input, W_SN, self.b, self.stride, self.padding)

		return out
import torch

from torch.autograd import Variable

from torch import nn
from torch.nn import functional as F


########## Layers for separable DCGAN generator ###########################################
class ConvTranspose2d_separable(nn.Module):
    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride=1, padding=0, bias=False, red_portion=0.5):
        super(ConvTranspose2d_separable, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion)

        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red

        self.convt_half = nn.ConvTranspose2d(self.n_input_ch_red, self.n_output_ch_red,
                                             kernel_size, stride, padding, bias=bias)
        self.convt_all = nn.ConvTranspose2d(self.n_input_ch, self.n_output_ch_green,
                                            kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        first_half = input[:, :self.n_input_ch_red, :, :]
        first_half_conv = self.convt_half(first_half)
        full_conv = self.convt_all(input)
        all_conv = torch.cat((first_half_conv, full_conv), 1)
        return all_conv


class Conv2d_separable(nn.Module):
    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride, padding, bias=False, red_portion=0.5):
        super(Conv2d_separable, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion)

        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red

        self.conv_half = nn.Conv2d(self.n_input_ch_red, self.n_output_ch_red,
                                   kernel_size, stride, padding, bias=bias)
        self.conv_all = nn.Conv2d(self.n_input_ch, self.n_output_ch_green,
                                  kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        first_half = input[:, :self.n_input_ch_red, :, :]
        first_half_conv = self.conv_half(first_half)
        full_conv = self.conv_all(input)
        all_conv = torch.cat((first_half_conv, full_conv), 1)
        return all_conv
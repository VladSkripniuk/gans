import os

import numpy as np

import torch

from torch import optim
from torch import nn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from layers.ChannelDrop import ChannelDrop

import datasets

import gan

from logger import Logger

class mnistnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(mnistnet_G,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*4,kernel_size=4,bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 4 x 4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 8 x 8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 16 x 16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1,bias=bias),
                                 # nn.Sigmoid())
                                 nn.Tanh())
        self.layer_drop = ChannelDrop(nc, groupby=3)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer_drop(out)
        return out


opt = gan.Options()

opt.cuda = True

opt.path = 'multiGAN_CIFAR/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000
opt.conditional = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


netG = mnistnet_G(nc=30,nz=100)
netG.load_state_dict(torch.load('multiGAN_CIFAR/gen_100000.pth'))


def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(64, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')

    fake = gan.gen_fake_data(64, opt.nz, noise=save_samples.noise)
    # fake = next(data_iter)
    # print(fake.min(), fake.max())

    fake = fake.view(-1, 3, 32, 32)

    fake_01 = (fake.data.cpu() + 1.0) * 0.5
    # print(fake_01.min(), fake_01.max())

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.jpeg'.format(i_iter), nrow=10)
    # alkjfd
    gan.netG.train()



gan1 = gan.GAN(netG=netG, netD=None, optimizerD=None, optimizerG=None, opt=opt)

save_samples(gan1, 'final')
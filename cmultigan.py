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

from layers.SNConv2d import SNConv2d
from layers.SNLinear import SNLinear

import datasets

import gan

from logger import Logger

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class mnistnet_D(nn.Module):
    def __init__(self, nc=1, ndf=64, BN=True, bias=True):
        super(mnistnet_D,self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf),
                               nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf*2),
                               nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf*4),
                               nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
                               # nn.Sigmoid())

    def forward(self,input):
        x, y = input
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out.view(-1)


class mnistnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False, n_classes=10): # 256 ok
        super(mnistnet_G,self).__init__()
        self.n_classes = n_classes
        self.nc = nc

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

        self.apply(weights_init)

    def forward(self,x):
        _, y = torch.max(x[:,self.nc:,0,0], dim=1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        multiout = Variable(torch.FloatTensor((out.size(0), self.n_classes, out.size(2), out.size(3))))
        if x.is_cuda:
            multiout = multiout.cuda()
        print(out.size())
        print(out)

        multiout.scatter_(1, y.view(-1,1,1,1).expand(len(out), 1, 32, 32), out)

        return multiout


opt = gan.Options()

opt.cuda = True

opt.path = 'cmultiGAN/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000

opt.conditionalD = False
opt.conditional = True

opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='multiGAN')

data = datasets.labeledMNISTDataset()


mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = mnistnet_G(nc=1,nz=110)
netD = mnistnet_D(nc=10,BN=True)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))




def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(100, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')



    y = np.repeat(np.arange(10), 10)
    y = torch.autograd.Variable(torch.from_numpy(y))
    if gan.opt.cuda:
        y = y.cuda()

    noise = save_samples.noise
    noise = gan.join_xy((noise, y))

    fake = netG(noise)
    
    fake = fake.view(-1, 1, 32, 32)

    fake_01 = (fake.data.cpu() + 1.0) * 0.5
    # print(fake_01.min(), fake_01.max())

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.jpeg'.format(i_iter), nrow=10)
    # alkjfd
    gan.netG.train()


def callback(gan, i_iter):

    # if i_iter % 5000 == 0:
    #     save_inception_score(gan, i_iter)

    if i_iter % 50 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()
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


class LINnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(LINnet_G,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*8,kernel_size=(3, 5), bias=bias),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        # 3 x 5
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 6 x 10
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 12 x 20
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 24 x 40
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1, bias=bias),
                                 # nn.Sigmoid())
                                 nn.Tanh())
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class LINnet_D(nn.Module):
    def __init__(self,nc=1,ndf=64,BN=True,bias=False,n_classes=6): # 128 ok
        super(LINnet_D,self).__init__()

        self.n_classes=n_classes

        # 48 x 80
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 24 x 40
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 12 x 20
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 6 x 10
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 3 x 5
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias))#,
                                 # nn.Sigmoid())

        self.apply(weights_init)

    def forward(self,input):
        x = input
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)

        return h.view(-1)


opt = gan.Options()

opt.cuda = True

opt.path = 'condLIN41/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000

opt.conditionalD = True
opt.conditional = True

opt.wgangp_lambda = 10.0
opt.n_classes = 41
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='multiGAN')

# data = datasets.LINDataset(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
data = datasets.LINDataset(proteins='all', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = LINnet_G(nc=2,nz=141)
netD = LINnet_D(nc=43,BN=True)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(410, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')


    y = np.repeat(np.arange(41), 10)
    y = torch.autograd.Variable(torch.from_numpy(y))
    if gan.opt.cuda:
        y = y.cuda()

    noise = save_samples.noise
    noise = gan.join_xy((noise, y))

    fake = netG(noise)
    
    fake = fake.view(-1, 2, 48, 80)

    fake_01 = torch.FloatTensor(len(fake), 3, 48, 80).fill_(-1)
    fake_01[:,:2,:,:] = (fake.data.cpu() + 1.0) * 0.5
    # print(fake_01.min(), fake_01.max())

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.png'.format(i_iter), nrow=10)
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
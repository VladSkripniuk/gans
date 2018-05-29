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

# from layers.SNConv2d import SNConv2d
# from layers.SNLinear import SNLinear

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


class LINnet_G0(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(LINnet_G0,self).__init__()
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

class LINnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False, n_classes=6): # 256 ok
        self.n_classes = n_classes
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

        # self.linear = nn.Parameter(torch.FloatTensor(41, 10, 100).normal_(0.1, 0.02))

        self.apply(weights_init)

    def forward(self,x):

        # _, y = torch.max(x[:,-41:,:,:].sum(dim=2).sum(dim=2), dim=1)
        # x = x[:,:-41,:,:].contiguous()

        # matrix = Variable(torch.index_select(self.linear.data, 0, y.data)).contiguous()

        # x = (matrix.view(-1, 10, 100) * x.view(-1, 10, 1)).sum(dim=1)
        # x = x.view(-1, 100, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


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
        self.layer5 = ListModule(*[nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias)).cuda() for _ in range(41)])#,
                                 # nn.Sigmoid())
        # self.linear = nn.Linear(n_classes, 3*5*ndf*8)#,


        # self.embedding = nn.Linear(n_classes, ndf*8, bias=False)
        # self.embedding.weight.data.normal_(0,1)

        # self.linear = nn.Linear(ndf*8, 1, bias=False)
        # self.linear = nn.Linear(ndf*8+n_classes, 1, bias=False)

        self.apply(weights_init)

    def forward(self,x,y):
        # x, y = input
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        output = Variable(torch.FloatTensor(h.size(0))).cuda()

        for i in range(h.size(0)):
            output[i] = self.layer5[y.data[i]](h[i:i+1,:,:,:]).view(-1)

        # h = torch.sum(h, dim=2).sum(dim=2)  # Global pooling
        # # output = self.linear(h)

        # th = torch.cuda if h.is_cuda else torch

        # y_onehot = th.FloatTensor(y.size()[0], self.n_classes)
        # y_onehot.zero_()
        # y_onehot.scatter_(1, y.data.view(-1,1), 1)

        # y_onehot = Variable(y_onehot)

        # # # h = torch.cat([h, y_onehot], dim=1)

        # w_y = self.linear(y_onehot.float())        
        # # w_y = self.embedding(y_onehot.float())

        # # # output += torch.sum(w_y * h, dim=1).view(-1, 1)
        # output = torch.sum(w_y * h.view(h.size(0), -1), dim=1).view(-1, 1)
        # # print(h.size())
        # # output = self.linear(h)

        return output.view(-1)


opt = gan.Options()

opt.cuda = True

opt.path = 'multiconvLIN41_conv/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000

opt.conditionalD = False
opt.conditional = True

opt.wgangp_lambda = 10.0
opt.n_classes = 41
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='multiGAN')

# data = datasets.LINDataset(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
data = datasets.LINDataset(proteins='all', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
#['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1', 'Act1', 'Gef1', 'For3', 'Ras1', 'Scd2', 'Tip1']
mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = LINnet_G0(nc=2,nz=141)
netD = LINnet_D(nc=2,BN=True,n_classes=41)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(opt.n_classes*10, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')


    y = np.repeat(np.arange(opt.n_classes), 10)
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
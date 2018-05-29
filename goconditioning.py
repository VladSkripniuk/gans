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

from layers.separable import ConvTranspose2d_separable


import datasets

import gan

from logger import Logger

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('separable') != -1:
        m.convt_half.weight.data.normal_(0.0, 0.02)
        m.convt_all.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def join_xy(x, y, n_classes):
    th = torch.cuda if x.is_cuda else torch

    y_onehot = th.FloatTensor(x.size()[0], n_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, y.data.view(-1,1), 1)
    y_onehot = y_onehot.view(x.size()[0], n_classes, 1, 1)

    return torch.cat((x, torch.autograd.Variable(y_onehot.expand(x.size()[0], n_classes, x.size()[2], x.size()[3]))), 1)

def label2go2embedding(y, embedding_layer):

    n_classes = embedding_layer.weight.data.size()

    th = torch.cuda if y.is_cuda else torch
    y_onehot1 = th.FloatTensor(y.size()[0], emb)
    y_onehot1.zero_()
    y_onehot1.scatter_(1, y1.data.view(-1,1), 1)
    y_onehot1 = Variable(y_onehot1)


    pass

class LINnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False, n_gens=41,n_deletions=34): # 256 ok
        super(LINnet_G,self).__init__()
        self.n_gens=n_gens
        self.n_deletions=n_deletions
        
        self.nz = nz

        # 
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*8,kernel_size=(3, 8), bias=bias),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        
        self.layer11 = nn.Sequential(nn.ConvTranspose2d(nz//2+n_deletions,ngf*4,kernel_size=(3, 8), bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        self.layer12 = nn.Sequential(nn.ConvTranspose2d(nz+n_deletions+n_gens,ngf*4,kernel_size=(3, 8), bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # # 3 x 5
        self.layer2 = nn.Sequential(ConvTranspose2d_separable(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 6 x 10
        self.layer3 = nn.Sequential(ConvTranspose2d_separable(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 12 x 20
        self.layer4 = nn.Sequential(ConvTranspose2d_separable(ngf*2,ngf,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 24 x 40
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1, bias=bias),
                                 # nn.Sigmoid())
                                 nn.Tanh())
        self.apply(weights_init)

    def forward(self, x, y1, y2):

        h1 = join_xy(x[:,:self.nz//2,:,:], y2, self.n_deletions)
        h2 = join_xy(torch.cat([h1, x[:,self.nz//2:,:,:]], dim=1), y1, self.n_gens)
        # h2 = torch.cat([h1, x[:,self.nz//2:,:,:]], dim=1)

        # h1 = x[:,:self.nz//2,:,:]
        # h2 = join_xy(x, y1, self.n_classes)

        # h1 = join_xy(x[:,:self.nz//2,:,:], y2, self.n_deletions)
        # h2 = torch.cat([h1, x[:,self.nz//2:,:,:]], dim=1)
        out = torch.cat([self.layer11(h1), self.layer12(h2)], dim=1)
        
        # out = self.layer1(x)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # print(out.size())
        return out


class LINnet_D(nn.Module):
    def __init__(self,nc=1,ndf=64,BN=True,bias=False,n_gens=41,n_deletions=34): # 128 ok
        super(LINnet_D,self).__init__()

        self.n_gens=n_gens
        self.n_deletions=n_deletions

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
        # self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias))#,
                                 # nn.Sigmoid())

        self.embedding1 = nn.Linear(n_gens, ndf*4, bias=False)
        self.embedding2 = nn.Linear(n_deletions, ndf*4, bias=False)
        self.embedding1.weight.data.normal_(0,1)
        self.embedding2.weight.data.normal_(0,1)
        # self.linear = nn.Linear(ndf*8, 1, bias=True)

        # self.linear = nn.Linear(ndf*8, 1, bias=False)
        # self.linear = nn.Linear(ndf*8+n_classes, 1, bias=False)

        self.apply(weights_init)

    def forward(self, x, y1, y2):
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        # h = self.layer5(h)

        h = torch.sum(h, dim=2).sum(dim=2)  # Global pooling
        # output = self.linear(h)
        # print(y1)
        th = torch.cuda if h.is_cuda else torch
        y_onehot1 = th.FloatTensor(y1.size()[0], self.n_gens)
        y_onehot1.zero_()
        y_onehot1.scatter_(1, y1.data.view(-1,1), 1)
        y_onehot1 = Variable(y_onehot1)

        y_onehot2 = th.FloatTensor(y2.size()[0], self.n_deletions)
        y_onehot2.zero_()
        y_onehot2.scatter_(1, y2.data.view(-1,1), 1)
        y_onehot2 = Variable(y_onehot2)


        # # h = torch.cat([h, y_onehot], dim=1)
        # w_y = self.embedding2(y_onehot2.float())

        w_y1 = self.embedding1(y_onehot1.float())
        w_y2 = self.embedding2(y_onehot2.float())

        # w_y = torch.cat([w_y1, w_y2], dim=1)

        # output = self.linear(h)
        # output = torch.sum(w_y * h, dim=1).view(-1, 1)
        output1 = torch.sum(w_y1 * h[:,:256], dim=1).view(-1, 1)
        output2 = torch.sum(w_y2 * h[:,256:], dim=1).view(-1, 1)
        # output = torch.sum(w_y2 * h, dim=1).view(-1, 1)
        # # print(h.size())
        # # output = self.linear(h)

        # output = self.layer5(h)
        # print(output.size())

        # return output.view(-1)
        return output1.view(-1), output2.view(-1)


# from comet_ml import Experiment

# experiment  = Experiment(api_key="mTvouY9mIy0L8s56g4WVzZhZd")

opt = gan.Options()

opt.cuda = True

# opt.path = 'deletions_2heads_big_wo_WT/'
opt.path = 'deletions_2heads_big_wo_WT_fixed/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000

opt.conditionalD = False
opt.conditional = False

opt.wgangp_lambda = 10.0
# opt.n_classes = 34#44
opt.n_classes1 = 41
opt.n_classes2 = 34
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]
opt.two_labels = True
# opt.test_labels=True

log = Logger(base_dir=opt.path, tag='deletions')

# ['WT', 'alp14', 'cki2', 'efc25', 'fim1', 'for3', 'gef1', 'hob1', 'kin1', 'mal3', 'myo1', 'pal1', 'pck1', 'pck2', 'pmk1', 'pom1', 'ppb1', 'psd1', 'rga1', 'rgf1', 'rho2', 'scd2', 'skb1', 'ssp1', 'sts5', 'sty1', 'tea1', 'tea2', 'tea3', 'tea4', 'tip1', 'nak1', 'scd1', 'shk1', 'sid2']
# wo_deletions = ['alp14', 'cki2', 'efc25', 'fim1', 'for3', 'gef1', 'hob1', 'kin1', 'mal3', 'myo1', 'pal1', 'pck1', 'pck2', 'pmk1', 'pom1', 'ppb1', 'psd1', 'rga1', 'rgf1', 'rho2', 'scd2', 'skb1', 'ssp1', 'sts5', 'sty1', 'tea1', 'tea2', 'tea3', 'tea4', 'tip1', 'nak1', 'scd1', 'shk1', 'sid2']
wo_deletions=[]
wo_deletions=['WT']
# data = datasets.LINDataset(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
data = datasets.LINwithdeletions(transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), wo_deletions=wo_deletions)
# data = datasets.LINDataset(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)

# print(len(data.prt2id))

# print(data.deletions)
#['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1', 'Act1', 'Gef1', 'For3', 'Ra1', 'Scd2', 'Tip1']
mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = LINnet_G(nc=2,nz=100,n_gens=41, n_deletions=34)
netD = LINnet_D(nc=2,BN=True,n_gens=41, n_deletions=34)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(41*34, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')


    y1 = np.repeat(np.arange(41), 34)
    y1 = torch.autograd.Variable(torch.from_numpy(y1))
    if gan.opt.cuda:
        y1 = y1.cuda()

    y2 = np.tile(np.arange(34), 41)
    y2 = torch.autograd.Variable(torch.from_numpy(y2))
    if gan.opt.cuda:
        y2 = y2.cuda()

    noise = save_samples.noise
    # noise = gan.join_xy((noise, y))

    fake = netG(noise, y1, y2)
    # fake = netG(noise)
    
    fake = fake.view(-1, 2, 48, 128)

    fake_01 = torch.FloatTensor(len(fake), 3, 48, 128).fill_(-1)
    fake_01[:,:2,:,:] = (fake.data.cpu() + 1.0) * 0.5
    # print(fake_01.min(), fake_01.max())

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.png'.format(i_iter), nrow=34)
    # alkjfd
    gan.netG.train()


def callback(gan, i_iter):

    # if i_iter % 5000 == 0:
    #     save_inception_score(gan, i_iter)

    if i_iter % 200 == 0:
        torch.save(netD.embedding1.state_dict(), opt.path + 'emb{}.pth'.format(i_iter))
        torch.save(netD.embedding2.state_dict(), opt.path + '2emb{}.pth'.format(i_iter))

    if i_iter % 200 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close() 
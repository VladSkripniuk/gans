import os

import numpy as np

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms

import datasets

import toynet
import mnistnet

import gan
import wgan
import lsgan

from layers.SNConv2d import SNConv2d
from layers.SNLinear import SNLinear

from logger import Logger

from inception_score.model import get_inception_score

opt = gan.Options()

opt.cuda = True

opt.path = 'projection_CIFAR/'
opt.num_iter = 100000
opt.batch_size = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64, BN=True, bias=True, n_classes=10):
        super(Discriminator,self).__init__()

        self.n_classes=n_classes

        self.layer1 = nn.Sequential(SNConv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf),
                               nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(SNConv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf*2),
                               nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8
        self.layer3 = nn.Sequential(SNConv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                               nn.BatchNorm2d(ndf*4),
                               nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
                               # nn.Sigmoid())

        self.embedding = SNLinear(n_classes, ndf*4)
        self.linear = SNLinear(ndf*4, 1)

    def forward(self,input):
        x, y = input
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        h = torch.sum(h, dim=2).sum(dim=2)  # Global pooling
        output = self.linear(h)

        th = torch.cuda if h.is_cuda else torch

        y_onehot = th.FloatTensor(y.size()[0], self.n_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.data.view(-1,1), 1)

        y_onehot = Variable(y_onehot)

        w_y = self.embedding(y_onehot.float())

        output += torch.sum(w_y * h, dim=1).view(-1, 1)

        # out = self.layer4(out)
        return output.view(-1)



opt.conditional = True
opt.conditionalD = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (128,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='GAN')

data = datasets.CIFAR(labeled=True)
# data = datasets.MNISTDataset(selected=None)

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

netG = mnistnet.netG(nc=3, nz=138)
netD = mnistnet.netD(nc=3)


for name, module in netD.named_modules():
    module.name = name


def save_inception_score(gan, i_iter):
    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // 100 + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)

    gan.netG.eval()

    for _ in range(num_batches):
        images = gan.gen_fake_data(100, opt.nz)[0].data.cpu().numpy()
        images = np.rollaxis(images, 1, 4)
        eval_images.append(images)

    gan.netG.train()

    np.random.seed()
    eval_images = np.vstack(eval_images)
    
    eval_images = eval_images[:num_images_to_eval]
    eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    # Calc Inception score
    eval_images = list(eval_images)
    inception_score_mean, inception_score_std = get_inception_score(eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))

    log.add('inception_score', dict(mean=inception_score_mean, std=inception_score_std), i_iter)


def save_samples(gan, i_iter):
    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(64, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')

    fake = gan.gen_fake_data(64, opt.nz, noise=save_samples.noise)
    # fake = next(data_iter)

    fake_01 = (fake[0].data.cpu() + 1.0) * 0.5

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.jpeg'.format(i_iter))
    


def callback(gan, i_iter):
    if i_iter % 10000 == 0:
        save_inception_score(gan, i_iter)

    if i_iter % 50 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(.5, .999))

gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()
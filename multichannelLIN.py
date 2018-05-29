import os

from skimage.io import imread
from skimage import img_as_float

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
import wgan

from logger import Logger

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class multichannel_LIN(Dataset):
    """Points from multiple gaussians"""

    def __init__(self, proteins=['Arp3'], basedir='/home/ubuntu/LIN/LIN_Normalized_WT_size-48-80_train/', transform=None, conditional=False):
        self.images = []
        self.conditional = conditional
        self.prt2id = dict(zip(proteins, range(len(proteins))))

        self.images = []
        self.labels = []

        for protein in proteins:
            self.path = basedir + protein + '/'
            filenames = list(filter(lambda x: (x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')), os.listdir(self.path)))
            self.transform = transform

            for filename in filenames:
                img = imread(self.path + filename)
                # img = resize(img, (24, 40))
                img = img_as_float(img)
                img = np.rollaxis(img[:,:,:2], 2, 0) #.reshape((2, 24, 40))
                img = np.asarray(img, dtype=np.float32)
                img = torch.from_numpy(img)

                if self.transform:
                    img = self.transform(img)

                self.images.append(img)
                self.labels.append(self.prt2id[protein])


        original_images = torch.stack(self.images, dim=0)

        original_labels = torch.LongTensor(self.labels)

        # second_channel = torch.stack([self.images[i][1,:,:] for i in range(len(original_images))], dim=0).view(-1, 1, 48, 80)

        # multichannel_images = torch.FloatTensor(len(original_images), 1 + len(proteins), 48, 80).zero_()
        # multichannel_images[:,0,:,:] = original_images[:,0,:,:]
        # multichannel_images.scatter_(1, original_labels.view(-1,1,1,1).expand(len(original_images), 1, 48, 80) + 1, second_channel)

        # print(multichannel_images.min(), multichannel_images.max())
        # self.x = multichannel_images
        self.x = original_images
        self.y = original_labels
        self.proteins = proteins


        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.conditional:
            return self.images[idx], self.labels[idx]
        else:
            t = torch.FloatTensor(1 + len(self.proteins), 48, 80).zero_()
            t[0,:,:] = self.x[idx,0,:,:]
            t[1+self.y[idx],:,:] = self.x[idx,1,:,:]
            return t



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
                                 nn.Tanh(),
                                 ChannelDrop(nc, protected_channels=[0], out_nonzero_channels=2))
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class LINnet_D(nn.Module):
    def __init__(self,nc=1,ndf=64,BN=True,bias=False): # 128 ok
        super(LINnet_D,self).__init__()
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

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out.view(-1)


opt = gan.Options()

opt.cuda = True

opt.path = 'multiprojection_LIN41SNGAN/'
opt.num_iter = 200000
opt.batch_size = 64

opt.visualize_nth = 2000
opt.conditional = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 150000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='multiGAN')

proteins = os.listdir('../LIN/LIN_Normalized_WT_size-48-80_train/')
# data = multichannel_LIN(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
data = multichannel_LIN(proteins=proteins, transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
# data = datasets.LINDataset(proteins=['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1'], transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)


mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = LINnet_G(nc=42,ngf=64,nz=100)

from mnistnet import LINnet_DSN
netD = LINnet_DSN(nc=42,ndf=64,spec_norm=True)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))




def save_samples(gan, i_iter):
    gan.netG.eval()

    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(64, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')

    fake = gan.gen_fake_data(64, opt.nz, noise=save_samples.noise)
    # fake = next(data_iter)

    fake_list = []

    for i in range(len(fake)):
        for j in range(41):
            img = torch.stack([fake[i,0,:,:], fake[i,j+1,:,:]], dim=0)
            fake_list.append(img)

    fake = torch.FloatTensor(len(fake_list), 3, 48, 80)
    fake_tensor = torch.stack(fake_list, dim=0)
    # print(type(fake_tensor))
    # print(type(fake))
    fake[:,:2,:,:] = fake_tensor.data
    fake[:,2,:,:] = -1

    # fake = next(data_iter)
    # print(fake.min(), fake.max())

    # fake = fake.view(-1, 3, 32, 32)

    fake_01 = (fake.cpu() + 1.0) * 0.5
    # print(fake_01.min(), fake_01.max())

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.png'.format(i_iter), nrow=41)
    gan.netG.train()


def callback(gan, i_iter):

    if i_iter % 50 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()


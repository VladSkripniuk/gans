import numpy as np

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import mnistnet
import datasets

import gan

import os

from torchvision.utils import save_image
from torchvision import transforms


from layers.SNConv2d import SNConv2d
from layers.SNLinear import SNLinear
from layers.separable import ConvTranspose2d_separable



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            m.convt_half.weight.data.normal_(0.0, 0.02)
            m.convt_all.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINnet_G0(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(LINnet_G0,self).__init__()
        self.layer1 = nn.Sequential(ConvTranspose2d_separable(nz,ngf*8,kernel_size=(3, 5), bias=bias, red_portion=50.0/141),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        # 3 x 5
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
        self.layer5 = nn.Sequential(ConvTranspose2d_separable(ngf,nc,kernel_size=4,stride=2,padding=1, bias=bias),
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
        # self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias))#,
        #                          # nn.Sigmoid())

        self.embedding = nn.Linear(n_classes, ndf*4, bias=False)
        self.embedding.weight.data.normal_(0,1)

        self.embedding1 = nn.Linear(41, ndf*4, bias=False)

        self.embedding2 = nn.Linear(35, ndf*4, bias=False)


        # self.linear = nn.Linear(ndf*8, 1, bias=False)

        self.apply(weights_init)

    def forward(self,input):
        x, y = input
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = torch.sum(h, dim=2).sum(dim=2)  # Global pooling
        # output = self.linear(h)

        th = torch.cuda if h.is_cuda else torch

        y_onehot = th.FloatTensor(y.size()[0], self.n_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.data.view(-1,1), 1)

        y_onehot = Variable(y_onehot)

        w_y = self.embedding(y_onehot.float())

        # output += torch.sum(w_y * h, dim=1).view(-1, 1)
        output = torch.sum(w_y * h, dim=1).view(-1, 1)

        return output.view(-1)



netG = LINnet_G0(nc=2,nz=141)
# netD = LINnet_D(nc=2,BN=True,n_classes=44)
netD = LINnet_D(nc=2,BN=True,n_classes=41)

# netG.load_state_dict(torch.load('separableLIN41/gen_100000.pth'))
# netD.load_state_dict(torch.load('separableLIN41/disc_100000.pth'))

# netG.load_state_dict(torch.load('projectionLIN41wonorm/gen_100000.pth'))
# netD.load_state_dict(torch.load('projectionLIN41wonorm/disc_100000.pth'))

# netG.load_state_dict(torch.load('projectionLIN41pairedclasses/gen_10000.pth'))
# netD.load_state_dict(torch.load('projectionLIN41pairedclasses/disc_10000.pth'))



# # ############# fake 
# opt = gan.Options()
# opt.cuda = True
# opt.n_classes = 41
# opt.nz = (100,1,1)


# gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=None, optimizerG=None, opt=opt)

# gan1.netG.eval()

# noise = Variable(gan1.gen_latent_noise(10, opt.nz))
# # print(noise[:,:50].size())
# # print(noise[0:1,:50].expand(10, 50, 1, 1).size())
# # print(noise[:,1,0,0])
# noise.data[:,:50] = noise[0:1,:50].expand(10, 50, 1, 1).data
# # print(noise[0:1, 1])
# # print(noise[:,1,0,0])
# noise = noise.view(1, 10, 100, 1, 1).expand((41, 10, 100, 1, 1)).contiguous().view(410, 100, 1, 1)

# if not os.path.exists('samples_sep/'):
#     os.makedirs('samples_sep/')


# y = np.repeat(np.arange(41), 10)
# y = torch.autograd.Variable(torch.from_numpy(y))
# if gan1.opt.cuda:
#     y = y.cuda()

# noise = gan1.join_xy((noise, y))

# fake = netG(noise)

# fake = fake.view(-1, 2, 48, 80)

# fake_01 = torch.FloatTensor(len(fake), 3, 48, 80).fill_(-1)
# fake_01[:,:2,:,:] = (fake.data.cpu() + 1.0) * 0.5
# # print(fake_01.min(), fake_01.max())

# save_image(fake_01, 'samples_sep/fake_varied.png', nrow=10)
# # alkjfd
# gan1.netG.train()
#######################

# ################ real 
# data = datasets.LINDataset(proteins='all', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), conditional=opt.conditional)
# print(data.proteins)

# import numpy as np

# data.labels = np.array(data.labels)

# fake_01 = torch.FloatTensor(410, 3, 48, 80).fill_(-1)
# fake_01[:,:2,:,:] = (fake.data.cpu() + 1.0) * 0.5

# for i in range(41):

# 	for k, j in enumerate(np.random.permutation(np.where(data.labels==i)[0])[:10]):
# 		fake_01[i*10+k,:2,:,:] = (data.images[j]+1.0)*0.5

# save_image(fake_01, 'samples_sep/real.png', nrow=10)

######################

# netG.load_state_dict(torch.load('separableLIN41/gen_100000.pth'))
# netD.load_state_dict(torch.load('separableLIN41/disc_100000.pth'))

from matplotlib import pyplot as plt

embedding = netD.embedding.weight.data.cpu().numpy()

products = np.sum(embedding[:,np.newaxis,:] * embedding[:,:,np.newaxis], axis=0)

import seaborn as sns
fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(products, annot=True,  linewidths=.5, ax=ax)
# fig.add_subplot(ax)
fig.savefig('test.png')



###################################3


DIR = 'deletions_split1'
DIR = 'projection'
DIR = 'deletions_2heads_big'

netD.embedding2.load_state_dict(torch.load('{}/2emb{}.pth'.format(DIR, 0)))
from copy import copy
embedding0 = copy(netD.embedding2.weight.data.cpu().numpy())

norms = np.zeros((len(range(0, 10000, 200)), 35))
dists = np.zeros((len(range(0, 10000, 200)), 35))

# norms = np.zeros((len(range(0, 1500, 50)), 10))
# dists = np.zeros((len(range(0, 1500, 50)), 10))


#[1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000]

for i, checkpoint in enumerate(range(0, 10000, 200)):
# for i, checkpoint in enumerate(range(0, 1500, 50)):
    # netD.load_state_dict(torch.load('projectionLIN411/disc_{}.pth'.format(checkpoint)))
    netD.embedding2.load_state_dict(torch.load('{}/2emb{}.pth'.format(DIR, checkpoint)))

    from matplotlib import pyplot as plt

    embedding = netD.embedding2.weight.data.cpu().numpy()
    norm = np.sqrt(np.sum(embedding**2, axis=0))
    norms[i,:] = norm

    dist = np.sqrt(np.sum((embedding - embedding0)**2, axis=0))
    dists[i,:] = dist

    # products = np.sum(embedding[:,np.newaxis,:] * embedding[:,:,np.newaxis], axis=0)

    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(20,20)) 
    # sns.heatmap(products, annot=True,  linewidths=.5, ax=ax)
    # # fig.add_subplot(ax)
    # fig.savefig('test.png')


fig, ax = plt.subplots(figsize=(20,20)) 
for i in range(35):
    ax.plot(range(0, 10000, 200), norms[:,i], label=str(i))

# for i in range(10):
    # ax.plot(range(0, 1500, 50), norms[:,i], label=str(i))

# fig.add_subplot(ax)
fig.savefig('{}/norms2.png'.format(DIR))

fig, ax = plt.subplots(figsize=(20,20)) 

for i in range(35):
    ax.plot(range(0, 10000, 200), dists[:,i], label=str(i))

# for i in range(10):
    # ax.plot(range(0, 1500, 50), dists[:,i], label=str(i))
# fig.add_subplot(ax)
fig.savefig('{}/dists2.png'.format(DIR))
print(np.max(dists))


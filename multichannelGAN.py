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

from inception_score.model import get_inception_score

from layers.ChannelDrop import ChannelDrop

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


class multichannel_CIFAR(Dataset):

    def __init__(self, train=True):

        super(multichannel_CIFAR, self).__init__()

        mnist = dset.CIFAR10(root = './data/',
                         transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                          download = True, train=train)

        if train:
            onechannel_images = mnist.train_data#.view(-1, 1, 28, 28)
            original_labels = mnist.train_labels
        else:
            onechannel_images = mnist.test_data#.view(-1, 1, 28, 28)
            original_labels = mnist.test_labels

        onechannel_images = np.rollaxis(onechannel_images, 3, 1)
        
        onechannel_images = torch.FloatTensor(onechannel_images)

        original_labels = torch.LongTensor(original_labels)

        # transform = transforms.Compose([
        #                        transforms.ToPILImage(),
        #                        transforms.Scale(32),
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                    ])

        # onechannel_images = torch.stack([transform(image) for image in onechannel_images], dim=0).view(-1, 1, 3, 32, 32)

        onechannel_images = torch.stack([mnist[i][0] for i in range(len(onechannel_images))], dim=0).view(-1, 1, 3, 32, 32)        

        multichannel_images = torch.FloatTensor(len(onechannel_images), 10, 3, 32, 32)
        multichannel_images.scatter_(1, original_labels.view(-1,1,1,1,1).expand(len(onechannel_images), 1, 3, 32, 32), onechannel_images)

        multichannel_images = multichannel_images.view(len(onechannel_images), 30, 32, 32)
        # print(multichannel_images.min(), multichannel_images.max())
        self.x = multichannel_images

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]



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

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out.view(-1)


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

        self.apply(weights_init)

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


log = Logger(base_dir=opt.path, tag='multiGAN')

data = multichannel_CIFAR()


mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = mnistnet_G(nc=30,nz=100)
netD = mnistnet_D(nc=30,BN=True)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


def save_inception_score(gan, i_iter):
    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // 100 + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)

    gan.netG.eval()

    for _ in range(num_batches):
        index = np.arange(100) * 10 + np.tile(np.arange(10), 10)
        images = gan.gen_fake_data(100, opt.nz).data.cpu().numpy()
        images = images.reshape((-1, 3, 32, 32))
        images = np.rollaxis(images, 1, 4)
        images = images[index]
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

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.jpeg'.format(i_iter))
    # alkjfd
    gan.netG.train()


def callback(gan, i_iter):

    if i_iter % 5000 == 0:
        save_inception_score(gan, i_iter)

    if i_iter % 50 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()
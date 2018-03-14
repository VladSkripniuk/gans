import torch
from torch import optim
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms

import datasets

import mnistnet

import gan
import wgan
import lsgan


from logger import Logger


opt = gan.Options()

opt.cuda = True
opt.num_iter = 20000
opt.batch_size = 64
opt.conditional = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]

basedir = 'oneclass_gans/'

logger = Logger(base_dir=basedir, tag='oneclass_gans')

for digit in range(10):
    opt.path = basedir + '{}/'.format(digit)

    log_t = Logger()


    data = datasets.MNISTDataset(selected=digit)

    mydataloader = datasets.MyDataLoader()
    data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

    netG = mnistnet.mnistnet_G(nz=100, ngf=128)
    netD = mnistnet.mnistnet_D(nc=1, BN=True, ndf=128)
    # netG = mnistnet.mnistnet_G(nz=110)
    # netD = mnistnet.mnistnet_D(nc=11)


    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


    gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

    gan1.train(data_iter, opt, logger=log_t)

    torch.save(netG.state_dict(), opt.path + 'gen.pth')
    torch.save(netD.state_dict(), opt.path + 'disc.pth')

    for key, value in log_t.store.items():
        logger.store['{}_{}'.format(digit, key)] = log_t.store[key]


logger.close()
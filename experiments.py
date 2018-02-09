import torch
from torch import optim
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms

import datasets

import toynet
import mnistnet

import gan
import wgan
import lsgan

from logger import Logger

opt = gan.Options()

opt.cuda = True

opt.path = 'SNGAN/'
opt.num_iter = 100000
opt.batch_size = 64

opt.visualize_nth = 2000
opt.conditional = True
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 5
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='SNGAN')

data = datasets.MNISTDataset(selected=None)


mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
netG = mnistnet.mnistnet_G(nz=100)
netD = mnistnet.mnistnet_DSN(nc=1,BN=True)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


gan1 = wgan.WGANGP(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()
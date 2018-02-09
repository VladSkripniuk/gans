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


opt = gan.Options()

opt.cuda = True
opt.path = 'LIN_gan500k4880/'
opt.num_iter = 500000
opt.batch_size = 64
opt.visualize_nth = 2000
opt.conditional = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (100,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]

data = datasets.LINDataset(protein='Arp3', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
# print(data.path)
# print(data[0].max())
# print(data[0].min())
# dfsdf

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

netG = mnistnet.LINnet_G(nz=100, nc=2)
netD = mnistnet.LINnet_D(nc=2)
# netG = mnistnet.mnistnet_G(nz=110)
# netD = mnistnet.mnistnet_D(nc=11)


optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(.5, .999))


gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

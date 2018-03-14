import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

import mnistnet
import datasets

import gan

import os

from torchvision.utils import save_image
from torchvision import transforms

# netG = mnistnet.LINnet_G(nc=2, nz=100)
netG = mnistnet.mnistnet_G(nc=1, nz=128, ngf=64)
netG = mnistnet.netG(nc=3, nz=128, ngf=64)

# netG.load_state_dict(torch.load('LIN_gan500k/gen_100000.pth'))
# netG.load_state_dict(torch.load('LIN_gan500k4880/gen_500000.pth'))

checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]

basedir = 'oneclass_gans'
basedir = 'SNGAN_nsnfixed'
basedir = 'WGAN_MNIST'
basedir = 'CIFAR'

for i in checkpoints:#range(6):
	netG.load_state_dict(torch.load('{}/gen_{}.pth'.format(basedir, i), map_location=lambda storage, loc: storage))
	netG.eval()

	opt = gan.Options()
	opt.cuda = False
	opt.nz = (128,1,1)
	opt.batch_size = 64

	gan1 = gan.GAN(netG, None, None, None, opt)

	iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None)

	a = next(iterator_fake).data.cpu()

	b = torch.from_numpy(np.zeros((opt.batch_size, 3, 48, 80))-1)
	# b.zero_()
	# b[:,:2,:,:] = a
	b = a
	b = b / 2.0 + 0.5
	save_image(b, '{}/sample_fake{}.jpeg'.format(basedir, i))
	print(b.min(), b.max())


# opt = gan.Options()

# opt.cuda = True
# opt.batch_size = 64
# opt.conditional = False
# opt.n_classes = 10
# opt.nz = (100,1,1)

# data = datasets.LINDataset(protein='Arp3', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
# # print(data.path)
# # print(data[0].max())
# # print(data[0].min())
# # dfsdf

# mydataloader = datasets.MyDataLoader()
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# a = next(data_iter).data.cpu()

# b = torch.from_numpy(np.zeros((opt.batch_size, 3, 48, 80))-1)
# # b.zero_()
# b[:,:2,:,:] = a


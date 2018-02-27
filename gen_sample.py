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

netG = mnistnet.mnistnet_G(nc=1, nz=100, ngf=128)
# netG = mnistnet.LINnet_G(nc=2, nz=106, ngf=64,bias=False)


opt = gan.Options()
opt.cuda = True
opt.n_classes = 6
opt.nz = (100,1,1)
opt.batch_size = 64
opt.conditional = False
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


dirname = 'LIN_wgan500k6/'
dirname = 'LIN_wgan6cond/'
dirname = 'SNGAN/'


for point in opt.checkpoints:
	# netG.load_state_dict(torch.load('LIN_gan500k/gen_100000.pth'))
	netG.load_state_dict(torch.load('{}gen_{}.pth'.format(dirname, point)))
	netG.eval()

	gan1 = gan.GAN(netG, None, None, None, opt)

	iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None)

	a = next(iterator_fake)[0].data.cpu()

	# b = torch.from_numpy(np.zeros((opt.batch_size, 3, 48, 80))-1)
	# b.zero_()
	# b[:,:2,:,:] = a
	# b=a

	# b = b / 2.0 + 0.5
	save_image(a, '{}/sample_fake{}.jpeg'.format(dirname, point))



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

# b = b / 2.0 + 0.5
# save_image(b, 'sample_real.jpeg')


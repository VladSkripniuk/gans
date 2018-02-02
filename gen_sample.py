import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

import mnistnet
import datasets

import gan

from torchvision.utils import save_image
from torchvision import transforms

netG = mnistnet.LINnet_G(nc=2, nz=100)

# netG.load_state_dict(torch.load('LIN_gan500k/gen_100000.pth'))
netG.load_state_dict(torch.load('LIN_gan500k4880/gen_5000.pth'))
netG.eval()

opt = gan.Options()
opt.cuda = True
opt.nz = (100,1,1)
opt.batch_size = 64

gan1 = gan.GAN(netG, None, None, None, opt)

iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None)

a = next(iterator_fake).data.cpu()

b = torch.from_numpy(np.zeros((opt.batch_size, 3, 48, 80))-1)
# b.zero_()
b[:,:2,:,:] = a
b = b / 2.0 + 0.5
save_image(b, 'sample_fake.jpeg')


opt = gan.Options()

opt.cuda = True
opt.batch_size = 64
opt.conditional = False
opt.n_classes = 10
opt.nz = (100,1,1)

data = datasets.LINDataset(protein='Arp3', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
# print(data.path)
# print(data[0].max())
# print(data[0].min())
# dfsdf

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

a = next(data_iter).data.cpu()

b = torch.from_numpy(np.zeros((opt.batch_size, 3, 48, 80))-1)
# b.zero_()
b[:,:2,:,:] = a

b = b / 2.0 + 0.5
save_image(b, 'sample_real.jpeg')


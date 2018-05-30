import toynet
import gan
import torch
from torch import optim
import wgan
import lsgan

from torch.utils.data import DataLoader
import datasets

import mnistnet

class Options:
    def __init__(self):
        self.cuda = False
        self.batch_size = 256
        self.nz = 2
        self.num_iter = 50
        self.num_disc_iters = 10
        self.wgangp_lambda = 0.1
        self.visualize_nth = 10
        self.n_classes = 4
        self.conditional = False
        
opt = Options()

################################# GAN
# import datasets
# from torch.utils.data import DataLoader


# grid = [-20, -10, 0, 10, 20]
# mean_list = []

# for x in grid:
#     for y in grid:
#         mean_list.append([x, y])

# data = datasets.GaussianMixtureDataset(mean_list=mean_list, component_size_list=[100]*len(mean_list))

# mydataloader = datasets.MyDataLoader()
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda)


# opt.path = 'gan251/'
# opt.num_iter = 6000
# opt.visualize_nth = 500
# opt.nz = (2,)

# netG = toynet.toynet_G([2, 512, 512, 512, 2])
# netD = toynet.toynet_D([2, 512, 512, 512, 1])

# optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(.5, .9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(.5, .9))

# gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

# gan1.train(data_iter, opt)

#################################### CGAN
# import datasets
# from torch.utils.data import DataLoader


# grid = [-20, -10, 0, 10, 20]
# mean_list = []

# for x in grid:
#     for y in grid:
#         mean_list.append([x, y])

# data = datasets.ConditionalGaussianMixtureDataset(mean_list=mean_list, component_size_list=[100]*len(mean_list), component_class_list=[0, 1, 2, 3, 4] * 5, n_classes=5)

# mydataloader = datasets.MyConditionalDataLoader(n_classes=5)
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda)


# opt.path = 'cgan25/'
# opt.num_iter = 6000
# opt.visualize_nth = 500
# opt.conditional = True
# opt.n_classes = 5

# netG = toynet.toynet_G([7, 512, 512, 512, 2])
# netD = toynet.toynet_D([7, 512, 512, 512, 1])

# optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(.5, .9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(.5, .9))

# gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

# gan1.train(data_iter, opt)

# ################################### LSGAN
# import datasets
# from torch.utils.data import DataLoader


# grid = [-20, -10, 0, 10, 20]
# grid = [-5, 5]
# mean_list = []

# for x in grid:
#     for y in grid:
#         mean_list.append([x, y])

# data = datasets.GaussianMixtureDataset(mean_list=mean_list, component_size_list=[500]*len(mean_list))

# mydataloader = datasets.MyDataLoader()
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda)


# opt.path = 'lsgan251/'
# opt.num_iter = 6000
# opt.visualize_nth = 500
# opt.conditional = False
# opt.n_classes = 5
# opt.nz = (2,)

# netG = toynet.toynet_G([2, 512, 512, 512, 2])
# netD = toynet.toynet_D([2, 512, 512, 512, 1])

# optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(.5, .9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(.5, .9))

# gan1 = lsgan.LSGAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

# gan1.train(data_iter, opt)


#####################################    MNIST

# opt.path = 'ganMNIST/'
# opt.num_iter = 2000
# opt.batch_size = 64
# opt.visualize_nth = 100
# opt.conditional = False
# opt.n_classes = 5
# opt.nz = (1024, 1, 1)

# import datasets
# data = datasets.MNISTDataset()

# mydataloader = datasets.MyDataLoader()
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda)


# netG = mnistnet.mnistnet_G()
# netD = mnistnet.mnistnet_D()

# optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(.5, .9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(.5, .9))

# gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

# gan1.train(data_iter, opt)

############################## WGAN GP


# from torch.utils.data import DataLoader


# grid = [-20, -10, 0, 10, 20]

# mean_list = []

# for x in grid:
#     for y in grid:
#         mean_list.append([x, y])

# data = datasets.GaussianMixtureDataset(mean_list=mean_list, component_size_list=[100]*len(mean_list))

# mydataloader = datasets.MyDataLoader()
# data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda)


# opt.path = 'wgangp25/'
# opt.num_iter = 6000
# opt.visualize_nth = 500
# opt.conditional = False
# opt.n_classes = 5
# opt.nz = (2,)

# netG = toynet.toynet_G([2, 512, 512, 512, 2])
# netD = toynet.toynet_D([2, 512, 512, 512, 1])

# optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(.5, .9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(.5, .9))

# gan1 = wgan.WGANGP(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

# gan1.train(data_iter, opt)

################################ CWGANGP
import datasets
from torch.utils.data import DataLoader

opt.conditional = True
opt.path = 'cwgangp25/'
opt.num_iter = 6000
opt.visualize_nth = 500
opt.n_classes = 5
opt.nz = (2,)

grid = [-20, -10, 0, 10, 20]
mean_list = []

for x in grid:
    for y in grid:
        mean_list.append([x, y])

data = datasets.ConditionalGaussianMixtureDataset(mean_list=mean_list, component_size_list=[100]*len(mean_list), component_class_list=[0, 1, 2, 3, 4] * 5, n_classes=5)

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1), is_cuda=opt.cuda, conditional=True, n_classes=opt.n_classes)

# print(next(data_iter).size())


netG = toynet.toynet_G([7, 512, 512, 512, 2])
netD = toynet.toynet_D([7, 512, 512, 512, 1])

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(.5, .9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(.5, .9))

gan1 = wgan.WGANGP(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt)

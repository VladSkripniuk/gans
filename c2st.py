from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

from tqdm import tqdm

import torch

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms, utils

from torchvision.utils import save_image

import gan
import wgan

import mnistnet
from mnistnet import Generator, Discriminator

import copy

N_ATTEMPTS = 10
N_EPOCHS = 5

class modifiedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0].cuda()

def variableIter(dataloader):
    for batch in dataloader:
        yield Variable(batch).cuda()

def c2st(netG, netG_path, netD_0, gan_type, opt, real_dataset):

    netG.load_state_dict(torch.load(netG_path))
    netG.eval()

    gan1 = gan.GAN(netG, None, None, None, opt)

    iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None)

    data = real_dataset
    # data = datasets.MNIST(root = './data/',
    #                          transform=transforms.Compose([
    #                                transforms.Scale(32),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                            ]),
    #                           download = True, train=False)

    # data = modifiedDataset(data)

    random_state = [23, 42, 180, 34, 194, 3424, 234, 23423, 221, 236]

    roc_list = []
    loss_list = []

    for attempt in range(N_ATTEMPTS):

        train_indices, test_indices = train_test_split(range(len(data)), test_size=0.1, random_state=random_state[attempt])

        netD = copy.deepcopy(netD_0)
        # netD = mnistnet.Discriminator()
        netD.train()
        optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
        gan_t = gan_type(None, netD, optimizerD, None, opt)


        for _ in range(N_EPOCHS):
            iterator_real = variableIter(DataLoader(data, sampler=SubsetRandomSampler(train_indices), batch_size=opt.batch_size))
            print(len(data))
            for i_iter in tqdm(range(180)):
                gan_t.train_D_one_step(iterator_real, iterator_fake)

        netD.eval()

        iterator_real = variableIter(DataLoader(data, sampler=SubsetRandomSampler(test_indices), batch_size=opt.batch_size))

        err = 0

        loss = []
        y_true = []
        y_score = []

        for i in range(20):
            batch_real = next(iterator_real)
            batch_fake = next(iterator_fake)

            if gan_type == wgan.WGANGP:
                y_true = y_true + [0] * batch_real.size()[0]
                y_true = y_true + [1] * batch_real.size()[0]
            else:
                y_true = y_true + [1] * batch_real.size()[0]
                y_true = y_true + [0] * batch_real.size()[0]


            y_score = y_score + list(gan_t.netD(batch_real).cpu().data.numpy())
            y_score = y_score + list(gan_t.netD(batch_fake).cpu().data.numpy())

            loss.append(float(gan_t.compute_disc_score(batch_real, batch_fake).data.cpu().numpy()))

        loss = np.mean(loss)
        roc = roc_auc_score(y_true, y_score)

        loss_list.append(loss)
        roc_list.append(roc)

    return loss_list, roc_list

# opt = gan.Options()
# opt.cuda = True
# opt.nz = (100,1,1)
# opt.batch_size = 50

# print(c2st(Generator(), 'wgan_test/gen_1000.pth', Discriminator(), wgan.WGANGP, opt))
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

import datasets


N_ATTEMPTS = 10
N_EPOCHS = 10
N_ITER = 1000

from datasets import MNISTDataset

from copy import copy, deepcopy

def varIter(data, opt):
    for batch in data:
        if opt.cuda:
            yield Variable(batch).cuda()
        else:
            yield Variable(batch)


def c2st(netG, netG_path, netD_0, gan_type, opt, real_dataset, selected=None, logger=None):

    netG.load_state_dict(torch.load(netG_path))
    netG.eval()

    gan1 = gan.GAN(netG, None, None, None, opt)
    opt = copy(opt)
    opt.conditional = False

    data = real_dataset

    if selected is not None:
        iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None, selected=selected, drop_labels=True)
    else:
        iterator_fake = gan1.fake_data_generator(opt.batch_size, opt.nz, None)

    random_state = [23, 42, 180, 34, 194, 3424, 234, 23423, 221, 236]

    roc_list = []
    loss_list = []

    for attempt in range(N_ATTEMPTS):

        train_indices, test_indices = train_test_split(range(len(data)), test_size=0.1, random_state=random_state[attempt])

        netD = deepcopy(netD_0)
        # netD = mnistnet.Discriminator()
        netD.train()
        optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(.5, .999))
        gan_t = gan_type(None, netD, optimizerD, None, opt)


        # for _ in range(N_EPOCHS):
        #     iterator_real = varIter(DataLoader(data, sampler=SubsetRandomSampler(train_indices), batch_size=opt.batch_size), opt)
        #     for i_iter in tqdm(range(int(len(train_indices) / opt.batch_size))):
        #         gan_t.train_D_one_step(iterator_real, iterator_fake)

        iterator_real = datasets.MyDataLoader().return_iterator(
            DataLoader(data, sampler=SubsetRandomSampler(train_indices),
                batch_size=opt.batch_size), is_cuda=opt.cuda,
            conditional=opt.conditional, n_classes=opt.n_classes)

        for i_iter in tqdm(range(N_ITER)):
            loss, _, _ = gan_t.train_D_one_step(iterator_real, iterator_fake)
            if logger is not None:
                logger.add('disc_loss{}'.format(attempt), loss, i_iter)


        gan_t.save(attempt)

        # iterator_real = varIter(DataLoader(data, sampler=SubsetRandomSampler(test_indices), batch_size=opt.batch_size), opt)

        iterator_real = datasets.MyDataLoader().return_iterator(
            DataLoader(data, sampler=SubsetRandomSampler(test_indices),
                batch_size=opt.batch_size), is_cuda=opt.cuda,
            conditional=opt.conditional, n_classes=opt.n_classes)

        err = 0

        loss = []
        y_true = []
        y_score = []

        for i in range(int(len(test_indices) / opt.batch_size)):
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
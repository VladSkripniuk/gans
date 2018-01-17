import sys
import math

import torch
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as vutils

from gan import GAN_base

class LSGAN(GAN_base):
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        GAN_base.__init__(self, netG, netD, optimizerD, optimizerG, opt)

        # criterion for training
        self.criterion = self.LSGANLoss
        self.real_label = 1
        self.fake_label = -1
        self.generator_label = 1  # fake labels are real for generator cost

        shift = torch.ones(opt.batch_size)
        self.shift = Variable(shift.cuda()) if self.is_cuda else Variable(shift)

    def compute_disc_score(self, data_a, data_b):
        th = torch.cuda if self.is_cuda else torch

        if self.conditional:
            data_a = self.join_xy(data_a)
            data_b = self.join_xy(data_b)

        scores_a = self.netD(data_a)
        scores_b = self.netD(data_b)

        labels_a = Variable(th.FloatTensor(scores_a.size(0)).fill_(self.real_label))
        errD_a = self.criterion(scores_a, labels_a)

        labels_b = Variable(th.FloatTensor(scores_b.size(0)).fill_(self.fake_label))
        errD_b = self.criterion(scores_b, labels_b)
        
        errD = errD_a + errD_b
        return errD

    def compute_gen_score(self, data):
        th = torch.cuda if self.is_cuda else torch

        if self.conditional:
            data = self.join_xy(data)

        scores = self.netD(data)
        labels = Variable(th.FloatTensor(scores.size()).fill_(self.generator_label))
        errG = self.criterion(scores, labels)
        return errG

    def LSGANLoss(self, scores, labels):
        loss = torch.nn.MSELoss(size_average=True)

        return loss(scores * labels, self.shift[:labels.size()[0]])
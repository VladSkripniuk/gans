import os

import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms

import datasets

import toynet
import mnistnet

import gan
import wgan
import lsgan

from logger import Logger

from inception_score.model import get_inception_score

opt = gan.Options()

opt.cuda = True

opt.path = 'SNGAN_CIFAR/'
opt.num_iter = 100000
opt.batch_size = 64

<<<<<<< HEAD
=======
opt.visualize_nth = 2000
>>>>>>> 9e95b5e74e0fd776cdc9e507081d5cb6580a5034
opt.conditional = False
opt.wgangp_lambda = 10.0
opt.n_classes = 10
opt.nz = (128,1,1)
opt.num_disc_iters = 1
opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]


log = Logger(base_dir=opt.path, tag='GAN')

data = datasets.CIFAR()
# data = datasets.MNISTDataset(selected=None)

mydataloader = datasets.MyDataLoader()
data_iter = mydataloader.return_iterator(DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4), is_cuda=opt.cuda, conditional=opt.conditional, pictures=True)

# netG = mnistnet.Generator(nz=100, BN=True)
# netD = mnistnet.Discriminator(nc=1, BN=True)
# netG = mnistnet.mnistnet_G(nz=128, ngf=64)
# netD = mnistnet.mnistnet_D(nc=1,ndf=64)

netG = mnistnet.netG(nc=3, nz=128)
netD = mnistnet.netD(nc=3,spec_norm=True)


# def save_sigma(module, input, output):
#     # name = list(module.named_modules())[0]
#     name = module.name
#     # print(name)
#     sigma_approx = module.sigma_approx.data.cpu().numpy()[0]
#     # sigma_true = module.sigma_true.data.cpu().numpy()[0]
#     sigma_true = 1
#     log.add('{}_sigma_approx'.format(name), sigma_approx, None)
#     log.add('{}_sigma_rel'.format(name), sigma_true/sigma_approx, None)
#     # sigmas['{}_sigma_approx'.format(name)] = sigmas.get('{}_sigma_approx'.format(name), ()) + (module.sigma_approx,)
#     # sigmas['{}_sigma_rel'.format(name)] = sigmas.get('{}_sigma_rel'.format(name), ()) + (module.sigma_true/module.sigma_approx,)

# def save_grad(module, grad_input, grad_output):
#     pass
#     # print(grad_input[1][0,0,0,0])
#     # grad_input[1].data.norm()
#     # name = module.name
    # log.add('{}_gradnorm'.format(name), grad_input[1].norm().cpu().data.numpy(), None)


# def register_hooks(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.register_forward_hook(save_sigma)
#         m.register_backward_hook(save_grad)
# netD.apply(register_hooks)
# print(list(netD.named_modules()))
for name, module in netD.named_modules():
    module.name = name


def save_inception_score(gan, i_iter):
    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // 100 + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)

    gan.netG.eval()

    for _ in range(num_batches):
        images = gan.gen_fake_data(100, opt.nz).data.cpu().numpy()
        images = np.rollaxis(images, 1, 4)
        eval_images.append(images)

    gan.netG.train()

    np.random.seed()
    eval_images = np.vstack(eval_images)
    
    eval_images = eval_images[:num_images_to_eval]
    eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    # Calc Inception score
    eval_images = list(eval_images)
    inception_score_mean, inception_score_std = get_inception_score(eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))

    log.add('inception_score', dict(mean=inception_score_mean, std=inception_score_std), i_iter)


def save_samples(gan, i_iter):
    if 'noise' not in save_samples.__dict__:
        save_samples.noise = Variable(gan.gen_latent_noise(64, opt.nz))

    if not os.path.exists(opt.path + 'tmp/'):
        os.makedirs(opt.path + 'tmp/')

    fake = gan.gen_fake_data(64, opt.nz, noise=save_samples.noise)
    fake = next(data_iter)

    fake_01 = (fake.data.cpu() + 1.0) * 0.5

    save_image(fake_01, opt.path + 'tmp/' + '{:0>5}.jpeg'.format(i_iter))
    adfadf


def callback(gan, i_iter):
    # if i_iter % 5000 == 0:
        # save_inception_score(gan, i_iter)

    if i_iter % 50 == 0:
        save_samples(gan, i_iter)

    if i_iter % 50 == 0:
        log.save()


optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(.5, .999))

gan1 = gan.GAN(netG=netG, netD=netD, optimizerD=optimizerD, optimizerG=optimizerG, opt=opt)

gan1.train(data_iter, opt, logger=log, callback=callback)

torch.save(netG.state_dict(), opt.path + 'gen.pth')
torch.save(netD.state_dict(), opt.path + 'disc.pth')

log.close()
import numpy as np
from tensorboardX import SummaryWriter


from logger import Logger

from c2st import c2st

from mnistnet import Generator, Discriminator, mnistnet_D, mnistnet_G, LINnet_D, LINnet_G

import gan
import wgan

import datasets

from torchvision import transforms

opt = gan.Options()
opt.cuda = True
opt.nz = (100,1,1)
opt.batch_size = 64
opt.path = '100kGAN_GANc2st/'

writer = SummaryWriter('100kGAN_GANc2st')

opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]

logger = Logger(base_dir=opt.path, tag='100kGAN_GANc2st')

for checkpoint in opt.checkpoints:

    real_dataset = datasets.MNISTDataset(selected=None, train=False)

    # real_dataset = datasets.LINDataset(protein='Arp3', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     basedir='/home/ubuntu/LIN/LIN_Normalized_WT_size-48-80_train/')

    log_t = Logger()

    loss, roc = c2st(mnistnet_G(nc=1), 'DCGAN100k128/gen_{}.pth'.format(checkpoint), mnistnet_D(nc=1), gan.GAN, opt, real_dataset, logger=log_t)


    loss = sorted(loss)
    roc = sorted(roc)

    for key, value in log_t.store.items():
        logger.store['{}_{}'.format(checkpoint, key)] = log_t.store[key]

    logger.add('c2st_loss', {"0.1":loss[1],
                                "0.5":loss[4],
                                "0.9":loss[8]}, checkpoint)

    logger.add('c2st_roc', {"0.1":roc[1],
                                "0.5":roc[4],
                                "0.9":roc[8]}, checkpoint)

    logger.save()


    writer.add_scalars('loss', {"0.1":loss[1],
                                "0.5":loss[4],
                                "0.9":loss[8]}, 10 * np.log10(checkpoint/1000))

    writer.add_scalars('roc', {"0.1":roc[1],
                                "0.5":roc[4],
                                "0.9":roc[8]}, 10 * np.log10(checkpoint/1000))


writer.close()
logger.close()
import numpy as np
from tensorboardX import SummaryWriter


from c2st import c2st

from mnistnet import Generator, Discriminator, mnistnet_D, mnistnet_G, LINnet_D, LINnet_G

import gan
import wgan

import datasets

from torchvision import transforms

from logger import Logger

opt = gan.Options()
opt.cuda = True
opt.nz = (100,1,1)
opt.batch_size = 50

for i in range(5,6):

    # opt.path = 'test_GAN_GANc2st/'.format(i)
    opt.path = 'test_GAN_GANc2st/'

    writer = SummaryWriter(opt.path)

    opt.checkpoints = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 500000]
    opt.checkpoints = [1000, 2000, 4000, 8000, 12000, 16000, 20000]

    logger = Logger(base_dir=opt.path, tag=str(i))

    for checkpoint in opt.checkpoints:

        real_dataset = datasets.MNISTDataset(selected=i, train=False)
        # print(len(real_dataset))

        # real_dataset = datasets.LINDataset(protein='Arp3', transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     basedir='/home/ubuntu/LIN/LIN_Normalized_WT_size-48-80_train/')

        log_t = Logger()

        loss, roc = c2st(mnistnet_G(nc=1), 'oneclass_gans/{}/gen_{}.pth'.format(i, checkpoint), mnistnet_D(nc=1), gan.GAN, opt, real_dataset, logger=log_t)

        loss = sorted(loss)
        roc = sorted(roc)

        # writer.add_scalars('loss', {"0.1":loss[1],
        #                             "0.5":loss[4],
        #                             "0.9":loss[8]}, 10 * np.log10(checkpoint/1000))

        # writer.add_scalars('roc', {"0.1":roc[1],
        #                             "0.5":roc[4],
        #                             "0.9":roc[8]}, 10 * np.log10(checkpoint/1000))


        # print(log_t.store)
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
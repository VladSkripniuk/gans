import numpy as np
from tensorboardX import SummaryWriter


from c2st import c2st

from mnistnet import Generator, Discriminator

import gan
import wgan

opt = gan.Options()
opt.cuda = True
opt.nz = (100,1,1)
opt.batch_size = 50

writer = SummaryWriter('WGAN_WGANc2st')

for checkpoint in [1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000]:

    loss, roc = c2st(Generator(), 'wgan_test/gen_{}.pth'.format(checkpoint), Discriminator(), wgan.WGANGP, opt)

    loss = sorted(loss)
    roc = sorted(roc)

    writer.add_scalars('loss', {"0.1":loss[1],
                                "0.5":loss[4],
                                "0.9":loss[8]}, np.log10(checkpoint))

    writer.add_scalars('roc', {"0.1":roc[1],
                                "0.5":roc[4],
                                "0.9":roc[8]}, np.log10(checkpoint))

writer.close()
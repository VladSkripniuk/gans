# GANs for biological image synthesis

This repository contains implementations of vanilla [GAN](https://arxiv.org/abs/1406.2661), [Least Squares GAN](https://arxiv.org/abs/1611.04076), [Wasserstein GAN with Gradient Penalty](https://arxiv.org/abs/1704.00028), [Spectral Normalization GAN](https://arxiv.org/abs/1802.05957) and [cGAN with projection discriminator](https://arxiv.org/abs/1802.05637). Everything was done as a part of this [project](https://github.com/skripniuk/gans/blob/master/imgs/Thesis.pdf).

We validated all models on a dataset of 25 gaussians (samples from WGAN-GP):

![wgan100k](https://github.com/skripniuk/gans/blob/master/imgs/wgan100k.jpg)

## LIN dataset

LIN dataset contains photographs of 41 proteins in fission yeast cells. To visualize similar proteins we used [FID](https://arxiv.org/abs/1706.08500) metric.

![FID](https://github.com/skripniuk/gans/blob/master/imgs/Screenshot%20from%202018-05-29%2011-28-41.png)

## cGAN with projection discriminator
(left - real photographs, right - generated).

![LIN41](https://github.com/skripniuk/gans/blob/master/imgs/lin41.png)

Quntatitative comparison of different models.

<p align="center">
<img src="https://github.com/skripniuk/gans/blob/master/imgs/Screenshot%20from%202018-05-29%2011-30-12.png">
</p>

## Multichannel GAN

Samples from multichannel GAN (all details can be found [here](https://github.com/skripniuk/gans/blob/master/imgs/Thesis.pdf))

![multi](https://github.com/skripniuk/gans/blob/master/imgs/Screenshot%20from%202018-05-29%2011-29-34.png)



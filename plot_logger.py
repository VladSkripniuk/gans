from logger import Logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os

filename = 'logs/test'
filename = 'test_GAN_GANc2st/5'
filename = '100kLINGAN_GANc2st/100kLINGAN_GANc2st'
filename = '100kLINWGAN_GANc2st/100kLINWGAN_GANc2st'
filename = 'SNGAN_GANc2st/SNGAN_GANc2st'
filename = 'LIN_wgan500k6/LIN_wgan500k6'
filename = 'SNGAN_net_wo_spec/SNGAN'
filename = 'CIFAR/CIFAR'
filename = 'deletions2small/deletions'

log = Logger()
log.load(filename)

for key, value in log.store.items():
	fig, ax = plt.subplots(nrows=1, ncols=1)
	# print(type(value['values'][0]))
	if type(value['values'][0]) is not dict:
		if value['steps'][0] is None:
			ax.plot(range(len(value['values'])), value['values'])
		else:
			ax.plot(value['steps'][:4000], value['values'][:4000])
			# ax.set_ylim((0,10))
	elif '0.1' in value['values'][0].keys():
		v1 = []
		v5 = []
		v9 = []
		for i in range(len(value['values'])):
			v1.append(value['values'][i]['0.1'])
			v5.append(value['values'][i]['0.5'])
			v9.append(value['values'][i]['0.9'])

		ax.plot(value['steps'], v1)
		ax.plot(value['steps'], v5)
		ax.plot(value['steps'], v9)

	else:
		means = []
		for i in range(len(value['values'])):
			means.append(value['values'][i]['mean'])

		ax.plot(value['steps'], means)
		

	fig.savefig(os.path.join(os.path.dirname(filename), '{}_{}.jpeg'.format(key, log.tag)))

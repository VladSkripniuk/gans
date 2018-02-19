from logger import Logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os

filename = 'logs/test'
filename = 'test_GAN_GANc2st/5'

log = Logger()
log.load(filename)

for key, value in log.store.items():
	fig, ax = plt.subplots(nrows=1, ncols=1)
	# print(type(value['values'][0]))
	if type(value['values'][0]) is not dict:
		ax.plot(value['steps'], value['values'])
	else:
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

	fig.savefig(os.path.join(os.path.dirname(filename), '{}_{}'.format(key, log.tag)))

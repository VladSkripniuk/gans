from logger import Logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os

filename = 'logs/test'

log = Logger()
log.load(filename)

for key, value in log.store.items():
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(value['steps'], value['values'])
	fig.savefig(os.path.join(os.path.dirname(filename), '{}_{}'.format(key, log.tag)))
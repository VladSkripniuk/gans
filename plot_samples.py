from matplotlib import pyplot as plt

from skimage.io import imread

fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(30,30))

DIR = 'deletionsSN'

for i in range(20):
	im = imread('{}/tmp/{:0>5}.png'.format(DIR, i*200))

	ax[i // 5][i % 5].imshow(im[:48*4,:128*4])
	ax[i // 5][i % 5].set_title(i*200)

fig.savefig('{}/samples.png'.format(DIR))
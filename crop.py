# from matplotlib import pyplot as plt
# import torch
# from torch.utils.data import DataLoader

# import numpy as np

# import datasets

# data = datasets.LINwithdeletions()

# mydataloader = datasets.MyDataLoader()
# data_iter = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)

# y_list = []
# x_list = []

# cnt = 0

# for x, g, d in data_iter:
# 	# print(x.size())
# 	# print(x.min(), x.max())
# 	x = x[0]
# 	# print((x.sum(dim=0).sum(dim=0) > 0).numpy().shape)
# 	# print(x.sum(dim=0).sum(dim=0))
# 	x1 = np.argmax((x.sum(dim=0).sum(dim=0) > 0).numpy())
# 	x2 = np.argmax(1 - (x.sum(dim=0).sum(dim=0) > 0)[256:].numpy())+256
# 	# print(x1, x2)
# 	y1 = np.argmax((x.sum(dim=0).sum(dim=1) > 0).numpy())
# 	y2 = np.argmax(1 - (x.sum(dim=0).sum(dim=1) > 0)[64:].numpy())+64
	
# 	x_list.append(x1)
# 	x_list.append(x2)

# 	y_list.append(y1)
# 	y_list.append(y2)

# 	if x1 < 150:
# 		fig, ax = plt.subplots(figsize=(20,20), nrows=1, ncols=1) 
# 		img = np.zeros((128, 512, 3))
# 		img[:,:,:2] = np.rollaxis(x.numpy(), 0, 3)
# 		ax.imshow(img)
# 		# fig.add_subplot(ax)
# 		fig.savefig('{}.png'.format(cnt))
# 		cnt += 1
# 	elif x2 > 350:
# 		fig, ax = plt.subplots(figsize=(20,20), nrows=1, ncols=1) 
# 		img = np.zeros((128, 512, 3))
# 		img[:,:,:2] = np.rollaxis(x.numpy(), 0, 3)
# 		ax.imshow(img)
# 		# fig.add_subplot(ax)
# 		fig.savefig('{}.png'.format(cnt))
# 		cnt += 1

# 	if y1 < 32:
# 		fig, ax = plt.subplots(figsize=(20,20), nrows=1, ncols=1) 
# 		img = np.zeros((128, 512, 3))
# 		img[:,:,:2] = np.rollaxis(x.numpy(), 0, 3)
# 		ax.imshow(img)
# 		# fig.add_subplot(ax)
# 		fig.savefig('{}.png'.format(cnt))
# 		cnt += 1
# 	elif y2 > 96:
# 		fig, ax = plt.subplots(figsize=(20,20), nrows=1, ncols=1) 
# 		img = np.zeros((128, 512, 3))
# 		img[:,:,:2] = np.rollaxis(x.numpy(), 0, 3)
# 		ax.imshow(img)
# 		# fig.add_subplot(ax)
# 		fig.savefig('{}.png'.format(cnt))
# 		cnt += 1



from matplotlib import pyplot as plt
# fig, ax = plt.subplots(figsize=(20,20), nrows=1, ncols=2) 

# ax[0].hist(x_list, bins=100)
# ax[1].hist(y_list, bins=100)
# # fig.add_subplot(ax)
# fig.savefig('hist.png')

# print(min(x_list), max(x_list))
# print(min(y_list), max(y_list))

# print(len(data.prt2id))


from skimage.io import imread, imsave
from skimage import img_as_float

from skimage.transform import resize

import os
import numpy as np

basedir = '/home/ubuntu/LIN_deletions/LIN_Normalized_all_size-128-512_train/'
basedirt = '/home/ubuntu/LIN_deletions_cropped/'

if not os.path.exists(basedirt):
	os.makedirs(basedirt)

pairs = os.listdir(basedir)
pairs = sorted(pairs)

proteins = []
deletions = []

for pair in pairs:
    proteins.append(pair[2:].split('_D_')[0])
    # proteins.append(pair[2:].split('_D_')[1])
    deletions.append(pair[2:].split('_D_')[1])

proteins = sorted(set(proteins))
deletions = sorted(set(deletions))

images = []
prt2id = dict(zip(proteins, range(len(proteins))))
del2id = dict(zip(deletions, range(len(deletions))))


print(proteins)
print(deletions)

gens = []
deletions = []

from time import time

from tqdm import tqdm

gens_counter = dict()
deletions_counter = dict()

cnt = np.zeros((41, 35))

for pair in tqdm(pairs):
    path = basedir + pair + '/'
    filenames = list(filter(lambda x: (x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')), os.listdir(path)))

    if not os.path.exists(basedirt + pair + '/'):
        os.makedirs(basedirt + pair + '/')

    gen = pair[2:].split('_D_')[0]
    deletion = pair[2:].split('_D_')[1]

    gens_counter[gen] = gens_counter.get(gen, 0) + len(filenames)
    deletions_counter[deletion] = deletions_counter.get(deletion, 0) + len(filenames)

    cnt[prt2id[gen], del2id[deletion]] = len(filenames)

    for filename in filenames:
        # s = time()
        img = imread(path + filename)
        img = img_as_float(img)
        # print(time()-s)
        # s = time()

        img = img[16:-16,128:-128,:]
        img = resize(img, (48, 128))

        imsave(basedirt + pair + '/' + filename, img)
        

# for a in sorted(list(gens_counter.keys())[:6]):
#     print(a, gens_counter[a])

# for a in sorted(list(deletions_counter.keys())[:6]):
#     print(a, deletions_counter[a])


# import seaborn as sns
# fig, ax = plt.subplots(figsize=(40,40)) 
# sns.heatmap(cnt[:,1:], annot=True,  linewidths=.5, ax=ax)
# # fig.add_subplot(ax)
# fig.savefig('class_freq.png')

# print(np.sum(cnt==0))


# from matplotlib import pyplot as plt

# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(30, 15    ))

# # print(len(gens_counter.keys()))
# # print(gens_counter.keys())

# print(len(deletions_counter.keys()))
# print(deletions_counter.keys())

# print(list(set(gens_counter.keys()) - set(deletions_counter.keys())))
# print(list(set(deletions_counter.keys())-set(gens_counter.keys())))

# ax[0].bar(gens_counter.keys(), gens_counter.values())
# ax[1].bar(deletions_counter.keys(), deletions_counter.values())

# fig.savefig('classes.png')
import torch
from torchvision.utils import save_image

from matplotlib import pyplot as plt


from skimage.io import imread, imsave
from skimage import img_as_float

from skimage.transform import resize

import os
import numpy as np

# basedir = '/home/ubuntu/LIN_deletions/LIN_Normalized_all_size-128-512_train/'
basedir = '/home/ubuntu/LIN_deletions_cropped/'

# if not os.path.exists(basedir):
# 	os.makedirs(basedir)

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

gens = []
deletions = []

from time import time

from tqdm import tqdm

gens_counter = dict()
deletions_counter = dict()

cnt = np.zeros((41, 35))

deletion_filenames = dict()

for pair in tqdm(pairs):
    path = basedir + pair + '/'
    filenames = list(filter(lambda x: (x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')), os.listdir(path)))

    if not os.path.exists(basedir + pair + '/'):
        os.makedirs(basedir + pair + '/')

    gen = pair[2:].split('_D_')[0]
    deletion = pair[2:].split('_D_')[1]

    for filename in filenames:
        deletion_filenames[deletion] = deletion_filenames.get(deletion, []) + [path + filename]


tmp = torch.FloatTensor(35, 20, 3, 48, 128)

j = 0
for key, value in deletion_filenames.items():
    indices = np.random.permutation(len(value))
    for i in range(20):
        img = imread(value[indices[i]])

        img = img_as_float(img)
        img = np.rollaxis(img, 2, 0)

        tmp[j,i,:,:,:] = torch.from_numpy(img)

    j+=1

# print(tmp.size())
# print(tmp.view(-1, 3, 48, 128).size())

save_image(tmp.view(-1, 3, 48, 128), 'deletions.png', nrow=20)



    # gens_counter[gen] = gens_counter.get(gen, 0) + len(filenames)
    # deletions_counter[deletion] = deletions_counter.get(deletion, 0) + len(filenames)

    # cnt[prt2id[gen], del2id[deletion]] = len(filenames)

    # for filename in filenames:
    #     # s = time()
    #     img = imread(path + filename)
    #     img = img_as_float(img)
    #     # print(time()-s)
    #     # s = time()

    #     img = img[16:-16,128:-128,:]
    #     img = resize(img, (48, 128))

    #     imsave(basedirt + pair + '/' + filename, img)
        


# import seaborn as sns
# fig, ax = plt.subplots(figsize=(40,40)) 
# sns.heatmap(cnt[:,1:], annot=True,  linewidths=.5, ax=ax)
# # fig.add_subplot(ax)
# fig.savefig('class_freq.png')

# print(np.sum(cnt==0))


# from matplotlib import pyplot as plt

# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(30, 15    ))

# print(len(gens_counter.keys()))
# print(gens_counter.keys())

# print(len(deletions_counter.keys()))
# print(deletions_counter.keys())

# print(list(set(gens_counter.keys()) - set(deletions_counter.keys())))
# print(list(set(deletions_counter.keys())-set(gens_counter.keys())))

# ax[0].bar(gens_counter.keys(), gens_counter.values())
# ax[1].bar(deletions_counter.keys(), deletions_counter.values())

# fig.savefig('classes.png')
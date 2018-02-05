import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
from numpy.random import multivariate_normal, choice, normal, randint

import torchvision.datasets as dset
import torchvision.transforms as transforms

import os

from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize

class GaussianMixtureDataset(Dataset):
    """Points from multiple gaussians"""

    def __init__(self, mean_list, component_size_list):
        """
	Generate points from multiple gaussians.
        Args:
            mean_list: list of mean vectors.
            component_size: number of points in one component.
        """

        assert len(mean_list) == len(component_size_list)
        self.mean_list = mean_list
        self.component_size_list = component_size_list
        d = len(mean_list[0])
        self.data = np.zeros((0, d))
        self.n_components = len(mean_list)

        for i in range(self.n_components):
            self.data = np.concatenate([self.data, multivariate_normal(mean=mean_list[i], cov=np.eye(d), size=component_size_list[i])], axis=0)

        self.data = np.asarray(self.data, dtype=np.float32)           

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return self.data[idx,:]
        # print(type(self.mean_list))
        return self.mean_list[randint(0, len(self.mean_list))] + normal(size=len(self.mean_list[0]))


class ConditionalGaussianMixtureDataset(Dataset):
    """Points from multiple gaussians"""

    def __init__(self, mean_list, component_size_list, component_class_list, n_classes):
        """
    Generate points from multiple gaussians.
        Args:
            mean_list: list of mean vectors.
            component_size: number of points in one component.
        """
        assert len(mean_list) == len(component_size_list) == len(component_class_list)
        self.mean_list = mean_list
        self.component_size_list = component_size_list
        self.component_class_list = component_class_list
        self.n_classes = n_classes
        self.d = len(mean_list[0])
        self.data = np.zeros((0, self.d + self.n_classes))

        self.n_components = len(mean_list)

        for i in range(self.n_components):
            onehot = np.zeros((component_size_list[i], self.n_classes))
            onehot[:,component_class_list[i]] = 1
            batch = np.concatenate([multivariate_normal(mean=mean_list[i], cov=np.eye(self.d), size=component_size_list[i]), onehot], axis=1)
            self.data = np.concatenate([self.data, batch], axis=0)

        self.data = np.asarray(self.data, dtype=np.float32)           

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]


class MNISTDataset(Dataset):
    """Points from multiple gaussians"""

    def __init__(self, selected=None, train=True):
        self.data = dset.MNIST(root = './data/',
                         transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                          download = True, train=train)

        if selected is not None:
            if train:
                labels = self.data.train_labels.numpy()
            else:
                labels = self.data.test_labels.numpy()
            self.index = np.arange(len(self.data))[np.where(labels == selected)[0]]
            # self.data = torch.masked_select(self.data.train_data, (self.data.train_labels == selected).view(-1, 1, 1)).view(-1, 1, 32, 32)
        else:
            self.index = np.arange(len(self.data))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.data[self.index[idx]][0]
        
class labeledMNISTDataset(Dataset):
    """Points from multiple gaussians"""

    def __init__(self):
        self.data = dset.MNIST(root = './data/',
                         transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                          download = True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyDataLoader():
    '''multiple epochs added'''
    def __init__(self):
        self.i_epoch = 0
        self.last_images = None

    def return_iterator(self, dataloader, is_cuda=False, num_passes=None, conditional=False, pictures=False, n_classes=None):
        self.i_epoch = 0
        
        while num_passes is None or self.i_epoch < num_passes:
            for batch in dataloader:
                if not conditional:
                    if is_cuda:
                        batch = batch.cuda()
                    
                    batch = Variable(batch).float()

                if conditional:
                    if pictures:
                        data = batch[0]
                        labels = batch[1]

                        if is_cuda:
                            data = data.cuda()
                            labels = labels.cuda()
                    
                        data = Variable(data).float()
                        labels = Variable(labels)

                        # print(data.size())
                        
                        batch = data, labels
                        
                    else:
                        if is_cuda:
                            batch = batch.cuda()
                    
                        batch = Variable(batch).float()
                        
                        data = batch[:,:-n_classes]
                        onehot = batch[:,-n_classes:]
                        _, label = torch.max(onehot, dim=1)
                        batch = data, label
                
                yield batch
            self.i_epoch += 1



class LINDataset(Dataset):
    """Points from multiple gaussians"""

    def __init__(self, protein='Arp3', basedir='/home/ubuntu/LIN/LIN_Normalized_WT_size-48-80_train/', transform=None):
        self.path = basedir + protein + '/'
        self.filenames = list(filter(lambda x: (x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')), os.listdir(self.path)))
        self.transform = transform

        self.images = []

        for filename in self.filenames:
            img = imread(self.path + filename)
            # img = resize(img, (24, 40))
            img = img_as_float(img)
            img = np.rollaxis(img[:,:,:2], 2, 0) #.reshape((2, 24, 40))
            img = np.asarray(img, dtype=np.float32)
            img = torch.from_numpy(img)

            if self.transform:
                img = self.transform(img)

            self.images.append(img)

        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # img = imread(self.path + self.filenames[idx])
        # img = resize(img, (24, 40))
        # img = img_as_float(img)
        # img = img[:,:,:2].reshape((2, 24, 40))
        # img = torch.from_numpy(img)

        # if self.transform:
        #     img = self.transform(img)

        return self.images[idx]
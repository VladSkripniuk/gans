import torch
import torch.nn as nn


class toynet_D(nn.Module):
    '''Defines a network to discriminate fake and real points in R^n'''
    def __init__(self, dim_list):
        super(toynet_D, self).__init__()

        main = nn.Sequential()
        # stack fully connected layers
        for i in range(len(dim_list)-1):        
            main.add_module('FC{0}'.format(i), nn.Linear(dim_list[i], dim_list[i+1]))
            if i != len(dim_list)-2:
                # main.add_module('BN{0}'.format(i), nn.BatchNorm1d(dim_list[i+1]))
                main.add_module('LeakyReLU{0}'.format(i), nn.ReLU())#nn.LeakyReLU(0.2, inplace=True))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


class toynet_G(nn.Module):
    def __init__(self, dim_list):
        super(toynet_G, self).__init__()

        main = nn.Sequential()
        # input is Z, going into a fully connected network
        for i in range(len(dim_list)-1):        
            main.add_module('FC{0}'.format(i), nn.Linear(dim_list[i], dim_list[i+1]))
            if i != len(dim_list)-2:
                # main.add_module('BN{0}'.format(i), nn.BatchNorm1d(dim_list[i+1]))
                main.add_module('LeakyReLU{0}'.format(i), nn.ReLU())#nn.LeakyReLU(0.2, inplace=True))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

import load_data as ld
import dataloaders as dl

import pandas as pd
import numpy as np

import torch
from torch.utils import data as td
import torch.nn as nn
import torch.nn.functional as F

import sys 
import os

from sklearn.model_selection import train_test_split

# number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS = 100

# amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM = 6

# size of batch
global BATCH_SIZE
BATCH_SIZE = 16

# dimension of voxel with conformer
global VOXEL_DIM
VOXEL_DIM = 70

# amount of target values
global TARGET_NUM
TARGET_NUM = 12

#dataset folder
global DATASET_PATH
DATASET_PATH="C:\\Users\\Alice\\Documents\\skoltech\\isp\\research\\Final_version\\database"

# create train and validation functions

def train(model, optimizer, train_generator, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_generator):

        # set gradients to zero
        optimizer.zero_grad()

        # calculate output vector
        output = model(data)

        # create mask to get rid of Nan's in target
        mask = (target == target)
        output_masked = torch.masked_select(output, mask).type_as(output)
        target_masked = torch.masked_select(target, mask).type_as(output)

        # multi-label (not multi-class!) classification=>binary cross entropy loss
        loss = F.binary_cross_entropy(output_masked, target_masked)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item()))


def test(model, test_generator):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for data, target in test_generator:
            output = model(data)
            mask = (target == target)
            output_masked = torch.masked_select(output, mask).type_as(output)
            target_masked = torch.masked_select(target, mask).type_as(output)
            test_loss += F.binary_cross_entropy(output_masked, target_masked)
            pred = output_masked.ge(0.5).type_as(output)
            correct += pred.eq(target_masked).sum().item()
            total += output_masked.shape[0]
        test_loss /= len(test_generator.dataset)
        test_loss *= BATCH_SIZE
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, total,
                      100. * correct / total))


# create neural net
class Net(nn.Module):
    def __init__(self, dim=70, num_elems=6, num_targets=12, batch_size=BATCH_SIZE):
        super(Net, self).__init__()

        # initialize dimensions
        self.dim = dim
        self.num_elems = num_elems
        self.num_targets = num_targets
        self.batch_size = batch_size

        # create layers
        self.conv1 = nn.Conv3d(num_elems, 32, kernel_size=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(2048, num_targets)

        # initialize dense layer's weights
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.convolution = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.pool3,
            self.conv4,
            self.pool4
        )

        def weights_init(m):
            if type(m) == nn.Conv3d:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # initialize convolutional layers' weights
        self.convolution.apply(weights_init)

    def forward(self, x):
        x_conv = self.convolution(x)
        x_vect = x_conv.view(self.batch_size, -1)
        y = self.fc1(x_vect)
        #         multi-label (not multi-class!) classification => sigmoid non-linearity
        return torch.sigmoid(y)


def main():
    print('Start loading dataset...')
    # get dataset without duplicates from csv
    
    data = pd.read_csv(os.path.join(DATASET_PATH, 'tox21_10k_data_all_no_salts.csv'))

    # create elements dictionary
    elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM)

    # read databases to dictionary
    conf_calc = ld.reading_sql_database(os.path.join(DATASET_PATH, 'tox21_conformers.db'))

    # create indexing and label_dict for iteration
    indexing, label_dict = ld.indexing_label_dict(data, conf_calc)
    print('Dataset has been loaded')
    # create train and validation sets' indexes
    print('Neural network initialization...')
    train_indexes, test_indexes, _, _ = train_test_split(np.arange(0, len(conf_calc.keys())),
                                                         np.arange(0, len(conf_calc.keys())), test_size=0.2,
                                                         random_state=42)

    # make dataloader for Gauss transformation
#    train_set_gauss = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=3)
#    train_generator_gauss = td.DataLoader(train_set_gauss, batch_size=BATCH_SIZE, shuffle=True)

#    test_set_gauss = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=3)
#    test_generator_gauss = td.DataLoader(test_set_gauss, batch_size=BATCH_SIZE, shuffle=True)

    # make dataloader for Waves transformation
    train_set_waves = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=6)
    train_generator_waves = td.DataLoader(train_set_waves, batch_size=BATCH_SIZE, shuffle=True)

    test_set_waves = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=6)
    test_generator_waves = td.DataLoader(test_set_waves, batch_size=BATCH_SIZE, shuffle=True)

    # Construct our model by instantiating the class defined above
    model = Net(dim=VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, batch_size=BATCH_SIZE)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('Neural network has been initialized')
    # train procedure
    for epoch in range(1, 20 + 1):
        train(model, optimizer, train_generator_waves, epoch)
        test(model, test_generator_waves)
        torch.save(model.state_dict(), os.path.join(DATASET_PATH, 'model'+str(epoch)))

if __name__ == '__main__':
    sys.exit(main()) 
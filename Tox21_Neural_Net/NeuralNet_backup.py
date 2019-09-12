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
from sklearn.metrics import roc_auc_score

# number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS = 100

# amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM = 6

# size of batch
global BATCH_SIZE
BATCH_SIZE = 32

# dimension of voxel with conformer
global VOXEL_DIM
VOXEL_DIM = 70

# amount of target values
global TARGET_NUM
TARGET_NUM = 12

#dataset folder
global DATASET_PATH
DATASET_PATH="./Documents/Tox21_Neural_Net/database"

#logs path
global LOG_PATH
LOG_PATH="./Documents/Tox21_Neural_Net/logs"

#models path
global MODEL_PATH
MODEL_PATH="./Documents/Tox21_Neural_Net/models"

#number of epochs
global EPOCHS_NUM
EPOCHS_NUM=1

# create train and validation functions

def train(model, optimizer, train_generator, epoch, device,f_auc=None,f_loss=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_generator):
        data = data.to(device)
        target = target.to(device)
        # set gradients to zero
        optimizer.zero_grad()

        # calculate output vector
        output = model(data)

        # create mask to get rid of Nan's in target
        mask = (target == target)
        output_masked = torch.masked_select(output, mask).type_as(output)
        target_masked = torch.masked_select(target, mask).type_as(output)

        pred = output_masked.ge(0.5).type_as(output)
        try:
            auc=roc_auc_score(target_masked.cpu().detach(),pred.cpu().detach())
            if f_auc is not None:
                f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(auc)+'\n')
        except ValueError:
            pass
        # multi-label (not multi-class!) classification=>binary cross entropy loss
        loss = F.binary_cross_entropy(output_masked, target_masked)
        if f_loss is not None:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().detach().numpy().item())+'\n')
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item()))


def test(model, test_generator,epoch,device,f_loss=None,f_auc=None):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        errors=0
        for batch_idx, (data, target) in enumerate(test_generator):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            mask = (target == target)
            output_masked = torch.masked_select(output, mask).type_as(output)
            target_masked = torch.masked_select(target, mask).type_as(output)
            loss=F.binary_cross_entropy(output_masked, target_masked)
            test_loss += loss
            pred = output_masked.ge(0.5).type_as(output)
            
            try:
                auc=roc_auc_score(target_masked.cpu(),pred.cpu())
                if f_auc is not None:
                    f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(auc)+'\n')
                correct += auc
            except ValueError:
                errors+=1
            if f_loss is not None:
                f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().detach().numpy().item())+'\n')
#            total += output_masked.shape[0]
        test_loss /= len(test_generator.dataset)
        test_loss *= BATCH_SIZE
        batch_num=len(test_generator.dataset)/BATCH_SIZE-errors
        print('\nTest set: Average loss: {:.4f}, last AUC: {:.2f}, average AUC: {:.2f}\n'
              .format(test_loss, auc, correct/batch_num))


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
        x_vect = x_conv.view(x.shape[0], -1)
        y = self.fc1(x_vect)
        #         multi-label (not multi-class!) classification => sigmoid non-linearity
        return torch.sigmoid(y)


def main():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print('Start loading dataset...')
    # get dataset without duplicates from csv
    
    data = pd.read_csv(os.path.join(DATASET_PATH, 'tox21_10k_data_all_no_salts.csv'))

    # create elements dictionary
    elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM)

    # read databases to dictionary
    conf_calc = ld.reading_sql_database(os.path.join(DATASET_PATH, 'tox21_conformers.db'))
    keys=list(conf_calc.keys())
    print ('Initial dataset size = ', len(keys))
    for key in keys:
        conformers=list(conf_calc[key].keys())
        for conformer in conformers:
            try:
                conf_calc[key][conformer]['energy']
            except:
                del conf_calc[key][conformer]
        if conf_calc[key]=={}:
            del conf_calc[key]
    print ('Post-processed dataset size = ', len(list(conf_calc.keys())))
    # create indexing and label_dict for iteration
    indexing, label_dict = ld.indexing_label_dict(data, conf_calc)
    print('Dataset has been loaded')
    # create train and validation sets' indexes
    print('Neural network initialization...')
    train_indexes, test_indexes, _, _ = train_test_split(np.arange(0, len(conf_calc.keys())),
                                                         np.arange(0, len(conf_calc.keys())), test_size=0.2,
                                                         random_state=42)

    # make dataloader for Gauss transformation
    train_set_gauss = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=3)
    train_generator_gauss = td.DataLoader(train_set_gauss, batch_size=BATCH_SIZE, shuffle=True)

    test_set_gauss = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=3)
    test_generator_gauss = td.DataLoader(test_set_gauss, batch_size=BATCH_SIZE, shuffle=True)

    # make dataloader for Waves transformation
#    train_set_waves = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=6)
#    train_generator_waves = td.DataLoader(train_set_waves, batch_size=BATCH_SIZE, shuffle=True)
#
#    test_set_waves = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=6)
#    test_generator_waves = td.DataLoader(test_set_waves, batch_size=BATCH_SIZE, shuffle=True)

    # Construct our model by instantiating the class defined above
    model = Net(dim=VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, batch_size=BATCH_SIZE)
    print(model)
    model=model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    print('Neural network has been initialized')
    
    f_train_loss=open(LOG_PATH+'/log_train_loss.txt','w')
    f_test_loss=open(LOG_PATH+'/log_test_loss.txt','w')
    f_train_auc=open(LOG_PATH+'/log_train_auc.txt','w')
    f_test_auc=open(LOG_PATH+'/log_test_auc.txt','w')
    # train procedure
    for epoch in range(1, EPOCHS_NUM + 1):
        try:
#            train(model, optimizer, train_generator_waves, epoch,device,f_auc=f_train_auc,f_loss=f_train_loss)
#            test(model, test_generator_waves,epoch, device,f_auc=f_test_auc,f_loss=f_test_loss)
            train(model, optimizer, train_generator_gauss, epoch,device,f_auc=f_train_auc,f_loss=f_train_loss)
            test(model, test_generator_gauss,epoch, device,f_auc=f_test_auc,f_loss=f_test_loss)
        except KeyError:
            print('Key Error problem')
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model'+str(epoch)))
    f_train_loss.close()
    f_test_loss.close()
    f_test_auc.close()
    f_train_auc.close()
if __name__ == '__main__':
    sys.exit(main()) 

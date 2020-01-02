import load_data_multitox as ld
import dataloaders_sigma as dl
from Model_train_test_regression import Net, EarlyStopping, train, test

import pandas as pd
import numpy as np

import torch
from torch.utils import data as td
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchsummary import summary

import sys 
import os
import glob

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

import time
from sklearn.preprocessing import StandardScaler

import json


# number of conformers created for every molecule
NUM_CONFS = 100

# amount of chemical elements taking into account
AMOUNT_OF_ELEM = 9

# amount of target values
TARGET_NUM = 29

#dataset folder
# DATASET_PATH="~/Tox21-MultiTox/MultiTox"
DATASET_PATH="./"

#logs path

LOG_PATH=os.path.join(DATASET_PATH,"logs_sigma")


#models path
MODEL_PATH=os.path.join(DATASET_PATH,"models_sigma")


#loss penalty
#global PENALTY
#PENALTY = torch.FloatTensor([0.1,0.2,0.4,0.4,0.4,0.2,0.2,0.6,0.2,0.3,0.6,0.2])

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", dest="EPOCHS_NUM",
                    help="number of train epochs",default = 100,type=int)
parser.add_argument("-p", "--patience",
                    dest="PATIENCE", default=25,
                    help="number of epochs to wait before early stopping",type=int)
parser.add_argument("-s", "--sigma",
                    dest="SIGMA", default=12,
                    help="sigma parameter",type=int)
parser.add_argument("-b", "--batch_size",
                    dest="BATCH_SIZE", default=32,
                    help="size of train and test batches",type=int)
parser.add_argument("-t", "--transformation",
                    dest="TRANSF", default='g',choices =['g', 'w'],
                    help="type of augmentstion - g (gauss) or w (waves)")
parser.add_argument("-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment")
parser.add_argument("-v", "--voxel_dim",
                    dest="VOXEL_DIM", default=50,
                    help="size of produced voxel cube")
parser.add_argument("-r", "--learn_rate",
                    dest="LEARN_RATE", default=1e-5,
                    help="leraning rate for optimizer",type=float)

global args
args = parser.parse_args()

#name or flags - Either a name or a list of option strings, e.g. foo or -f, --foo.
#
#action - The basic type of action to be taken when this argument is encountered at the command line.
#
#nargs - The number of command-line arguments that should be consumed.
#
#const - A constant value required by some action and nargs selections.
#
#default - The value produced if the argument is absent from the command line.
#
#type - The type to which the command-line argument should be converted.
#
#choices - A container of the allowable values for the argument.
#
#required - Whether or not the command-line option may be omitted (optionals only).
#
#help - A brief description of what the argument does.
#
#metavar - A name for the argument in usage messages.
#
#dest - The name of the attribute to be added to the object returned by parse_args().


def main():
    
    global MODEL_PATH
    global DATASET_PATH
    global TARGET_NUM
    global VOXEL_DIM
    global AMOUNT_OF_ELEM
    global LOG_PATH
    global NUM_CONFS
    print(vars(args))
    path = os.path.join(LOG_PATH,'exp_'+args.NUM_EXP)
    original_umask = os.umask(0)
    try:
        original_umask = os.umask(0)
        os.mkdir(path, mode = 0o777)
    except FileExistsError:
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path,f))

    finally:
        os.umask(original_umask)
        LOG_PATH = path


    path = os.path.join(MODEL_PATH,'exp_'+args.NUM_EXP)
    print(path)
    try:
        original_umask = os.umask(0)
        os.mkdir(path, 0o777)
        print('Dir has been made')
    except FileExistsError:
        print('Dir already exists')
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path,f))
    finally:
        print('finita')
        os.umask(original_umask)
        MODEL_PATH = path
    start_time=time.time()
    writer=SummaryWriter(LOG_PATH)
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
    data = pd.read_csv(os.path.join(DATASET_PATH,'database', 'MultiTox.csv'))
    props = list(data)[1:]
    scaler = StandardScaler()
    data[props]=scaler.fit_transform(data[props])

    # create elements dictionary
    elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM+1)

    # read databases to dictionary
    conf_calc = ld.reading_sql_database(database_dir='./dat/')
    keys=list(conf_calc.keys())
    print ('Initial dataset size = ', len(keys))
    elems = []
    for key in keys:
        conformers=list(conf_calc[key].keys())
        for conformer in conformers:
            try:
                energy = conf_calc[key][conformer]['energy']
                elems = list(set(elems+list(conf_calc[key][conformer]['coordinates'].keys())))
            except:
                del conf_calc[key][conformer]
            
        if conf_calc[key]=={}:
            del conf_calc[key]

    print ('Post-processed dataset size = ', len(list(conf_calc.keys())))
    # create indexing and label_dict for iteration
    indexing, label_dict = ld.indexing_label_dict(data, conf_calc)
    print('Dataset has been loaded, ', int(time.time()-start_time),' s')
    start_time=time.time()
    # create train and validation sets' indexes
    print('Neural network initialization...')
    
    train_indexes, test_indexes, _, _ = train_test_split(np.arange(0, len(conf_calc.keys())),
                                                         np.arange(0, len(conf_calc.keys())), test_size=0.2,
                                                         random_state=115)
    train_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, train_indexes, dim = args.VOXEL_DIM)
    train_generator = td.DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)

    test_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, test_indexes, dim = args.VOXEL_DIM)
    test_generator = td.DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=True)
    
    model = Net(dim=args.VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, elements=elements, transformation=args.TRANSF,device=device,sigma_0 = args.SIGMA,sigma_trainable = True)


    # Construct our model by instantiating the class defined above

    model=model.to(device)

    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARN_RATE)

    print('Neural network has been initialized, ', int(time.time()-start_time),' s')
    
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_parameters.json'),'w') as f:
        json.dump(vars(args), f)
    
    f_train_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss.txt'),'w')
    f_train_loss_ch=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss_channels.txt'),'w')
    f_test_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_test_loss.txt'),'w')
    
    early_stopping = EarlyStopping(patience=args.PATIENCE, verbose=True,model_path=MODEL_PATH)

    start_time=time.time()
    # train procedure
    for epoch in range(1, args.EPOCHS_NUM + 1):
        try:
            train(model, optimizer, train_generator, epoch,device,writer=writer,f_loss=f_train_loss,f_loss_ch=f_train_loss_ch, elements=elements,batch_size = args.BATCH_SIZE)
            test_loss = test(model, test_generator,epoch, device,writer=writer,f_loss=f_test_loss, elements=elements,batch_size = args.BATCH_SIZE)
            early_stopping(test_loss, model)

            if early_stopping.early_stop:
                print(epoch,"Early stopping")
                break
            if epoch%100==0:
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, args.NUM_EXP+'_model_'+str(epoch)))
        except KeyError:
            print(epoch,'Key Error problem')
        
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, args.NUM_EXP+'_model'+str(epoch)+'_fin'))
    f_train_loss.close()
    f_test_loss.close()
    writer.close()
    print('Training has finished in ',round((time.time()-start_time)/60,3),' min.')
if __name__ == '__main__':
    sys.exit(main()) 

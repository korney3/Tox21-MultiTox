#script with classification with new model and database of nine elements without H

import load_data as ld
import load_data_multitox as ld
import dataloaders as dl

import pandas as pd
import numpy as np

import torch
from torch.utils import data as td
import torch.nn as nn
import torch.nn.functional as F

import sys 
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


from tensorboardX import SummaryWriter

from Model_train_test_regression import EarlyStopping

import json

# number of conformers created for every molecule
NUM_CONFS = 100

# amount of chemical elements taking into account
AMOUNT_OF_ELEM = 9

# size of batch
BATCH_SIZE = 32

# dimension of voxel with conformer
# VOXEL_DIM = 70

# amount of target values
# TARGET_NUM = 12

#dataset folder
DATASET_PATH="/gpfs/gpfs0/a.alenicheva/Tox21/elements_9"

#logs path
LOG_PATH="./Documents/Tox21_Neural_Net"

#models path
MODEL_PATH="./Documents/Tox21_Neural_Net"

#number of epochs
# EPOCHS_NUM=100

#loss penalty
PENALTY = torch.FloatTensor([0.1,0.2,0.4,0.4,0.4,0.2,0.2,0.6,0.2,0.3,0.6,0.2])

#patience for early stopping
# PATIENCE = 25

# #sigma parameter for preprocessing
# SIGMA = 3

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", dest="EPOCHS_NUM",
                    help="number of train epochs. Default is 100.",default = 100,type=int)
parser.add_argument("-p", "--patience",
                    dest="PATIENCE", default=25,
                    help="number of epochs to wait before early stopping. Default is 25.",type=int)
parser.add_argument("-s", "--sigma",
                    dest="SIGMA", default=1.2,
                    help="sigma parameter",type=float)
parser.add_argument("-b", "--batch_size",
                    dest="BATCH_SIZE", default=32,
                    help="size of train and test batches. Default is 32.",type=int)
parser.add_argument("-t", "--transformation",
                    dest="TRANSF", default='g',choices =['g', 'w'],
                    help="type of augmentstion - 'g' (gauss) or 'w' (waves). Default is 'g'.")
parser.add_argument("-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment")
parser.add_argument("-v", "--voxel_dim",
                    dest="VOXEL_DIM", default=50,
                    help="size of produced voxel cube. Default is 50.", type=int)
parser.add_argument("-r", "--learn_rate",
                    dest="LEARN_RATE", default=1e-5,
                    help="learning rate for optimizer. Default is 1e-5.",type=float)
parser.add_argument("-a", "--sigma_train",
                    dest="SIGMA_TRAIN", default=False,
                    help="regime of training sigma (True) or not (False). Default is False.",type=bool)
parser.add_argument("-m", "--mode",
                    dest="MODE", default='r', choices =['r', 'c'],
                    help="choosing classification ('c' option) or regression ('r' option) tasks. Default is 'r'",type=str)
parser.add_argument("-c", "--continue",
                    dest="CONTINUE", default=1,
                    help="number of epoch to continue training process from. Default is 1",type=int)
parser.add_argument("-l", "--trlearning",
                    dest="TRLEARNING", default=0, choices =[0, 1],
                    help="regime of transferlearning for classification task, 0 - transfer learning off, 1 - on. Default is 0",type=int)

args = parser.parse_args()
        

def main():
    global NUM_CONFS
    global AMOUNT_OF_ELEM
    global BATCH_SIZE
    global TARGET_NUM
    global DATASET_PATH
    global LOG_PATH
    global MODEL_PATH
    global EPOCHS_NUM
    global PENALTY
    global PATIENCE
    global SIGMA
    
    global args
    
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_parameters.json'),'w') as f:
        json.dump(vars(args), f)

    if args.MODE == "r":
        TARGET_NUM = 29
        from Model_train_test_regression import train_regression as train, test_regression as test
    elif args.MODE == "c":
        TARGET_NUM = 12
        from Model_train_test_regression import train_classification as train, test_classification as test
    
    if args.SIGMA_TRAIN:
        from Model_train_test_regression import Net_with_transform as Net
    else:
        from Model_train_test_regression import Net_without_transform as Net
        
    # create log directories
    path = os.path.join(LOG_PATH,'exp_'+args.NUM_EXP)
    os.makedirs(path, exist_ok=True)
    LOG_PATH = path
    
    
    path = os.path.join(MODEL_PATH,'exp_'+args.NUM_EXP)
    os.makedirs(path, exist_ok=True)
    MODEL_PATH = path
    
    # setting log files
    # writer - tensorboard writer
    # f_log - file with progress messages
    writer=SummaryWriter(LOG_PATH)
    f_log=open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'w')
    f_log.close()
    
    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Using device:'+str(device)+'\n')
    print()
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write(torch.cuda.get_device_name(0)+'\n'+'Memory Usage:'+'\n'+'Allocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+ 'GB'+'\n'+'Cached:   '+str(round(torch.cuda.memory_cached(0)/1024**3,1))+'GB'+'\n')
            
    # loading dataset from .csv file
    print('Start loading dataset...')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Start loading dataset...'+'\n')
    start_time=time.time()
    
    if args.MODE == 'r':
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Regression mode'+'\n')
        data = pd.read_csv(os.path.join(MULTITOX_STORAGE,'database/data', 'MultiTox.csv'))
        props = list(data)
        props.remove("SMILES")
        print(props)
        scaler = MinMaxScaler()
        data[props]=scaler.fit_transform(data[props])
    elif args.MODE == 'c':
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Classification mode'+'\n')
        data = pd.read_csv(os.path.join('../Tox21_Neural_Net/database', 'tox21_10k_data_all_no_salts.csv'))
    
    

    # create elements dictionary
#     elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM)
    elements = {'I': 0,
 'P': 1,
 'Br': 2,
 'F': 3,
 'S': 4,
 'Cl': 5,
 'N': 6,
 'O': 7,
 'C': 8}

    # read databases from .db files to dictionary
    if args.MODE == "r":
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                f_log.write('Regression mode load database'+'\n')
        conf_calc = ld.reading_sql_database(database_dir=os.path.join(DATASET_PATH,"MultiTox"))
    elif args.MODE == "c":
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Classification mode load database'+'\n')
        conf_calc = ld.reading_sql_database(os.path.join(DATASET_PATH))
    # remove broken molecules
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
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Post-processed dataset size = '+str(len(list(conf_calc.keys())))+'\n')
    # create indexing and label_dict for iteration
    indexing, label_dict = ld.indexing_label_dict(data, conf_calc)
    print('Dataset has been loaded, ', int(time.time()-start_time),' s')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Dataset has been loaded, '+str(int(time.time()-start_time))+' s'+'\n')
    
    # set dataloaders
    # create train and validation sets' indexes
    print('Neural network initialization...')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Neural network initialization...'+'\n')
    start_time=time.time()
    train_indexes, test_indexes, _, _ = train_test_split(np.arange(0, len(conf_calc.keys())),
                                                         np.arange(0, len(conf_calc.keys())), test_size=0.2,
                                                         random_state=42)
    if args.SIGMA_TRAIN:
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Sigma train dataloader'+'\n')
        pass
    else:
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Not sigma train dataloader'+'\n')
        if args.TRANSF == 'g':
            with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                f_log.write('Gauss mode'+'\n')
            train_set = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=args.SIGMA, dim = args.VOXEL_DIM)
            train_generator = td.DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)

            test_set = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=args.SIGMA, dim = args.VOXEL_DIM)
            test_generator = td.DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=False)
        elif args.TRANSF == 'w':
            with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                f_log.write('Waves mode'+'\n')
            train_set = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=args.SIGMA, dim = args.VOXEL_DIM)
            train_generator = td.DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)

            test_set = dl.Waves_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=args.SIGMA, dim = args.VOXEL_DIM)
            test_generator = td.DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=False)
                                        
    # set model
    if args.SIGMA_TRAIN:
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Sigma train model'+'\n')               
        pass
    else:
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Not sigma train model'+'\n')                                
        if args.TRLEARNING:
            with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                f_log.write('transfer learning model'+'\n')
            pass
        else:
            with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                f_log.write('Not transfer model'+'\n')
            model = Net(dim=args.VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, device=device, mode=args.MODE)
                                                                          
    if args.CONTINUE>1:
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Continue mode'+'\n')
        pass
    else:
        pass
                                        
    print(model)
                                        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print ('Run in parallel!')
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Run in parallel!'+'\n')
    model=model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    print('Neural network has been initialized')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Neural network has been initialized, '+str(int(time.time()-start_time))+' s'+'\n')
    
    # set log txt files
    f_train_loss=open(LOG_PATH+'/log_train_loss.txt','w')
    f_train_loss_ch=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss_channels.txt'),'w')
    
    f_test_loss=open(LOG_PATH+'/log_test_loss.txt','w')

    f_train_auc=open(LOG_PATH+'/log_train_auc.txt','w')
    f_test_auc=open(LOG_PATH+'/log_test_auc.txt','w')
    
    early_stopping = EarlyStopping(patience=args.PATIENCE, verbose=True)

    
    # train procedure
    start_time=time.time()
    for epoch in range(1, args.EPOCHS_NUM + 1):
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Epoch, '+str(epoch)+'\n')
        try:
#            train(model, optimizer, train_generator_waves, epoch,device,f_auc=f_train_auc,f_loss=f_train_loss)
#            test(model, test_generator_waves,epoch, device,f_auc=f_test_auc,f_loss=f_test_loss)
            if args.MODE == 'r':
                with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                    f_log.write('Regression'+'\n')
                train (model, optimizer, train_generator, epoch,device,writer=writer,f_loss=f_train_loss,f_loss_ch=f_train_loss_ch, elements=elements,batch_size = args.BATCH_SIZE,MODEL_PATH=MODEL_PATH, sigma_train = args.SIGMA_TRAIN)
                test_loss = test(model, test_generator,epoch, device,writer=writer,f_loss=f_test_loss, elements=elements,batch_size = args.BATCH_SIZE)
            elif args.MODE=='c':
                with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
                    f_log.write('Classification'+'\n')
                train(model, optimizer, train_generator, epoch,device,batch_size=args.BATCH_SIZE, writer=writer,f_loss=f_train_loss, f_loss_ch = f_train_auc, elements = elements, MODEL_PATH = MODEL_PATH, PENALTY = PENALTY)
                test_loss = test(model, test_generator,epoch, device,batch_size=args.BATCH_SIZE,writer=writer,f_loss=f_test_loss, elements = elements, PENALTY = PENALTY)
            
        except KeyError:
            print('Key Error problem')
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch%10==0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model'+str(epoch)))
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model'+str(epoch)+'_fin'))
    f_train_loss.close()
    f_test_loss.close()
    f_test_auc.close()
    f_train_auc.close()
    writer.close()
if __name__ == '__main__':
    sys.exit(main()) 

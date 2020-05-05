import load_data_multitox as ld
import dataloaders_sigma as dl
from Model_train_test_regression import Net, EarlyStopping, train_regression, train_classification, test_regression, test_classification

import pandas as pd
import numpy as np

import torch
from torch.utils import data as td
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


import sys 
import os
import glob

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

import time
from sklearn.preprocessing import MinMaxScaler#StandardScaler

import json


# number of conformers created for every molecule
NUM_CONFS = 100

# amount of chemical elements taking into account
AMOUNT_OF_ELEM = 9

# amount of target values
TARGET_NUM = 29

#loss penalty for classification
PENALTY = torch.FloatTensor([0.1,0.2,0.4,0.4,0.4,0.2,0.2,0.6,0.2,0.3,0.6,0.2])


#dataset folder
# DATASET_PATH="~/Tox21-MultiTox/MultiTox"
DATASET_PATH = "/gpfs/gpfs0/a.alenicheva"

TOX21_STORAGE = "../Tox21_Neural_Net"
TOX21_STORAGE = "./"

MULTITOX_STORAGE = "./"

EXPERIMENTS_DATA = "./"

#logs path
LOG_PATH=os.path.join(EXPERIMENTS_DATA, "logs_sigma_right")


#models path
MODEL_PATH=os.path.join(EXPERIMENTS_DATA, "models_sigma_right")


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
                    help="size of produced voxel cube. Default is 50.")
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
    global TOX21_STORAGE
    global MULTITOX_STORAGE
    global EXPERIMENTS_DATA
    global PENALTY
    global args
    
    if args.MODE == "r":
        TARGET_NUM = 29
    elif args.MODE == "c":
        TARGET_NUM = 12
    
    print(vars(args))
    path = os.path.join(LOG_PATH,'exp_'+args.NUM_EXP)
    os.makedirs(path, exist_ok=True)
#     original_umask = os.umask(0)
    LOG_PATH = path
#     try:
#         original_umask = os.umask(0)
#         os.mkdir(path, mode = 0o777)
#     except FileExistsError:
#         files = os.listdir(path)
#         for f in files:
#             os.remove(os.path.join(path,f))

#     finally:
#         os.umask(original_umask)
#         LOG_PATH = path


    path = os.path.join(MODEL_PATH,'exp_'+args.NUM_EXP)
    os.makedirs(path, exist_ok=True)
#     original_umask = os.umask(0)
    MODEL_PATH = path
#     try:
#         original_umask = os.umask(0)
#         os.mkdir(path, 0o777)
#         print('Dir has been made')
#     except FileExistsError:
#         print('Dir already exists')
#         files = os.listdir(path)
#         for f in files:
#             os.remove(os.path.join(path,f))
#     finally:
#         print('finita')
#         os.umask(original_umask)
#         MODEL_PATH = path
        
    f_log=open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'w')
    f_log.close()
    start_time=time.time()
    writer=SummaryWriter(LOG_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    print('Start loading dataset...')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Start loading dataset...'+'\n')
    # get dataset without duplicates from csv
    if args.MODE == 'r':
        data = pd.read_csv(os.path.join(MULTITOX_STORAGE,'database/data', 'MultiTox.csv'))
        props = list(data)
        props.remove("SMILES")
        print(props)
        scaler = MinMaxScaler()
        data[props]=scaler.fit_transform(data[props])
    elif args.MODE == 'c':
        data = pd.read_csv(os.path.join(TOX21_STORAGE,'database/data', 'tox21_10k_data_all_no_salts.csv'))
        
        
    

    # create elements dictionary
#     elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM+1)
    elements = {'I': 0,
 'P': 1,
 'Br': 2,
 'F': 3,
 'S': 4,
 'Cl': 5,
 'N': 6,
 'O': 7,
 'C': 8}
    # read databases to dictionary
    if args.MODE == "r":
        conf_calc = ld.reading_sql_database(database_dir=os.path.join(DATASET_PATH,"MultiTox"))
    elif args.MODE == "c":
        conf_calc = ld.reading_sql_database(database_dir=os.path.join(DATASET_PATH,"Tox21", "elements_9"))
#     with open(os.path.join(DATASET_PATH,'many_elems.json'), 'r') as fp:
#         conf_calc = json.load(fp)
    
    keys=list(conf_calc.keys())
    print ('Initial dataset size = ', len(keys))
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Initial dataset size = '+str(len(keys))+'\n')
    new_conf_calc={}
    for smiles in conf_calc.keys():
        for conf_num in conf_calc[smiles]:
            if smiles in new_conf_calc.keys():
                new_conf_calc[smiles][int(conf_num)]=conf_calc[smiles][conf_num]
            else:
                new_conf_calc[smiles]={}
                new_conf_calc[smiles][int(conf_num)]=conf_calc[smiles][conf_num]

    conf_calc=new_conf_calc
    
    elems = []
    for key in keys:
        conformers=list(conf_calc[key].keys())
        for conformer in conformers:
            try:
                energy = conf_calc[key][conformer]['energy']
                elems = list(set(elems+list(conf_calc[key][conformer]['coordinates'].keys())))
            except:
                del conf_calc[key][conformer]
        if set(conf_calc[key].keys())!=set(range(100)):
              del conf_calc[key]
        elif conf_calc[key]=={}:
            del conf_calc[key]

    print ('Post-processed dataset size = ', len(list(conf_calc.keys())))
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Post-processed dataset size = '+str(len(list(conf_calc.keys())))+'\n')
    # create indexing and label_dict for iteration
    indexing, label_dict = ld.indexing_label_dict(data, conf_calc)
    print('Dataset has been loaded, ', int(time.time()-start_time),' s')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Dataset has been loaded, '+str(int(time.time()-start_time))+' s'+'\n')
    
    start_time=time.time()
    # create train and validation sets' indexes
    print('Neural network initialization...')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Neural network initialization...'+'\n')
    train_indexes, test_indexes, _, _ = train_test_split(np.arange(0, len(conf_calc.keys())),
                                                         np.arange(0, len(conf_calc.keys())), test_size=0.2,
                                                         random_state=115)
    train_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, train_indexes, dim = args.VOXEL_DIM)
    train_generator = td.DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)

    test_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, test_indexes, dim = args.VOXEL_DIM)
    test_generator = td.DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=True)
    
    if args.TRLEARNING:
        model = Net(dim=args.VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=29, elements=elements, transformation=args.TRANSF,device=device,sigma_0 = args.SIGMA,sigma_trainable = args.SIGMA_TRAIN, mode = args.MODE)
    else:
        model = Net(dim=args.VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, elements=elements, transformation=args.TRANSF,device=device,sigma_0 = args.SIGMA,sigma_trainable = args.SIGMA_TRAIN, mode = args.MODE)
    
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_parameters.json'),'w') as f:
        json.dump(vars(args), f)
        
    if args.CONTINUE>1:
        model_checkpoints = os.listdir(MODEL_PATH)
        if 'checkpoint_es.pt' in model_checkpoints:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
        elif 'checkpoint.pt' in model_checkpoints:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
    
    if args.TRLEARNING:
        model_checkpoints = os.listdir(MODEL_PATH)
        if 'checkpoint_es.pt' in model_checkpoints:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
        elif 'checkpoint.pt' in model_checkpoints:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
        for param in model.parameters():
            param.requires_grad = False
        model.fc1 = nn.Linear(256, 128)
        model.fc2 = nn.Linear(128,TARGET_NUM)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('total parameters: '+str(total_params)+'\n')
            f_log.write('training parameters: '+str(total_trainable_params)+'\n')
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print ('Run in parallel!')
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Run in parallel!'+'\n')

    # Construct our model by instantiating the class defined above

    model=model.to(device)
    
#     if args.CONTINUE>1:
#         model_checkpoints = os.listdir(MODEL_PATH)
#         if 'checkpoint_es.pt' in model_checkpoints:
#             model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
# #             if torch.cuda.device_count() > 1:
# #                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
# #             else:
# #                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#         elif 'checkpoint.pt' in model_checkpoints:
#             model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
# #             if torch.cuda.device_count() > 1:
# #                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
# #             else:
# #                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
                
#     if args.TRLEARNING:
#         model_checkpoints = os.listdir(MODEL_PATH)
#         if 'checkpoint_es.pt' in model_checkpoints:
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint_es.pt')))
#         elif 'checkpoint.pt' in model_checkpoints:
#             if torch.cuda.device_count() > 1:
#                 model.module.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#             else:
#                 model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
#         for param in model.parameters():
#             param.requires_grad = False
#         model.module.fc1 = nn.Linear(256, 128)
#         model.module.fc2 = nn.Linear(128,TARGET_NUM)
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f'{total_params:,} total parameters.')
#         total_trainable_params = sum(
#             p.numel() for p in model.parameters() if p.requires_grad)
#         print(f'{total_trainable_params:,} training parameters.')
#         with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
#             f_log.write('total parameters: '+str(total_params)+'\n')
#             f_log.write('training parameters: '+str(total_trainable_params)+'\n')



    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARN_RATE)

    print('Neural network has been initialized, ', int(time.time()-start_time),' s')
    with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Neural network has been initialized, '+str(int(time.time()-start_time))+' s'+'\n')

    
    
    f_train_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss.txt'),'w')
    f_train_loss_ch=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss_channels.txt'),'w')
    f_test_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_test_loss.txt'),'w')
    
    early_stopping = EarlyStopping(patience=args.PATIENCE, verbose=True,model_path=MODEL_PATH)

    start_time=time.time()
    # train procedure
    for epoch in range(args.CONTINUE, args.EPOCHS_NUM + args.CONTINUE):
        with open(os.path.join(LOG_PATH,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Epoch , '+str(epoch)+'\n')
        try:
            if args.MODE == 'r':
                train_regression(model, optimizer, train_generator, epoch,device,writer=writer,f_loss=f_train_loss,f_loss_ch=f_train_loss_ch, elements=elements,batch_size = args.BATCH_SIZE,MODEL_PATH=MODEL_PATH)
                test_loss = test_regression(model, test_generator,epoch, device,writer=writer,f_loss=f_test_loss, elements=elements,batch_size = args.BATCH_SIZE)
                early_stopping(test_loss, model)
                
            elif args.MODE == 'c':
                train_classification(model, optimizer, train_generator, epoch,device,writer=writer,f_loss=f_train_loss,f_loss_ch=f_train_loss_ch, elements=elements,batch_size = args.BATCH_SIZE,MODEL_PATH=MODEL_PATH, PENALTY = PENALTY)
                test_loss = test_classification(model, test_generator,epoch, device,writer=writer,f_loss=f_test_loss, elements=elements,batch_size = args.BATCH_SIZE, PENALTY = PENALTY)
                early_stopping(test_loss, model)
    
            if early_stopping.early_stop:
                print(epoch,"Early stopping")
                break
            if epoch%10==0:
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

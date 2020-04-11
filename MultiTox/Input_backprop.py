import load_data_multitox as ld
import dataloaders_sigma as dl
from Model_train_test_regression import Net, EarlyStopping, train, test
from visualization import plot_visualization_input_as_parameter, VolToDx

import pandas as pd
import numpy as np

import torch
from torch.utils import data as td
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

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

#dataset folder
# DATASET_PATH="~/Tox21-MultiTox/MultiTox"
DATASET_PATH="./"

#logs path
LOG_PATH_LOAD=os.path.join(DATASET_PATH,"logs_sigma_right")


#models path
MODEL_PATH_LOAD=os.path.join(DATASET_PATH,"models_sigma_right")


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", dest="EPOCHS_NUM",
                    help="number of train epochs",default = 100000,type=int)
parser.add_argument("-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment")
parser.add_argument("-r", "--learn_rate",
                    dest="LEARN_RATE", default=1e3,
                    help="learning rate for optimizer",type=float)
parser.add_argument("-o", "--optimizer",
                    dest="OPTIMIZER", default='Adam',choices =['Adam', 'SGDL2'],
                    help="optimizer choice - Adam with weight decay ('Adam') or SGD with L2 regularization ('SGDL2') ",type=str)
parser.add_argument("-w", "--weight_decay",
                    dest="WEIGHT_DECAY", default=1e-1,
                    help="weight decay for Adam optimizer or lambda for L2 regularization",type=float)
parser.add_argument("-i", "--init",
                    dest="INIT", default='noise',choices =['noise', 'molecule'],
                    help="initial input conditions - 'noise' or molecule from dataset ('molecule') ",type=str)
parser.add_argument("-t", "--target",
                    dest="TARGET", default=29,
                    help="target toxicity: -1 for all zeros, 0<target<29 for one in exact toxicity label, 29 for all ones",type=int)


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
    global MODEL_PATH_LOAD
    global MODEL_PATH_SAVE
    global DATASET_PATH
    global TARGET_NUM
    global VOXEL_DIM
    global AMOUNT_OF_ELEM
    global LOG_PATH_LOAD
    global LOG_PATH_SAVE
    global NUM_CONFS
    print(vars(args))
    LOG_PATH_LOAD = os.path.join(LOG_PATH_LOAD,'exp_'+args.NUM_EXP)
    path = os.path.join(LOG_PATH_LOAD, 'input_backprop')
    original_umask = os.umask(0)
    os.makedirs(path, mode=0o777, exist_ok=True)
    os.umask(original_umask)
    LOG_PATH_SAVE = path
    path = os.path.join(LOG_PATH_SAVE,'images')
    original_umask = os.umask(0)
    os.makedirs(path, mode=0o777, exist_ok=True)
    os.umask(original_umask)
    
    path = os.path.join(LOG_PATH_SAVE,'pymol')
    original_umask = os.umask(0)
    os.makedirs(path, mode=0o777, exist_ok=True)
    os.umask(original_umask)

    path = os.path.join(MODEL_PATH_LOAD,'exp_'+args.NUM_EXP)
    MODEL_PATH_LOAD = path
    os.makedirs(path, mode=0o777, exist_ok=True)
    os.umask(original_umask)
    MODEL_PATH_SAVE = path
        
    with open(os.path.join(LOG_PATH_LOAD,args.NUM_EXP+'_parameters.json'),'r') as f:
        args_dict = json.load(f)
    f_log=open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'w')
    f_log.close()
    start_time=time.time()
    writer=SummaryWriter(LOG_PATH_SAVE)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Using device:'+str(device)+'\n')
    print()
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
        with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write(torch.cuda.get_device_name(0)+'\n'+'Memory Usage:'+'\n'+'Allocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+ 'GB'+'\n'+'Cached:   '+str(round(torch.cuda.memory_cached(0)/1024**3,1))+'GB'+'\n')
    print('Start loading dataset...')
    with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Start loading dataset...'+'\n')
    # get dataset without duplicates from csv

    # create elements dictionary
#     elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM+1)
    elements={'N':0,'C':1,'Cl':2,'I':3,'Br':4,'F':5,'O':6,'P':7,'S':8}
    inv_elems = {v: k for k, v in elements.items()}


    
    # create indexing and label_dict for iteration
    
    start_time=time.time()
    # create train and validation sets' indexes
    print('Neural network initialization...')
    with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
        f_log.write('Neural network initialization...'+'\n') 
    if args.INIT == 'noise':
        molecule = Variable(torch.randn(1,9,50,50,50).to(device),requires_grad=True)
    elif args.INIT == 'molecule':
        pass
    model = Net(dim=args_dict['VOXEL_DIM'], num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, elements=elements, transformation=args_dict['TRANSF'],device=device,sigma_0 = args_dict['SIGMA'],sigma_trainable = args_dict['SIGMA_TRAIN'], x_trainable=True, x_input=torch.randn(1,9,50,50,50))
    model=model.to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH_LOAD,'checkpoint.pt')))
    model.x_input=Parameter(molecule,requires_grad=True)
    
    if args.TARGET == -1:
        target = torch.zeros(29)
    elif args.TARGET == 29:
        target = torch.ones(1,29)
    elif args.TARGET>=0 and args.TARGET<29:
        target = torch.tensor(1.0)
#     target=torch.tensor(1.0)
    
    target = target.to(device)

    
    
    with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_parameters.json'),'w') as f:
        json.dump(vars(args), f)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print ('Run in parallel!')
        with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Run in parallel!'+'\n')

    # Construct our model by instantiating the class defined above

    model=model.to(device)

    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())
    # set optimizer
    if args.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam([model.x_input], lr=args.LEARN_RATE, weight_decay=args.WEIGHT_DECAY)
    elif args.OPTIMIZER == 'SGDL2':
        optimizer = torch.optim.SGD([model.x_input], lr=args.LEARN_RATE)
    

    print('Neural network has been initialized, ', int(time.time()-start_time),' s')
    with open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_logs.txt'),'a') as f_log:
            f_log.write('Neural network has been initialized, '+str(int(time.time()-start_time))+' s'+'\n')

    
    
    f_train_loss=open(os.path.join(LOG_PATH_SAVE,args.NUM_EXP+'_log_train_loss.txt'),'w')
    
    start_time=time.time()
    # train procedure
    for epoch in range(args.EPOCHS_NUM):
        output = model(model.x_input)
        criterion=nn.MSELoss()
        if args.TARGET>=0 and args.TARGET<29:
            loss = criterion(output[args.TARGET], target)
        else:
            loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        writer.add_scalar('Loss/', loss.cpu().detach().numpy().item(), epoch)
        optimizer.step()
        if epoch%5==0:
            fig, ax = plot_visualization_input_as_parameter(model,elements,grad_step=10**3,name=str(epoch))
            fig.savefig(os.path.join(LOG_PATH_SAVE,'images','img_'+str(epoch))+'.png',dpi=150,format='png')
            
            for element in elements:
                s = VolToDx()(**{'volume':model.x_input.cpu().detach()[0,elements[element],:,:,:].squeeze().numpy(),'origin':np.array([-17.5,-17.5,-17.5]),'dsteps':np.array([0.5,0.5,0.5])})
                with open(os.path.join(LOG_PATH_SAVE,'pymol',element+'_pymol_'+str(epoch)+'.dx'),'w') as f:
                    f.write(s)
            s = VolToDx()(**{'volume':model.x_input.cpu().detach()[0,:,:,:,:].squeeze().numpy().sum(axis=0),'origin':np.array([-17.5,-17.5,-17.5]),'dsteps':np.array([0.5,0.5,0.5])})
            with open(os.path.join(LOG_PATH_SAVE,'pymol','pymol_'+str(epoch)+'.dx'),'w') as f:
                f.write(s)
        if epoch %100==0 and epoch>5000:
            torch.save(model.state_dict(),os.path.join(MODEL_PATH_SAVE,'model_'+str(epoch)+'.pt'))
        
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH_SAVE,'checkpoint.pt')))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH_SAVE, args.NUM_EXP+'_model'+str(epoch)+'_fin'))
    f_train_loss.close()
    f_test_loss.close()
    writer.close()
    print('Training has finished in ',round((time.time()-start_time)/60,3),' min.')
if __name__ == '__main__':
    sys.exit(main()) 

import load_data_multitox as ld
import dataloaders_sigma as dl

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

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

import time
from sklearn.preprocessing import StandardScaler


# number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS = 100

# amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM = 6


# dimension of voxel with conformer
global VOXEL_DIM
VOXEL_DIM = 70

# amount of target values
global TARGET_NUM
TARGET_NUM = 29

#dataset folder
global DATASET_PATH
DATASET_PATH="~/Tox21-MultiTox/MultiTox"

#logs path
global LOG_PATH
LOG_PATH="./logs_sigma/"

#models path
global MODEL_PATH
MODEL_PATH="./models_sigma"


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
                    dest="SIGMA", default=3,
                    help="sigma parameter",type=int)
parser.add_argument("-b", "--batch_size",
                    dest="BATCH_SIZE", default=32,
                    help="size of train and test batches",type=int)
parser.add_argument("-a", "--augmentation",
                    dest="AUG", default='g',choices =['g', 'w'],
                    help="type of augmentstion - g (gauss) or w (waves)")
parser.add_argument("-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment")

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

# create train and validation functions

def train(model, optimizer, train_generator, epoch, device, writer = None,f_loss=None,f_loss_ch = None, elements=None):
    elems=dict([(elements[element], element) for element in elements.keys()])
    model.train()
    train_loss=0
    losses=np.zeros(TARGET_NUM)
    num_losses=np.zeros(TARGET_NUM)
    for batch_idx, (data, target) in enumerate(train_generator):
        data = data.to(device)
        target = target.to(device)
        # set gradients to zero
        optimizer.zero_grad()
        output = model(data)
#         print(model.sigma.grad)
        i=0
        for one_target,one_output in zip(target.cpu().t(),output.cpu().t()):
            with torch.no_grad():
                
                mask = (one_target == one_target)
                output_masked = torch.masked_select(one_output, mask).type_as(one_output)
                target_masked = torch.masked_select(one_target, mask).type_as(one_output)
                criterion=nn.MSELoss()
                loss = criterion(output_masked.cpu(),target_masked.cpu())
                if loss == loss:
                    losses[i]+=loss
                    num_losses[i]+=1
#                        if f_loss is not None:
#                            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss.cpu().numpy().item())+'\n')
            i+=1
        # calculate output vector
        
        # create mask to get rid of Nan's in target
        mask = (target == target)
        output_masked = torch.masked_select(output, mask).type_as(output)
        target_masked = torch.masked_select(target, mask).type_as(output)
#        penalty_masked = torch.masked_select(PENALTY.to(device), mask).type_as(output)
#        pred = output_masked.ge(0.5).type_as(output)
#        try:
#            auc=roc_auc_score(target_masked.cpu().detach(),pred.cpu().detach())
#            if f_auc is not None:
#                f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(auc)+'\n')
#        except ValueError:
#            pass
        # multi-label (not multi-class!) classification=>binary cross entropy loss
#        class_weights=(1-penalty_masked)*(target_masked).to(device)+penalty_masked
        criterion=nn.MSELoss()
        loss = criterion(output_masked, target_masked)
        
        if f_loss is not None:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().detach().numpy().item())+'\n')
        loss.backward()
        optimizer.step()
        train_loss+=loss.cpu().detach().numpy().item()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item()))
    train_loss /= len(train_generator.dataset)
    train_loss *= args.BATCH_SIZE
    if writer is not None:
        writer.add_scalar('Train/Loss/', train_loss, epoch)
    sigmas = model.sigma.cpu().detach().numpy()
    for idx,sigma in enumerate(sigmas):
        writer.add_scalar('Sigma/'+elems[idx], sigma, epoch)
    losses/=num_losses    
    for i,loss in enumerate(losses):
        if f_loss_ch is not None and loss==loss:
            f_loss_ch.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss)+'\n')
        
        


def test(model, test_generator,epoch,device,writer=None,f_loss=None, elements=None):
#    print(f_auc is not None)
    with torch.no_grad():
        model.eval()
        test_loss = 0
        losses=np.zeros(TARGET_NUM)
        num_losses=np.zeros(TARGET_NUM)
        for batch_idx, (data, target) in enumerate(test_generator):
            data = data.to(device)
            target = target.to(device)
            output = model(data)   
            i=0
            for one_target,one_output in zip(target.cpu().t(),output.cpu().t()):
                with torch.no_grad():
                    mask = (one_target == one_target)
                    output_masked = torch.masked_select(one_output, mask).type_as(one_output)
                    target_masked = torch.masked_select(one_target, mask).type_as(one_output)
                    criterion=nn.MSELoss()
                    loss = criterion(output_masked.cpu(),target_masked.cpu())
                    if loss == loss:
                        losses[i]+=loss
                        num_losses[i]+=1
#                        if f_loss is not None:
#                            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss.cpu().numpy().item())+'\n')
                    i+=1
            mask = (target == target)
            output_masked = torch.masked_select(output, mask).type_as(output)
            target_masked = torch.masked_select(target, mask).type_as(output)
#            penalty_masked = torch.masked_select(PENALTY.to(device), mask).type_as(output)
#            class_weights=(1-penalty_masked)*(target_masked).to(device)+penalty_masked

            criterion=nn.MSELoss()
            loss = criterion(output_masked, target_masked)

            test_loss += loss
            
            
#            if f_loss is not None:
#                f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().numpy().item())+'\n')
#            total += output_masked.shape[0]
        test_loss /= len(test_generator.dataset)
        test_loss *= args.BATCH_SIZE

        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))
    if writer is not None:
        writer.add_scalar('Test/Loss/', test_loss, epoch)
    losses/=num_losses    
    for i,loss in enumerate(losses):
        if f_loss is not None and loss == loss:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss)+'\n')
    return test_loss


# create neural net
class Net(nn.Module):
    def __init__(self, dim=70, num_elems=6, num_targets=12, batch_size=args.BATCH_SIZE, aug = args.AUG, dx=0.5, kern_dim=50,elements=None,device='cpu',sigma_0=args.SIGMA):
        super(Net, self).__init__()
        
        self.sigma = Parameter(sigma_0*torch.ones(num_elems).float().to(device),requires_grad=True)
#         self.sigma = Parameter(torch.tensor(sigma_0).float().to(device),requires_grad=True)
#         print(self.sigma.grad)


        # initialize dimensions
        self.dim = dim
        self.num_elems = num_elems
        self.num_targets = num_targets
        self.batch_size = batch_size
        self.elements=elements
        self.dx=dx
        self.kern_dim=kern_dim
        self.aug=aug
        self.device=device
        self.elements=elements

        # create layers
        self.conv1 = nn.Conv3d(num_elems, 32, kernel_size=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, TARGET_NUM)

        # initialize dense layer's weights
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.convolution = nn.Sequential(
            self.conv1,
            self.pool1,
            nn.ReLU(),
            self.conv2,
            self.pool2,
            nn.ReLU(),
            self.conv3,
            self.pool3,
            nn.ReLU(),
            self.conv4,
            self.pool4,
            nn.ReLU()
        )

        def weights_init(m):
            if type(m) == nn.Conv3d:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # initialize convolutional layers' weights
        self.convolution.apply(weights_init)
        
    def blur (self, batch):#(molecule,elements,sigma=2,dimx=70,dx=0.5,kern_dim=50):
        from math import floor

        dimx=self.dim
        dx=self.dx
        kern_dim=self.kern_dim
        device=self.device

        batch=batch.to(device)
        dimelem=len(self.elements)
#         cube=torch.zeros((dimelem,dimx,dimx,dimx))
        cube=torch.zeros(batch.shape)
#         print(cube.shape)
        cube=cube.to(device)

        #build the kernel
#         x = torch.arange(-kern_dim/4,kern_dim/4,dx)

#         y = torch.arange(-kern_dim/4,kern_dim/4,dx)
#         z = torch.arange(-kern_dim/4,kern_dim/4,dx)
        x = torch.arange(-dimx/2,dimx/2)

        y = torch.arange(-dimx/2,dimx/2)
        z = torch.arange(-dimx/2,dimx/2)
        xx, yy, zz = torch.meshgrid((x,y,z))
        x=x.to(device)
        y=y.to(device)
        z=z.to(device)
        xx=xx.to(device)
        yy=yy.to(device)
        zz=zz.to(device)
#         print(self.aug)
        for idx,molecule in enumerate(batch):
#             cube[idx]=cube[idx]*self.sigma
            for num_atom in range(dimelem):
                for x0,y0,z0 in molecule[num_atom].nonzero():
                    if self.aug=='g':
                        cube[idx][num_atom] = cube[idx][num_atom] + torch.exp(-((xx-x0)**2 + (yy-y0)**2 + (zz-z0)**2)/(2*self.sigma[num_atom]**2))
                    if self.aug=='w':
                        cube[idx][num_atom] = cube[idx][num_atom] + torch.exp(-(xx**2 + yy**2 + zz**2)/(2*self.sigma[num_atom]**2))*torch.cos(2*np.pi/self.sigma[num_atom]*torch.sqrt(xx**2+yy**2+zz**2))
#         f = cube.sum()
#         f.backward()
#         print (self.sigma.grad)
        return cube

    def forward(self, x):
        x_cube = self.blur(x)
        x_conv = self.convolution(x_cube)
        x_vect = x_conv.view(x.shape[0], -1)
        y1 = F.relu(self.fc1(x_vect))
        y2=self.fc2(y1)
        #         multi-label (not multi-class!) classification => sigmoid non-linearity
        return y2

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(MODEL_PATH,'checkpoint.pt'))
        self.val_loss_min = val_loss
        

def main():
    path = os.path.join(LOG_PATH,'/exp_'+args.num_exp)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        LOG_PATH = path
    # setting device on GPU if available, else CPU
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
    data = pd.read_csv(os.path.join(DATASET_PATH+'/database', 'MultiTox.csv'))
    props = list(data)[1:]
    scaler = StandardScaler()
    data[props]=scaler.fit_transform(data[props])

    # create elements dictionary
    elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM)

    # read databases to dictionary
    conf_calc = ld.reading_sql_database(database_dir='./')
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
                                                         random_state=115)
    train_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, train_indexes)
    train_generator = td.DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)

    test_set = dl.Cube_dataset(conf_calc, label_dict, elements, indexing, test_indexes)
    test_generator = td.DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=False)
    
    model = Net(dim=VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, batch_size=args.BATCH_SIZE,elements=elements, aug=args.AUG,device=device)


    # Construct our model by instantiating the class defined above
#     model = Net(dim=VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, batch_size=args.BATCH_SIZE)
#     print(model)
    model=model.to(device)
#     summary(model, (6, 70, 70,70))
#     torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model_fin'))
    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#     optimizer.add_param_group({"params": model.sigma})
    print('Neural network has been initialized')
    
    f_train_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss.txt'),'w')
    f_train_loss_ch=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_train_loss_channels.txt'),'w')
    f_test_loss=open(os.path.join(LOG_PATH,args.NUM_EXP+'_log_test_loss.txt'),'w')
    
    early_stopping = EarlyStopping(patience=args.PATIENCE, verbose=True)

    
    # train procedure
    for epoch in range(1, args.EPOCHS_NUM + 1):
        try:
#            train(model, optimizer, train_generator_waves, epoch,device,f_auc=f_train_auc,f_loss=f_train_loss)
#            test(model, test_generator_waves,epoch, device,f_auc=f_test_auc,f_loss=f_test_loss)
            train(model, optimizer, train_generator, epoch,device,writer=writer,f_loss=f_train_loss,f_loss_ch=f_train_loss_ch, elements=elements)
            test_loss = test(model, test_generator,epoch, device,writer=writer,f_loss=f_test_loss, elements=elements)
            
        except KeyError:
            print('Key Error problem')
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch%100==0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, args.NUM_EXP+'_model_'+str(epoch)))
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'checkpoint.pt')))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, args.NUM_EXP+'_model'+str(epoch)+'_fin'))
    f_train_loss.close()
    f_test_loss.close()
    writer.close()
    print('Training has finished in ',round((time.time()-start_time)/60,3),' min.')
if __name__ == '__main__':
    sys.exit(main()) 

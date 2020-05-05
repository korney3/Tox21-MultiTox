#script with classification with new model and database of nine elements without H

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

from tensorboardX import SummaryWriter


# number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS = 100

# amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM = 9

# size of batch
global BATCH_SIZE
BATCH_SIZE = 32

# dimension of voxel with conformer
global VOXEL_DIM
VOXEL_DIM = 50

# amount of target values
global TARGET_NUM
TARGET_NUM = 12

#dataset folder
global DATASET_PATH
DATASET_PATH="/gpfs/gpfs0/a.alenicheva/Tox21/elements_9"

#logs path
global LOG_PATH
LOG_PATH="./Documents/Tox21_Neural_Net/logs10"

#models path
global MODEL_PATH
MODEL_PATH="./Documents/Tox21_Neural_Net/models10"

#number of epochs
global EPOCHS_NUM
EPOCHS_NUM=100

#loss penalty
global PENALTY
PENALTY = torch.FloatTensor([0.1,0.2,0.4,0.4,0.4,0.2,0.2,0.6,0.2,0.3,0.6,0.2])

#patience for early stopping
global PATIENCE
PATIENCE = 25

#sigma parameter for preprocessing
global SIGMA
SIGMA = 6

global TRANSF
TRANSF = 'w'

def conformer_choice(props):
    """choosing conformer of molecule according to probability

        Parameters
        ----------
        props
            dictionary of molecule conformers and its properties {conformer id: [probability of choice,[coordinates of atom]],...}

        Returns
        -------
        np.asscalar(conformer)
            number of conformer
        """
    probabilities=[props[key]['energy'] for key in props.keys()]
    conformer=np.random.choice(range(len(props)),1,probabilities)
    return np.asscalar(conformer)

#class of dataset creating cube for performing transformation
class Cube_dataset(td.Dataset):
    """
    The Cube_dataset constructs tensor of shape (num_elems, dim, dim ,dim) from smiles molecule description.

    Attributes
    ----------
    Xs : dict {smile:conformer:{energy:,coordinates:}}
        Dictionary with stored molecules and conformers info
    Ys : dict
        Dictionary contained labels for molecules
    elements: dict
        Dictionary with {atom name : number} mapping
    indexing : dict
        Dictionary with mapping number to smiles
    dx : float
        Size of grid cell in Angstrom
    indexes : list
        Set of indexes from indexing to  make dataset from
    dim : int
        Size of cube
    """
    def __init__(self,conf_calc,label_dict,elements,indexing, indexes,dx=0.5,dim=70,print_name=False):
        self.Xs=conf_calc
        self.Ys=label_dict
        self.elements=elements
        self.indexing = indexing
        self.dx = dx
        self.indexes=indexes
        self.dim = dim
        self.print_name = print_name

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        from math import floor
        'Generates one sample of data'
        dimelem = len(self.elements)
        
        cube=torch.zeros((dimelem,self.dim,self.dim,self.dim))
        
        i=self.indexes[index]
        smiles=self.indexing[i]
        
        if self.print_name:
            print(smiles)
        
        y= self.Ys[smiles]

        description=self.Xs[smiles][conformer_choice(self.Xs[smiles])]['coordinates']
        for atom in description.keys():
        
            num_atom=self.elements[atom]

            for x0,y0,z0 in description[atom]:
                if self.dim/2+x0/self.dx<self.dim-1 and self.dim/2+y0/self.dx<self.dim-1 and self.dim/2+z0/self.dx<self.dim-1:
                    cube[num_atom, min(self.dim -1,floor(self.dim/2+x0/self.dx)), min(self.dim -1,floor(self.dim/2+y0/self.dx)), min(self.dim -1,floor(self.dim/2+z0/self.dx))]=1
        X= cube
        return X, y
# create train and validation functions
def train(model, optimizer, train_generator, epoch, device, writer = None, f_auc=None,f_loss=None):
    model.train()
    aucs=np.zeros(TARGET_NUM)
    num_aucs=np.zeros(TARGET_NUM)
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_generator):
        data = data.to(device)
        target = target.to(device)
        # set gradients to zero
        optimizer.zero_grad()

        # calculate output vector
        output = model(data)
        i=0
        for one_target,one_output in zip(target.cpu().detach().t(),output.cpu().detach().t()):
            mask = (one_target == one_target)
            output_masked = torch.masked_select(one_output, mask).type_as(one_output)
            target_masked = torch.masked_select(one_target, mask).type_as(one_output)
            pred = output_masked.ge(0.5).type_as(one_output)
            try:
                auc=roc_auc_score(target_masked.cpu().detach(),pred.cpu().detach())
                aucs[i]+=auc
                num_aucs[i]+=1
                if f_auc is not None:
                    f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(auc)+'\n')
            except ValueError:
                pass
            i+=1
        # create mask to get rid of Nan's in target
        mask = (target == target)
        output_masked = torch.masked_select(output, mask).type_as(output)
        target_masked = torch.masked_select(target, mask).type_as(output)
        penalty_masked = torch.masked_select(PENALTY.to(device), mask).type_as(output)
#        pred = output_masked.ge(0.5).type_as(output)
#        try:
#            auc=roc_auc_score(target_masked.cpu().detach(),pred.cpu().detach())
#            if f_auc is not None:
#                f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(auc)+'\n')
#        except ValueError:
#            pass
        # multi-label (not multi-class!) classification=>binary cross entropy loss
        class_weights=(1-penalty_masked)*(target_masked).to(device)+penalty_masked

        loss = F.binary_cross_entropy(output_masked, target_masked,weight=class_weights)
        
        if f_loss is not None:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().detach().numpy().item())+'\n')
        loss.backward()
        optimizer.step()
        train_loss+=loss.cpu().detach().numpy().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item()))
    aucs=aucs/num_aucs
    for i,auc in enumerate(aucs):
        if writer is not None:
            writer.add_scalar('Train/AUC/'+str(i), auc, epoch)
    train_loss /= len(train_generator.dataset)
    train_loss *= BATCH_SIZE
    if writer is not None:
        writer.add_scalar('Train/Loss/'+str(epoch), train_loss, epoch)
        
        


def test(model, test_generator,epoch,device,writer=None,f_loss=None,f_auc=None):
#    print(f_auc is not None)
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        errors=0
        aucs=np.zeros(TARGET_NUM)
        num_aucs=np.zeros(TARGET_NUM)
        for batch_idx, (data, target) in enumerate(test_generator):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            i=0
            for one_target,one_output in zip(target.cpu().t(),output.cpu().t()):
                mask = (one_target == one_target)
                output_masked = torch.masked_select(one_output, mask).type_as(one_output)
                target_masked = torch.masked_select(one_target, mask).type_as(one_output)
                pred = output_masked.ge(0.5).type_as(one_output)
                try:
                    auc=roc_auc_score(target_masked.cpu(),pred.cpu())
                    aucs[i]+=auc
                    num_aucs[i]+=1
                    f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(auc)+'\n')
                except ValueError:
                    pass
                i+=1
            mask = (target == target)
            output_masked = torch.masked_select(output, mask).type_as(output)
            target_masked = torch.masked_select(target, mask).type_as(output)
            penalty_masked = torch.masked_select(PENALTY.to(device), mask).type_as(output)
            class_weights=(1-penalty_masked)*(target_masked).to(device)+penalty_masked

            loss = F.binary_cross_entropy(output_masked, target_masked,weight=class_weights)

            test_loss += loss
            pred = output_masked.ge(0.5).type_as(output)
            
            try:
                auc=roc_auc_score(target_masked.cpu(),pred.cpu())
#                if f_auc is not None:
#                    f_auc.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(auc)+'\n')
                correct += auc
            except ValueError:
                errors+=1
            if f_loss is not None:
                f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().numpy().item())+'\n')
#            total += output_masked.shape[0]
        test_loss /= len(test_generator.dataset)
        test_loss *= BATCH_SIZE
        batch_num=len(test_generator.dataset)/BATCH_SIZE-errors
        print('\nTest set: Average loss: {:.4f}, last AUC: {:.2f}, average AUC: {:.2f}\n'
              .format(test_loss, auc, correct/batch_num))
    aucs=aucs/num_aucs
    for i,auc in enumerate(aucs):
        if writer is not None:
            writer.add_scalar('Test/AUC/'+str(i), auc, epoch)
    if writer is not None:
        writer.add_scalar('Test/Loss/'+str(epoch), test_loss, epoch)
    return test_loss


# create neural net
class Net(nn.Module):
    """
    The Net class constructs neural network with ActivNet4 architecture.

    Attributes
    ----------
    dim : int
        Dimension of 3D cube where each type of atoms are stored
    num_elems : int
        Number of types of atoms represented molecule (number of cubes storing information about molecule's structure) 
    num_targets : int
        Number of predicted labels
    transform : str
        Type of transformation applied to atom grid:
        'g' - Gauss transformation
        'w' - Waves transformation
    dx : float
        Size of grid cell in angstrom
    elements: dict
        Dictionary with {atom name : number} mapping
    device : str
        Torch device
    sigma : torch.Tensor
        Tensor containing sigmas for each type of atom
        
    convolution : nn.Sequential
        Set of convolutions, pooling and non-linearities 
    fc1 : nn.Linear
        First dense layer
    fc2 : nn.Linear
        Second dense layer
        
    blur : function
        Apply transformation to batch of molecules
    forward : function
        Apply neural network to batch of molecules
    """
    def __init__(self, dim=50, kernel_size=51,
                 num_elems=9, 
                 num_targets=12, 
                 transformation='g', 
                 dx=0.5,
                 elements=None,
                 device='cpu',
                 sigma_trainable = False,
                 sigma_0=3, 
                 x_trainable = False,
                 mode = 'c',
                 x_input=None):
        """
        Initialize neural network.

        Parameters
        ----------
        dim : int
            Dimension of 3D cube where each type of atoms are stored
        kernel_size : int
            Size of convolution kernel for gauss or wave transformation
        num_elems : int
            Number of types of atoms represented molecule (number of cubes storing information about molecule's structure) 
        num_targets : int
            Number of predicted labels
        transformation : str
            Type of transformation applied to atom grid:
            'g' - Gauss transformation
            'w' - Waves transformation
        dx : float
            Size of grid cell in angstrom
        elements: dict
            Dictionary with {atom name : number} mapping
        device : str
            Torch device
        sigma_trainable : boolean
            Should sigma be trainable parameter or not
        sigma_0 : float or numpy array (len(elements),)
            Initial value of sigma parameter (in grid cells)
        """
        super(Net, self).__init__()
        from math import floor
        if sigma_trainable:
            self.sigma = Parameter(sigma_0*torch.ones(num_elems).float().to(device),requires_grad=True)
            self.register_parameter('sigma',self.sigma)
        else:
            self.register_buffer('sigma', sigma_0*torch.ones(num_elems).float().to(device))
            
        if x_trainable:
            self.x_input = Parameter(x_input.to(device),requires_grad=True)
            self.register_parameter('x_input',self.x_input)
        else:
            self.register_buffer('x_input',torch.zeros(1, num_elems, dim, dim, dim).float().to(device))
        

        

        # initialize dimensions
        self.dim = dim
        self.kernel_size=kernel_size
        self.num_elems = num_elems
        self.num_targets = num_targets
        self.elements=elements
        self.dx=dx
        self.transform=transformation
        self.device=device
        self.elements=elements
        self.mode = mode
        # create layers
        self.conv1 = nn.Conv3d(num_elems, 32, kernel_size=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, num_targets)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.num_targets)

        # initialize dense layer's weights
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

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
        
        with torch.no_grad():
            

            dimx=dim
            device=self.sigma.device
            kernel_size=self.kernel_size


            x = torch.arange(0,dimx+1).float()
            y = torch.arange(0,dimx+1).float()
            z = torch.arange(0,dimx+1).float()
            xx, yy, zz = torch.meshgrid((x,y,z))
            xx=xx.reshape(dimx+1,dimx+1,dimx+1,1)
            yy=yy.reshape(dimx+1,dimx+1,dimx+1,1)
            zz=zz.reshape(dimx+1,dimx+1,dimx+1,1)
            xx = xx.repeat( 1, 1, 1, self.num_elems)
            yy = yy.repeat( 1, 1, 1, self.num_elems)
            zz = zz.repeat( 1, 1, 1, self.num_elems)

            xx=xx.to(device)
            yy=yy.to(device)
            zz=zz.to(device)         

            mean = (kernel_size - 1)/2.
            variance = self.sigma**2.
            omega = 1/self.sigma
            if self.transform=='g':
                a = (1./(2.*np.pi*variance))
                b = -((xx-mean)**2+(yy-mean)**2+(zz-mean)**2)/(2*variance)
                kernel = a*torch.exp(b)
            if self.transform=='w':
                kernel = torch.exp(-((xx-mean)**2+(yy-mean)**2+(zz-mean)**2)/(2*variance))*torch.cos(2*np.pi*omega*torch.sqrt(((xx-mean)**2+(yy-mean)**2+(zz-mean)**2)))
            kernel=torch.transpose(kernel, 3,0)
            kernel = kernel / torch.sum(kernel)

            self.kernel = kernel.view(self.num_elems, 1, kernel_size, kernel_size, kernel_size)

    def blur (self,batch):
        """ Applying Gauss or Wave transformation to batch of cubes

        Parameters
        ----------
        batch
            Batch of torch tensors with shape (batch_size, num_elems, dim, dim ,dim)

        Returns
        -------
        torch.Tensor 
            Tensor of shape (batch_size, num_elems, dim, dim ,dim) fulfilled with transformation
        """

        
        res = F.conv3d(batch, weight=self.kernel, bias=None, padding=25,groups=self.num_elems)
        res -= res.min()
        res /= res.max()
        return  res

    def forward(self, x):
        """ Applying Neural Network transformation to batch of molecules:
            blur, convolution, view, fc, relu, fc

        Parameters
        ----------
        batch
            Batch of torch tensors with shape (batch_size, num_elems, dim, dim ,dim)

        Returns
        -------
        torch.Tensor 
            Tensor of shape (batch_size, num_targets) fulfilled with predicted values
        """

        x_cube = self.blur(x)
        x_conv = self.convolution(x_cube)
        x_vect = x_conv.view(x.shape[0], -1)
        y1 = F.relu(self.fc1(x_vect))
        y2=self.fc2(y1)
        if self.mode == 'r':
            return y2
        elif self.mode == 'c':
            return torch.sigmoid(y2)

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
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        

def main():
    # setting device on GPU if available, else CPU
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
    
    data = pd.read_csv(os.path.join('../Tox21_Neural_Net/database', 'tox21_10k_data_all_no_salts.csv'))

    # create elements dictionary
#     elements = ld.create_element_dict(data, amount=AMOUNT_OF_ELEM)
    elements = {'S': 0, 'Cl': 1, 'N': 2, 'O': 3, 'C': 4, 'H': 5}
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
    conf_calc = ld.reading_sql_database(os.path.join(DATASET_PATH, 'tox21_10k_data_all_no_salts.db'))
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
#     train_set = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, train_indexes, sigma=SIGMA)
#     train_generator = td.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

#     test_set = dl.Gauss_dataset(conf_calc, label_dict, elements, indexing, test_indexes, sigma=SIGMA)
#     test_generator = td.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # make dataloader for Waves transformation
    train_set = Cube_dataset(conf_calc, label_dict, elements, indexing, train_indexes, dim = VOXEL_DIM)
    train_generator = td.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = Cube_dataset(conf_calc, label_dict, elements, indexing, test_indexes, dim = VOXEL_DIM)
    test_generator = td.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Construct our model by instantiating the class defined above
    model = Net(dim=VOXEL_DIM, num_elems=AMOUNT_OF_ELEM, num_targets=TARGET_NUM, sigma_0=SIGMA, elements = elements, device = device,transformation=TRANSF)
    print(model)
    model=model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    print('Neural network has been initialized')
    
    f_train_loss=open(LOG_PATH+'/log_train_loss.txt','w')
    f_test_loss=open(LOG_PATH+'/log_test_loss.txt','w')
    f_train_auc=open(LOG_PATH+'/log_train_auc.txt','w')
    f_test_auc=open(LOG_PATH+'/log_test_auc.txt','w')
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    
    # train procedure
    for epoch in range(1, EPOCHS_NUM + 1):
        try:
#            train(model, optimizer, train_generator_waves, epoch,device,f_auc=f_train_auc,f_loss=f_train_loss)
#            test(model, test_generator_waves,epoch, device,f_auc=f_test_auc,f_loss=f_test_loss)
            train(model, optimizer, train_generator, epoch,device,writer=writer,f_auc=f_train_auc,f_loss=f_train_loss)
            test_loss = test(model, test_generator,epoch, device,writer=writer,f_auc=f_test_auc,f_loss=f_test_loss)
            
        except KeyError:
            print('Key Error problem')
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
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

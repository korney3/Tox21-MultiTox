import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import os


import torch.nn.functional as F
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
    def __init__(self, dim=70, kernel_size=51,
                 num_elems=6, 
                 num_targets=29, 
                 transformation='g', 
                 dx=0.5,
                 elements=None,
                 device='cpu',
                 sigma_trainable = False,
                 sigma_0=3, 
                 x_trainable = False,
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
        
        if sigma_trainable:
            self.sigma = Parameter(sigma_0*torch.ones(num_elems).float().to(device),requires_grad=True)
            self.register_parameter('sigma',self.sigma)
        else:
#             self.sigma = 
            self.register_buffer('sigma', sigma_0*torch.ones(num_elems).float().to(device))
            
        if x_trainable:
            self.x_input = Parameter(x_input.to(device),requires_grad=True)
            self.register_parameter('x_input',self.x_input)
        else:
#             self.x_input = 
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
        self.fc2 = nn.Linear(128, num_targets)

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

        from math import floor

        dimx=self.dim
        dx=self.dx
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
        
        kernel = kernel.view(self.num_elems, 1, kernel_size, kernel_size, kernel_size)
        res = F.conv3d(batch, weight=kernel, bias=None, padding=25,groups=self.num_elems)
        res = res/res.max()
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

        return y2
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False,model_path='./'):
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
        self.model_path=model_path

    def __call__(self, val_loss, model):
        
        global MODEL_PATH

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
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(self.model_path,'checkpoint.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(self.model_path,'checkpoint.pt'))
        self.val_loss_min = val_loss
    
def train(model, optimizer, train_generator, epoch, device, batch_size, num_targets=29, writer = None,f_loss=None,f_loss_ch = None, elements=None, MODEL_PATH=None):
    """ Train model and write logs to tensorboard and .txt files

        Parameters
        ----------
        model
            torch.nn.Module object to train
        optimizer
            torch.optim object
        train_generator
            torch.utils.data.DataLoader object, contain iterable set of torch.Tensor data (num_elems, dim,dim,dim) and torch.Tensor labels (num_targets, )
        epoch
            number of trained epoch
        device
            torch.device
        writer
            tensorboardX.SummaryWriter
        f_loss
            .txt file for train loss saving
        f_loss_ch
            .txt file for loss per target saving
        elements
            dictionary with {atom name : number} mapping

        Returns
        -------
        None
        """
    elems=dict([(elements[element], element) for element in elements.keys()])
    model.train()
    train_loss=0
    losses=np.zeros(num_targets)
    num_losses=np.zeros(num_targets)
    for batch_idx, (data, target) in enumerate(train_generator):
        data = data.to(device)
        target = target.to(device)
        # set gradients to zero
        optimizer.zero_grad()
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

            i+=1
        # calculate output vector
        
        # create mask to get rid of Nan's in target
        mask = (target == target)
        output_masked = torch.masked_select(output, mask).type_as(output)
        target_masked = torch.masked_select(target, mask).type_as(output)
        criterion=nn.MSELoss()
        loss = criterion(output_masked, target_masked)
        
        if f_loss is not None:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(loss.cpu().detach().numpy().item())+'\n')
        loss.backward()
        optimizer.step()
        train_loss+=loss.cpu().detach().numpy().item()
        
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item()))
            if writer is not None:
                writer.add_scalar('iters/Train/Loss/', train_loss, batch_idx)
            if torch.cuda.device_count() > 1:
                sigmas = model.module.sigma.cpu().detach().numpy()
            else:
                sigmas = model.sigma.cpu().detach().numpy()
            for idx,sigma in enumerate(sigmas):
                writer.add_scalar('/iters/Sigma/'+elems[idx], sigma, batch_idx)
            losses/=num_losses    
            for i,loss in enumerate(losses):
                if f_loss_ch is not None and loss==loss:
                    f_loss_ch.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss)+'\n')
                    writer.add_scalar('/iters/Train/Loss/'+str(i), loss, batch_idx)
            if MODEL_PATH is not None:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), os.path.join(MODEL_PATH,'checkpoint.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(MODEL_PATH,'checkpoint.pt'))
         
                
    train_loss /= len(train_generator.dataset)
    train_loss *= batch_size
    if writer is not None:
        writer.add_scalar('Train/Loss/', train_loss, epoch)
    if torch.cuda.device_count() > 1:
        sigmas = model.module.sigma.cpu().detach().numpy()
    else:
        sigmas = model.sigma.cpu().detach().numpy()
    for idx,sigma in enumerate(sigmas):
        writer.add_scalar('Sigma/'+elems[idx], sigma, epoch)
    losses/=num_losses    
    for i,loss in enumerate(losses):
        if f_loss_ch is not None and loss==loss:
            f_loss_ch.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss)+'\n')
            writer.add_scalar('Train/Loss/'+str(i), loss, epoch)
    return
        
        


def test(model, test_generator,epoch,device,batch_size,num_targets=29,writer=None,f_loss=None, elements=None):
    """ Validation of trained model

        Parameters
        ----------
        model
            torch.nn.Module object to train
        test_generator
            torch.utils.data.DataLoader object, contain iterable set of torch.Tensor data (num_elems, dim,dim,dim) and torch.Tensor labels (num_targets, )
        epoch
            number of validated epoch
        device
            torch.device
        writer
            tensorboardX.SummaryWriter
        f_loss
            .txt file for test loss saving
        elements
            dictionary with {atom name : number} mapping

        Returns
        -------
        test_loss
            Loss for validation set
        """
    with torch.no_grad():
        model.eval()
        test_loss = 0
        losses=np.zeros(num_targets)
        num_losses=np.zeros(num_targets)
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

                    i+=1
            mask = (target == target)
            output_masked = torch.masked_select(output, mask).type_as(output)
            target_masked = torch.masked_select(target, mask).type_as(output)


            criterion=nn.MSELoss()
            loss = criterion(output_masked, target_masked)

            test_loss += loss
            
        test_loss /= len(test_generator.dataset)
        test_loss *= batch_size

        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))
    if writer is not None:
        writer.add_scalar('Test/Loss/', test_loss, epoch)
    losses/=num_losses    
    for i,loss in enumerate(losses):
        if f_loss is not None and loss == loss:
            f_loss.write(str(epoch)+'\t'+str(batch_idx)+'\t'+str(i)+'\t'+str(loss)+'\n')
            writer.add_scalar('Test/Loss/'+str(i), loss, epoch)
    return test_loss
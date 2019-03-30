# -*- coding: utf-8 -*-

#import all the necessary libraries


import numpy as np

import torch
from torch.utils import data as td

#number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS=100

#amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM=6

#choosing conformer of molecule according to probability
#input - dictionary of molecule conformers and its properties {conformer id: [probability of choice,
#[coordinates of atom]],...}
#output - number of conformer

def conformer_choice(props):
    probabilities=[props[key]['energy'] for key in props.keys()]
    conformer=np.random.choice(range(len(props)),1,probabilities)
    return np.asscalar(conformer)

#gaussian blur 3D cordinate transformation
#input - molecule - dictionary{atom:[(x1,x2,x3),...]}
#sigma - parameter of kernel

def gaussian_blur(molecule,elements,sigma=2,dimx=70,dx=0.5,kern_dim=50):
    from math import floor
    
    dimelem=len(elements)
    cube=torch.zeros((dimelem,dimx,dimx,dimx))
    
    #build the kernel
    x = torch.arange(-kern_dim/4,kern_dim/4,dx)
    y = torch.arange(-kern_dim/4,kern_dim/4,dx)
    z = torch.arange(-kern_dim/4,kern_dim/4,dx)
    xx, yy, zz = torch.meshgrid((x,y,z))
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    
    for atom in molecule.keys():
        
        num_atom=elements[atom]
        
        for x0,y0,z0 in molecule[atom]:
            
            x_range=[max(floor(x[0]/dx+x0/dx+dimx/2),0),min(floor(x[-1]/dx+x0/dx+dimx/2+1),cube.shape[1])]
            y_range=[max(floor(y[0]/dx+y0/dx+dimx/2),0),min(floor(y[-1]/dx+y0/dx+dimx/2+1),cube.shape[2])]
            z_range=[max(floor(z[0]/dx+z0/dx+dimx/2),0),min(floor(z[-1]/dx+z0/dx+dimx/2+1),cube.shape[3])]
            coord_ranges=[x_range,y_range,z_range]
            for i in range(3):
                if coord_ranges[i][1]-coord_ranges[i][0]>50:
                    coord_ranges[i][1]=coord_ranges[i][0]+50
            cube_part=cube[num_atom,coord_ranges[0][0]:coord_ranges[0][1],
                           coord_ranges[1][0]:coord_ranges[1][1],
                           coord_ranges[2][0]:coord_ranges[2][1]]
            
            kern_ranges=[[],[],[]]
            for i in range(3):
                if coord_ranges[i][0]==0:
                    kern_ranges[i].append(kern_dim-cube_part.shape[i])
                else:
                    kern_ranges[i].append(0)
                if coord_ranges[i][1]==cube.shape[i+1]:
                    kern_ranges[i].append(cube_part.shape[i])
                else:
                    kern_ranges[i].append(kern_dim)
                    
            cube_part=cube_part+kernel[kern_ranges[0][0]:kern_ranges[0][1],
                                       kern_ranges[1][0]:kern_ranges[1][1],
                                       kern_ranges[2][0]:kern_ranges[2][1]]
            
            cube[num_atom,coord_ranges[0][0]:coord_ranges[0][1],
                 coord_ranges[1][0]:coord_ranges[1][1],
                 coord_ranges[2][0]:coord_ranges[2][1]]=cube_part

    return cube

#waves 3D cordinate transformation
#input - molecule - dictionary{atom:[(x1,x2,x3),...]}
#sigma - parameter of kernel

def waves(molecule,elements,sigma=1,dimx=70,dx=0.5,kern_dim=50):
    from math import floor
    omega=1/sigma
    dimelem=len(elements)
    cube=torch.zeros((dimelem,dimx,dimx,dimx))
    
    #build the kernel
    x = torch.arange(-kern_dim/4,kern_dim/4,dx)  
    y = torch.arange(-kern_dim/4,kern_dim/4,dx)
    z = torch.arange(-kern_dim/4,kern_dim/4,dx)
    xx, yy, zz = torch.meshgrid((x,y,z))
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))*np.cos(2*np.pi*omega*np.sqrt(xx**2+yy**2+zz**2))
    
    for atom in molecule.keys():
        num_atom=elements[atom]
        
        for x0,y0,z0 in molecule[atom]:
            
            x_range=[max(floor(x[0]/dx+x0/dx+dimx/2),0),min(floor(x[-1]/dx+x0/dx+dimx/2+1),cube.shape[1])]
            y_range=[max(floor(y[0]/dx+y0/dx+dimx/2),0),min(floor(y[-1]/dx+y0/dx+dimx/2+1),cube.shape[2])]
            z_range=[max(floor(z[0]/dx+z0/dx+dimx/2),0),min(floor(z[-1]/dx+z0/dx+dimx/2+1),cube.shape[3])]
            coord_ranges=[x_range,y_range,z_range]
            for i in range(3):
                if coord_ranges[i][1]-coord_ranges[i][0]>kern_dim:
                    coord_ranges[i][1]=coord_ranges[i][0]+kern_dim
            cube_part=cube[num_atom,coord_ranges[0][0]:coord_ranges[0][1],
                           coord_ranges[1][0]:coord_ranges[1][1],
                           coord_ranges[2][0]:coord_ranges[2][1]]
            
            kern_ranges=[[],[],[]]
            for i in range(3):
                if coord_ranges[i][0]==0:
                    kern_ranges[i].append(kern_dim-cube_part.shape[i])
                else:
                    kern_ranges[i].append(0)
                if coord_ranges[i][1]==cube.shape[i+1]:
                    kern_ranges[i].append(cube_part.shape[i])
                else:
                    kern_ranges[i].append(kern_dim)
                    
            cube_part=cube_part+kernel[kern_ranges[0][0]:kern_ranges[0][1],
                                       kern_ranges[1][0]:kern_ranges[1][1],
                                       kern_ranges[2][0]:kern_ranges[2][1]]
            
            cube[num_atom,coord_ranges[0][0]:coord_ranges[0][1],
                 coord_ranges[1][0]:coord_ranges[1][1],
                 coord_ranges[2][0]:coord_ranges[2][1]]=cube_part
    return cube

#class of dataset created by gauss transformation
class Gauss_dataset(td.Dataset):
    def __init__(self,conf_calc,label_dict,elements,indexing, indexes, sigma=1,dim=70,dx=0.5,kern_dim=50):
        self.Xs=conf_calc
        self.Ys=label_dict
        self.elements=elements
        self.indexing = indexing
        self.sigma=sigma
        self.dim=dim
        self.dx=dx
        self.kern_dim=kern_dim
        self.indexes=indexes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'

        i=self.indexes[index]
        smiles=self.indexing[i]
        
        y= self.Ys[smiles]

        description=self.Xs[smiles][conformer_choice(self.Xs[smiles])]['coordinates']
        X = gaussian_blur(description,self.elements,sigma=self.sigma,dimx=self.dim,dx=self.dx,kern_dim=self.kern_dim)
        
        return X, y
    
#class of dataset created by waves transformation
class Waves_dataset(td.Dataset):
    def __init__(self,conf_calc,label_dict,elements,indexing,indexes, sigma=1,dim=70,dx=0.5,kern_dim=50):
        self.Xs=conf_calc
        self.Ys=label_dict
        self.elements=elements
        self.indexing = indexing
        self.sigma=sigma
        self.dim=dim
        self.dx=dx
        self.kern_dim=kern_dim
        self.indexes = indexes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'

        i = self.indexes[index]
        smiles = self.indexing[i]

        y= self.Ys[smiles]

        description=self.Xs[smiles][conformer_choice(self.Xs[smiles])]['coordinates']
        X = waves(description,self.elements,sigma=self.sigma,dimx=self.dim,dx=self.dx,kern_dim=self.kern_dim)
        
        return X, y
# -*- coding: utf-8 -*-

#import all the necessary libraries
import numpy as np

import torch
from torch.utils import data as td

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

def gaussian_blur(molecule,elements,sigma=3,dimx=50,dx=0.5,kern_dim=50):
    """calculate gaussian blur of the molecule

        Parameters
        ----------
        molecule
            dictionary {type of atom:[(x1,x2,x3),...]}
        elements
            dictionary {type of atom:number}
        sigma
            integer or vector of sigma parameter
        dimx
            integer of size of cube of voxels
        dx
            float angstrom per grid cell
        kern_dim
            integer size of kernel in voxels

        Returns
        -------
        cube
            transformed voxels of molecule
        """
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

    cube-=cube.min()
    cube/=cube.max()
    return cube


def waves(molecule,elements,sigma=6,dimx=50,dx=0.5,kern_dim=50):
    """calculate waves transformation of the molecule

        Parameters
        ----------
        molecule
            dictionary {type of atom:[(x1,x2,x3),...]}
        elements
            dictionary {type of atom:number}
        sigma
            integer or vector of sigma parameter
        dimx
            integer of size of cube of voxels
        dx
            float angstrom per grid cell
        kern_dim
            integer size of kernel in voxels

        Returns
        -------
        cube
            transformed voxels of molecule
        """
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
    cube-=cube.min()
    cube/=cube.max()
    return cube

class Gauss_dataset(td.Dataset):
    """
    The Gauss_dataset constructs transformed with gaussian blur tensor of shape (num_elems, dim, dim ,dim) from smiles molecule description.

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
    sigma : float
        Parameter of transformation
    kern_dim : int
        Size of kernel for transformation
    """
    def __init__(self,conf_calc,label_dict,elements,indexing, indexes, sigma=3,dim=50,dx=0.5,kern_dim=50):
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
    
class Waves_dataset(td.Dataset):
    """
    The Gauss_dataset constructs transformed with waves tensor of shape (num_elems, dim, dim ,dim) from smiles molecule description.

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
    sigma : float
        Parameter of transformation
    kern_dim : int
        Size of kernel for transformation
    """
    def __init__(self,conf_calc,label_dict,elements,indexing,indexes, sigma=6,dim=50,dx=0.5,kern_dim=50):
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
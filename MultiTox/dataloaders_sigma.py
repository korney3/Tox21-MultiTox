# -*- coding: utf-8 -*-
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
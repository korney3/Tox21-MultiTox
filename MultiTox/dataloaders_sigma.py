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

#class of dataset creating cube for performing transformation
class Cube_dataset(td.Dataset):
    def __init__(self,conf_calc,label_dict,elements,indexing, indexes,dx=0.5,dim=70):
        self.Xs=conf_calc
        self.Ys=label_dict
        self.elements=elements
        self.indexing = indexing
        self.dx = dx
        self.indexes=indexes
        self.dim = dim

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
        
        y= self.Ys[smiles]

        description=self.Xs[smiles][conformer_choice(self.Xs[smiles])]['coordinates']
        for atom in description.keys():
        
            num_atom=self.elements[atom]

            for x0,y0,z0 in description[atom]:
                cube[num_atom, min(self.dim -1,floor(self.dim/2+x0/self.dx)), min(self.dim -1,floor(self.dim/2+y0/self.dx)), min(self.dim -1,floor(self.dim/2+z0/self.dx))]=1
        X= cube
        return X, y
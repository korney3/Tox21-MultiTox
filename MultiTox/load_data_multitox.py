# -*- coding: utf-8 -*-
#import all the necessary libraries

import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import torch
import sqlite3
import glob
import os

#number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS=100

#amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM=6

#define what chemical elements are used in molecules
#output - dictionary {element: number}
def create_element_dict(data,amount=AMOUNT_OF_ELEM,treshold=10):
    elements={}
    norm=0
    for smile in data['SMILES']:
        molecule=Chem.MolFromSmiles(smile)
        molecule=Chem.AddHs(molecule)
        
        for i in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(i)
            element=atom.GetSymbol()
            norm+=1
            if element in elements.keys():
                elements[element]+=1
            else:
                elements[element]=1
    for key in elements.keys():
        elements[key]/=norm
    from collections import OrderedDict
    dd = OrderedDict(sorted(elements.items(), key=lambda x: x[1]))
    
#    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharey=False, constrained_layout=True)
#    axs.bar(list(dd.keys())[-amount:],list(dd.values())[-amount:])
#    axs.set_ylabel('Relative amount of atoms in dataset',fontsize=12)
#    axs.set_xticklabels(list(dd.keys())[-amount:],fontsize=15)

    elements=list(dd.keys())[-amount:]  
    elements=dict((elem,i) for i, elem in enumerate(elements))            
    return elements

#reading data from sql database to dictionary
def reading_sql_database(database_dir='./database'):
    
    conf_calc={}
    
    for filename in glob.glob(os.path.join(database_dir,'*.db')):
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        for row in c.execute('SELECT * FROM tox'):
            smile, conformer,energy,type_of_atom,x,y,z = row
            if smile in conf_calc.keys():
                if conformer in conf_calc[smile].keys():
                    conf_calc[smile][conformer]['energy']=energy
                    if 'coordinates' in conf_calc[smile][conformer].keys():
                        if type_of_atom in conf_calc[smile][conformer]['coordinates'].keys():
                            conf_calc[smile][conformer]['coordinates'][type_of_atom].append((x,y,z))
                        else:
                            conf_calc[smile][conformer]['coordinates'][type_of_atom]=[]
                    else:
                        conf_calc[smile][conformer]['coordinates']={}
                else:
                    conf_calc[smile][conformer]={}
    
            else:
                conf_calc[smile]={}
        conn.close()
    return conf_calc

#set a number to each smile and get a dictionary indexing:{number:smile}
#and dictionary of labels for each smile label_dict {smile:labels}
def indexing_label_dict(data,conf_calc):
    props=list(data)
    props.remove('SMILES')
    label_dict={}
    indexing={}
    for (i,smiles) in enumerate(conf_calc.keys()):
        labels=data.loc[data['SMILES']==smiles][props].values[0]
        label_dict[smiles]=torch.from_numpy(labels)
        indexing[i]=smiles
        
    return indexing, label_dict
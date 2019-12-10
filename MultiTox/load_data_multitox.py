# -*- coding: utf-8 -*-
#import all the necessary libraries

import numpy as np
from rdkit import Chem
import torch

import sqlite3
import glob
import os


def create_element_dict(data,amount=9, add_H=False):
    """Define what chemical elements are used in molecules

        Parameters
        ----------
        data
            pandas.DataFrame containing smiles of molecules in dataset
        amount
            Number of types of atoms to store
        add_H
            True or False: store info of H atoms or not        

        Returns
        -------
        elements
            dictionary with {atom name : number} mapping
        """
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
    elements=list(dd.keys())[-amount:]  
    elements=dict((elem,i) for i, elem in enumerate(elements))  
    if not add_H:
        del elements['H']
    return elements


def reading_sql_database(database_dir='./database'):
    """Reading data from sql database to dictionary

        Parameters
        ----------
        database_dir
            directory stored files in .db format
        
        Returns
        -------
        conf_calc
            dictionary with {smile : conformer : {energy:, coordinates:}} information
        """
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

#
#and dictionary of labels for each smile label_dict {smile:labels}
def indexing_label_dict(data,conf_calc):
    """Set a number to each smile and get a dictionary indexing:{number:smile}
    and dictionary of labels for each smile label_dict {smile:labels}

        Parameters
        ----------
        data
            pandas.DataFrame containing smiles of molecules in dataset
            
        conf_calc
            dictionary with {smile : conformer : {energy:, coordinates:}} information
        
        Returns
        -------
        indexing
            dictionary with {number:smile} mapping
        label_dict
            dictionary with {smile:labels} mapping
        """
    props=list(data)
    props.remove('SMILES')
    label_dict={}
    indexing={}
    for (i,smiles) in enumerate(conf_calc.keys()):
        labels=data.loc[data['SMILES']==smiles][props].values[0]
        label_dict[smiles]=torch.from_numpy(labels)
        indexing[i]=smiles
        
    return indexing, label_dict
# -*- coding: utf-8 -*-
#import all the necessary libraries

import pandas as pd
import numpy as np
from rdkit import Chem
import molvs as mv

#function removing salts (leave the longest part of the molecule)

def del_salts(molecule):
    s=mv.standardize.Standardizer()
    stand_mol=s.fragment_parent(molecule)
    return stand_mol

#converting energiies in possibilities

def possibility_norm(conf_calc, a=1):
    from math import exp
    
    if conf_calc=={}:
        return conf_calc
    
    energies=[x[0] for x in conf_calc.values()]
    norm=max(energies)
    
    for key in conf_calc.keys():
        conf_calc[key][0]/=norm
        conf_calc[key][0]=exp(-a*conf_calc[key][0])
        
    energies=[x[0] for x in conf_calc.values()]
    norm=sum(energies)
    
    for key in conf_calc.keys():
        conf_calc[key][0]/=norm
        
    return conf_calc

#merging features of the same molecules

def merging_rows(row):
    row_without_nan=[x for x in row if x is not np.nan]
    
    if not row_without_nan:
        return np.nan
    
    if len(np.unique(row_without_nan))>1:
        return np.nan
    else:
        return row_without_nan[0]
    
#create pandas dataframe fron sdf file

def create_dataset(filename="tox21_10k_data_all.sdf"):
    name=filename.split('.')[0]
    suppl = Chem.SDMolSupplier(filename)
    #the list of molecules
    dataset=[]
    
    for i,molecule in enumerate(suppl):
        try:
            row = {}
            
            #converting molecule from file to smile representation
            smile=Chem.MolToSmiles(molecule)
            row['smiles'] = smile
            
            #removing salts
            stand_mol=del_salts(molecule)
            smile_no_salt=Chem.MolToSmiles(stand_mol)
            row['smiles_no_salt'] = smile_no_salt
            
            #add properties from file
            for propname in molecule.GetPropNames():
                row[propname] = molecule.GetProp(propname)
            dataset.append(row)
        except:
            print ("Can't smile molecule ",i)

    data = pd.DataFrame(dataset)
    
    #drop useless info
    try:
        data=data.drop(columns=['DSSTox_CID','FW','Formula','Unnamed:\n0'])
    except:
        data=data.drop(columns=['DSSTox_CID','FW','Formula'])
    data.to_csv(name + '.csv',index=False)
    return data

#create unique "smiles_no_salt" values
#drop molecule if there is contradicted info
#merge nan and 0/1 values

def del_duplicates(data,filename="tox21_10k_data_all.sdf"):
    name=filename.split('.')[0]
    props=list(data)
    props.remove('smiles')
    props.remove('smiles_no_salt')
    
    data_grouped=pd.DataFrame(data['smiles_no_salt'])
    data_grouped=data_grouped.drop_duplicates()
    
    for prop in props:
        grouped_prop=data.groupby(by='smiles_no_salt')[prop].apply(merging_rows)
        grouped_prop=pd.DataFrame(grouped_prop)
        data_grouped=data_grouped.join(grouped_prop,on='smiles_no_salt')
        
    data_grouped=data_grouped.dropna(subset=props,how='all')
    data_grouped.to_csv(name + '_no_salts.csv',index=False)
    return data_grouped

def del_wrong_smiles(data, filename="Wrong SMILES"):
    f=open(filename,'r')
    del_keys=f.read().split('\n')[:-1]
    f.close()
    for key in del_keys:
            data=data[data.smiles_no_salt != key]
    data.to_csv('tox21_10k_data_all_no_salts.csv',index=False)
    return data
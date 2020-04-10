#!/usr/bin/env python
from __future__ import print_function

from mpi4py import MPI

import sqlite3

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as rdmt
from func_timeout import FunctionTimedOut,func_set_timeout

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-a", "--amount_of_elem", dest="AMOUNT_OF_ELEM",
                    help="number of atoms to store", default = 10,type=int)
parser.add_argument("-n", "--num_confs",
                    dest="NUM_CONFS", default=100,
                    help="number of conformers to store",type=int)
parser.add_argument("-f", "--filename",
                    dest="FILENAME", default='./data/MultiTox0',
                    help="name of file to preprocess",type=str)


global args
args = parser.parse_args()

#number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS=args.NUM_CONFS

#amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM=args.AMOUNT_OF_ELEM


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

#creating array of elements in each chemical compound to use
def create_element_dict(data,amount=9,treshold=10, add_H=False):
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

#read dataset
data=pd.read_csv('./data/MultiTox.csv')
global elements
elements=create_element_dict(data,amount=AMOUNT_OF_ELEM)
data=pd.read_csv(args.FILENAME+'.csv')

if rank == 0:
    # Master process executes code below
    f=open('Wrong SMILES','w')
    conn = sqlite3.connect(args.FILENAME+'.db')
    c = conn.cursor()
    # Create table
    c.execute('DROP table IF EXISTS tox')
    c.execute('''CREATE TABLE tox
                 (smile, conformer,energy,type of atom,x,y,z)''')
    tasks = data['SMILES']
    task_index = 0
    num_workers = size - 1
    closed_workers = 0
    print("Master starting with %d workers" % num_workers)
    while closed_workers < num_workers:
        props = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=source, tag=tags.START)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            results = props
            smiles=results[0]
            for key in results[1].keys():
                if results[1][key]==None:
                    f.write(str(smiles)+'\n')
                    break
                energy=results[1][key]['energy']
                for atom in results[1][key]['coordinates'].keys():
                    for x,y,z in results[1][key]['coordinates'][atom]:
                # Insert a row of data
                        params = (smiles, key, energy, atom, x, y, z)
                        c.execute("INSERT INTO tox VALUES (?, ?, ?, ?, ?, ?, ?)", params)
            conn.commit()
        elif tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1
            conn.commit()
    conn.commit()
    conn.close()
    f.close()
    print("Master finishing")
else:
    # Worker processes execute code below
    @func_set_timeout(200)
    def f_confs(smile):
        molecule=Chem.MolFromSmiles(smile)
        molecule=Chem.AddHs(molecule)
        AllChem.EmbedMultipleConfs(molecule, numConfs=NUM_CONFS,maxAttempts=10000, pruneRmsThresh=-1)
        conformers=molecule.GetConformers()
        confIds=[conf.GetId() for conf in conformers]
        try:
            AllChem.MMFFSanitizeMolecule(molecule)
            mmff_props=AllChem.MMFFGetMoleculeProperties(molecule)
        except:
            return None
        def f_props(confId):
            try:
                ff=AllChem.MMFFGetMoleculeForceField(molecule,mmff_props,confId=confId)
                ff.Minimize(maxIts=1000)
                energy = ff.CalcEnergy()
                rdmt.CanonicalizeConformer(molecule.GetConformer(id=confId))
            except:
                return None
            xyz = [(lambda atom, pos: [atom.GetSymbol(),pos.x, pos.y, pos.z])(
            molecule.GetAtomWithIdx(i), molecule.GetConformer(id=confId).GetAtomPosition(i))
                         for i in range(molecule.GetNumAtoms())]
            xyz_dict={}
            for elem in xyz:
                atom=elem[0]
                x,y,z=(elem[1],elem[2],elem[3])
                if atom in elements.keys():
                    if atom in xyz_dict.keys():
                        xyz_dict[atom].append((x,y,z))
                    else:
                        xyz_dict[atom]=[(x,y,z)]
            xyz=xyz_dict
            for key in xyz_dict.keys():
                xyz_dict[key]=np.array(xyz_dict[key])
            return{'energy':energy,'coordinates':xyz}
        def f_possibility_norm(props, a=1):
            from math import exp
            try:
                energies=[x[0] for x in props]
            except:
                return props
            norm=max(energies)
            for i in range(len(props)):
                props[i][0]/=norm
                props[i][0]=exp(-a*props[i][0])
            energies=[x[0] for x in props]
            norm=sum(energies)
            for i in range(len(props)):
                props[i][0]/=norm
            return props
        props=list(map(f_props,confIds))
        props=f_possibility_norm(props)
        keys=confIds
        props_dict=dict(zip(keys,props))
        return props_dict 
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == tags.START:
            try:
                properties=f_confs(task)
            except FunctionTimedOut:
                print ("TIMEOUT!")
                properties={0:None}
            comm.send([task,properties], dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break
    comm.send(None, dest=0, tag=tags.EXIT)

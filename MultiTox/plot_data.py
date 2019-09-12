# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:41:14 2019

@author: Alice
"""
import numpy as np
import matplotlib.pyplot as plt
#input - file in format #epoch, #batch, loss or accuracy
#output - numpy array of mean value per epoch
def extract_data(filename):
    f=open(filename,'r')
    values=list(map(lambda x: x.split('\t'),f.read().split('\n')))[:-1]
    values=list(map(lambda x: list(map(lambda y: float(y),x)),values))
    values=np.array(values)
    epoch_prev=1.0
    epoch_input=[]
    inputs=[]
    for epoch,input in zip(values[:,0],values[:,2]):
        if epoch_prev==epoch:
            epoch_input.append(input)
        else:
            inputs.append(np.array(epoch_input).mean())
            epoch_input=[]
        epoch_prev=epoch
    f.close()
    return (np.array(inputs))

def plot_plain_graph(filename,title='',xlabel='',ylabel=''):
    inputs=extract_data(filename)
    plt.plot(inputs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
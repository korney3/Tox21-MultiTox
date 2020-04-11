# -*- coding: utf-8 -*-
#import all the necessary libraries


import numpy as np
import matplotlib.pyplot as plt


import torch


#number of conformers created for every molecule
global NUM_CONFS
NUM_CONFS=100

#amount of chemical elements taking into account
global AMOUNT_OF_ELEM
AMOUNT_OF_ELEM=6


#input:
#description - [atom name,x,y,z]
#elements - dictionary{element: number}
#dim - amount of grid points, dx - grid primitive cell size

#output:
#cube dimx*dimx*dimx*amount of zeros and ones

def creating_cube(elements, description, dimx=70, dx=0.5):
    dimelem=len(elements.keys())
    cube=torch.zeros((dimx,dimx,dimx,dimelem))
    for atom in description.keys():
        for x,y,z in description[atom]:
            xnum=int(dimx/2+x/dx)
            ynum=int(dimx/2+y/dx)
            znum=int(dimx/2+z/dx)
            cube[xnum][ynum][znum][elements[atom]]=1
    return cube

#creating string for dx pymol file

class VolToDx():
#     s = StringIO()
    str ="""object 1 class gridpositions counts {xlen} {ylen} {zlen}
origin    {OrX} {OrY} {OrZ}
delta  {dX} 0 0
delta  0 {dY} 0
delta  0 0 {dZ}
object 2 class gridconnections counts {xlen} {ylen} {zlen}
object 3 class array type double rank 0 items {length} data follows
{records}
    """
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        volume = kwargs['volume']
        xlen,ylen,zlen = volume.shape
        try:
            (OrX, OrY, OrZ) = kwargs['origin']
            (dX, dY, dZ) = kwargs['dsteps']
        except:
            raise NotImplementedError() #FixMe: !!!

        assert type(volume) == np.ndarray
        length = np.prod(volume.shape)
        records = ""
        flatten = volume.flatten()
        for i in range(1,length+1):
            records += str(flatten[i-1]) + " "
            if i % 3 == 0 and i!=length: records += "\n"
        #records = np.savetxt(self.s,volume.reshape(3,-1),delimiter=" ")
        list_of_variables = ["xlen","ylen","zlen","OrX", "OrY", "OrZ","dX","dY","dZ","length","records"]
        params = {}
        for k in list_of_variables: params[k] = locals()[k]
#        print(params)
        return self.str.format(**params)
    
#plot colored 2D projection of molecules
def molecule_visualization2D(minibatch2D):
    for batch in minibatch2D:
        plt.imshow(batch,interpolation='none',cmap='rainbow')
        plt.colorbar()
        plt.show()
        
def plot_visualization_input_as_parameter(model,elements,losses, epoch):
    import matplotlib.pyplot as plt
    inv_elems = {v: k for k, v in elements.items()}    
    data=model.x_input
    molecules = data.cpu().detach().sum(dim=0)
    fig = plt.figure(figsize=(10,15),constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    for i,grad in enumerate(molecules):
        f_ax = fig.add_subplot(gs[i//3,i%3])
        f_ax.imshow(grad.sum(dim=0))
        f_ax.set_title(inv_elems[i],fontsize=25)
    f_ax = fig.add_subplot(gs[-1, :])
    f_ax.plot(5*np.arange(0,len(losses),1),losses)
    f_ax.set_title('Loss function',fontsize=25)
    f_ax.set_xlabel('epochs',fontsize=25)
    f_ax.set_ylabel('loss',fontsize=25)
    fig.suptitle('Atom types in molecule',fontsize=25)
    
    plt.show()
    fig.savefig(os.path.join(LOG_PATH_SAVE,'images','img_'+str(epoch))+'.png',dpi=150,format='png')
    _ = plt.clf()
    
    
    
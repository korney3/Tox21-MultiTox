# Predicting the acute toxicity of organic molecules using 3D-convolutional neural networks

The project demonstrates the application of Neural Networks to prediction of the toxicity - complex bio-chemical property of an organic molecule. Due
to sparse voxel representation of molecules are not suitable for CNN inputs two types of transformations were used: with Waves and Gaussian
kernels. These approaches are being evaluated on Tox21 classification task and MultiTox regression datasets

The code aims to solve several tasks:

0.  Delete salts from inintial molecules set
1.  Produce database of molecule conformers based on its SMILES notation
2.  Train the model for predicting multilabel classification and regression tasks
3.  Create input backpropogation for evisualization of the quality of the model (in progress)

### Prerequisites

For running model train and evaluation based on existing datasets
```
numpy, torch, sklearn, tensorboardX 
```

For data preprocessing from SMILES to voxels
```
mpi4py, sqlite3, pandas, numpy, rdkit, func_timeout
```

### Installing

Database `tox21_conformers.db` made from Tox21 dataset can be downloaded [here](https://drive.google.com/drive/folders/1DmUrLd-ew3P_aLzL6hjonV-LW8mI4Zvq?usp=sharing) and should be placed to database folder
For reproducing database creation one can follow instruction below.

1.  Run `create_dataset` function from  `csv_dataframe_creating.py` for converting deleting salts from initial compounds. 	

    Input - `tox21_10k_data_all.sdf`	

    Output - `tox21_10k_data_all.csv`, data (pandas.DataFrame)	

2.  Then run `delete_duplicate` function from  `csv_dataframe_creating.py` for merging rows with the same compounds and different properties.	

    Input - data (pandas.DataFrame)	

    Output - `tox21_10k_data_all_no_salts.csv`	

3.  Run script `sql_database_creating.py` in command line as
    ```
    mpiexec -n N python sql_database_creating.py
    ```

    where N - amount of processes (Windows OS).	

    Input - `tox21_10k_data_all_no_salts.csv`	

    Output - `tox21_conformers.db`, `Wrong SMILES`	


## Running the training

```
python Neural_Net_sigma_train_optimization.py [flags]
```

Flags:
"-e", "--epochs", dest="EPOCHS_NUM",
                    help="number of train epochs",default = 100,type=int
"-p", "--patience",
                    dest="PATIENCE", default=25,
                    help="number of epochs to wait before early stopping",type=int
"-s", "--sigma",
                    dest="SIGMA", default=1.2,
                    help="sigma parameter",type=float
"-b", "--batch_size",
                    dest="BATCH_SIZE", default=32,
                    help="size of train and test batches",type=int
"-t", "--transformation",
                    dest="TRANSF", default='g',choices =['g', 'w'],
                    help="type of augmentstion - g (gauss) or w (waves)"
"-n", "--num_exp",
                    dest="NUM_EXP", default='',
                    help="number of current experiment"
"-v", "--voxel_dim",
                    dest="VOXEL_DIM", default=50,
                    help="size of produced voxel cube"
"-r", "--learn_rate",
                    dest="LEARN_RATE", default=1e-5,
                    help="learning rate for optimizer",type=float
"-a", "--sigma_train",
                    dest="SIGMA_TRAIN", default=False,
                    help="Regime of training sigma",type=bool
                    
                    
 The obtained model will be located in the `models_sigma_right\exp_N` directory

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lifelines
from lifelines.utils import concordance_index

dtype = torch.FloatTensor


# In[ ]:

def load_cv_indices_from_npz(base_path, fold_number):
    """
    Load cross-validation indices for a specific fold from .npz file.
    
    Parameters:
    -----------
    base_path : str
        Base path where the indices were saved
    fold_number : int
        Fold number (1-based indexing)
        
    Returns:
    --------
    tuple
        (train_indices, eval_indices) as numpy arrays
    """
    index_file_path = f"{base_path}cv_indices_fold_{fold_number}.npz"
    
    try:
        loaded_indices = np.load(index_file_path)
        train_idx = loaded_indices['train_idx']
        eval_idx = loaded_indices['eval_idx']
        
        print(f"Successfully loaded indices for fold {fold_number}")
        print(f"Training set size: {len(train_idx)}")
        print(f"Validation set size: {len(eval_idx)}")
        
        return train_idx, eval_idx


"""
Dataloaders

"""

def load_data(path, dtype):
    data = pd.read_csv(path)
    data.sort_values("ID",inplace = True)
    patient_id = data.loc[:, ["ID"]]
    patient_id.index = range(0, patient_id.shape[0], 1)
    x = data.iloc[:,1:].values
    X = torch.from_numpy(x).type(dtype)
    if torch.cuda.is_available():
        X = X.cuda()
    return(patient_id, X)

def load_pathway(path, dtype):
    '''Load a bi-adjacency matrix of pathways and genes, and then covert it to a Pytorch tensor.
    Input:
        path: path to input dataset (which is expected to be a csv file).
        dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
    Output:
        PATHWAY_MASK: a Pytorch tensor of the bi-adjacency matrix of pathways & genes.
    '''
    pathway_mask = pd.read_csv(path, index_col = 0).to_numpy()

    PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
    PATHWAY_MASK = torch.transpose(PATHWAY_MASK, 0, 1)
    ###if gpu is being used
    if torch.cuda.is_available():
        PATHWAY_MASK = PATHWAY_MASK.cuda()
    ###
    return(PATHWAY_MASK)


# In[ ]:


"""
Loss function for KL-PMVAE
"""

def bce_recon_loss(recon_x, x):
    batch_size = x.size(0)
    assert batch_size != 0
    bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum').div(batch_size)
    return bce_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    
    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(0)

    return total_kld, dimension_wise_kld, mean_kld


# In[ ]:


"""
Early stopping scheme when training LFSurv
"""

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, path="saved_model/sup_checkpoint.pt"):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epoch_count = 0
        self.best_epoch_num = 1
        self.early_stop = False
        self.max_acc = None
        self.delta = delta
        self.path = path

    def __call__(self, acc, model):
        if self.max_acc is None:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.max_acc = acc
            self.save_checkpoint(model)
        elif acc < self.max_acc + self.delta:
            self.epoch_count += 1
            self.counter += 1
            if self.counter % 20 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.max_acc = acc
            if self.verbose:
                print(f'Validation accuracy increased ({self.max_acc:.6f} --> {acc:.6f}).  Saving model ...')
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# In[ ]:


"""
Match patient IDs
"""

def get_match_id(id_1, id_2):
    match_id_cache = []
    for i in id_2["ID"]:
        match_id = id_1.index[id_1["ID"]==i].tolist()
        match_id_cache += match_id
    return(match_id_cache)


# In[ ]:


"""
Sample from the groups
"""

def splitExprandSample(condition, sampleSize, expr):
    split_expr = expr[condition].T
    split_expr = split_expr.sample(n=sampleSize, axis=1).T
    return split_expr


# In[ ]:





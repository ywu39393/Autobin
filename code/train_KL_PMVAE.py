#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import time
from code.KL_PMVAE import KL_PMVAE_2omics
from code.utils import bce_recon_loss, kl_divergence
from sklearn.model_selection import KFold

dtype = torch.FloatTensor


# In[ ]:


# def train_KL_PMVAE(train_x1, train_x2, eval_x1, eval_x2,
#                    z_dim, input_n1, input_n2, Pathway_Mask,
#                    Learning_Rate, L2, Cutting_Ratio, p1_epoch_num, num_cycles, dtype, save_model = False,
#                    path = "saved_model/unsup_checkpoint.pt"):
#     net = KL_PMVAE_2omics(z_dim, input_n1, input_n2, Pathway_Mask)
#     train_real = torch.cat((train_x1, train_x2), 1)
#     eval_real = torch.cat((eval_x1, eval_x2), 1)
    
#     if torch.cuda.is_available():
#         net.cuda()
#     opt = optim.Adam(net.parameters(), lr = Learning_Rate, weight_decay = L2)
#     cycle_iter = p1_epoch_num // num_cycles
#     start_time = time.time()
#     for epoch in range(p1_epoch_num):
#         tmp = float(epoch%cycle_iter)/cycle_iter
        
#         if tmp == 0:
#             beta = 0.1
#         elif tmp <= Cutting_Ratio:
#             beta = tmp/Cutting_Ratio
#         else:
#             beta = 1
        
#         net.train()
#         opt.zero_grad()
        
#         mean, logvar,  _, recon_x1, recon_x2 = net(train_x1, train_x2, s_dropout = True)
#         recon_x = torch.cat((recon_x1, recon_x2), 1)
#         recon_loss = bce_recon_loss(recon_x, train_real)
#         total_kld, _, _ = kl_divergence(mean, logvar)
#         loss_unsup = recon_loss + beta*total_kld
        
#         loss_unsup.backward()
#         opt.step()
        
#         if (epoch+1) % 100 == 0:
#             net.eval()
#             train_mean, train_logvar, _, train_recon1, train_recon2 = net(train_x1, train_x2, s_dropout = False)
#             train_recon = torch.cat((train_recon1, train_recon2), 1)
#             train_recon_loss = bce_recon_loss(train_recon, train_real)
#             train_total_kld, _, _ = kl_divergence(train_mean, train_logvar)
#             train_loss_unsup = train_recon_loss + beta*train_total_kld
            
#             net.eval()
#             eval_mean, eval_logvar, _, eval_recon1, eval_recon2 = net(eval_x1, eval_x2, s_dropout = False)
#             eval_recon = torch.cat((eval_recon1, eval_recon2), 1)
#             eval_recon_loss = bce_recon_loss(eval_recon, eval_real)
#             eval_total_kld, _, _ = kl_divergence(eval_mean, eval_logvar)
#             eval_loss_unsup = eval_recon_loss + beta*eval_total_kld
            
#             temp_epoch = epoch +1
#             print("Epoch: %s," %temp_epoch, "Loss in training: %s," %np.array(train_loss_unsup.detach().cpu().numpy()).round(4), "loss in validation: %s." %np.array(eval_loss_unsup.detach().cpu().numpy()).round(4))
    
#     if save_model:
#         print("Saving model...")
#         torch.save(net.state_dict(), path)
#         print("Model saved.")
    
#     print(np.array(time.time() - start_time).round(2))
#     return (train_mean, train_logvar, eval_mean, eval_logvar, train_loss_unsup, eval_loss_unsup)

def train_KL_PMVAE(train_x1, train_x2, eval_x1, eval_x2,
                   z_dim, input_n1, input_n2, Pathway_Mask,
                   Learning_Rate, L2, Cutting_Ratio, p1_epoch_num, num_cycles, dtype, save_model = False,
                   path = "saved_model/unsup_checkpoint.pt",
                   patience=30, min_delta=1e-4):
    """
    Added parameters:
    patience: Number of epochs to wait for improvement before stopping
    min_delta: Minimum change in validation loss to qualify as an improvement
    """
    net = KL_PMVAE_2omics(z_dim, input_n1, input_n2, Pathway_Mask)
    train_real = torch.cat((train_x1, train_x2), 1)
    eval_real = torch.cat((eval_x1, eval_x2), 1)
    
    if torch.cuda.is_available():
        net.cuda()
        
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
    cycle_iter = p1_epoch_num // num_cycles
    start_time = time.time()
    
    # Early stopping variables
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0
    
    # Store best outputs
    best_train_mean = None
    best_train_logvar = None
    best_eval_mean = None
    best_eval_logvar = None
    best_train_loss = None
    best_eval_loss = None
    
    for epoch in range(p1_epoch_num):
        tmp = float(epoch%cycle_iter)/cycle_iter
        
        if tmp == 0:
            beta = 0.1
        elif tmp <= Cutting_Ratio:
            beta = tmp/Cutting_Ratio
        else:
            beta = 1
        
        # Training step
        net.train()
        opt.zero_grad()
        
        mean, logvar, _, recon_x1, recon_x2 = net(train_x1, train_x2, s_dropout=True)
        recon_x = torch.cat((recon_x1, recon_x2), 1)
        recon_loss = bce_recon_loss(recon_x, train_real)
        total_kld, _, _ = kl_divergence(mean, logvar)
        loss_unsup = recon_loss + beta*total_kld
        
        loss_unsup.backward()
        opt.step()
        
        # Evaluation step (every 10 epochs)
        if (epoch+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                # Training set evaluation
                train_mean, train_logvar, _, train_recon1, train_recon2 = net(train_x1, train_x2, s_dropout=False)
                train_recon = torch.cat((train_recon1, train_recon2), 1)
                train_recon_loss = bce_recon_loss(train_recon, train_real)                
                train_total_kld, _, _ = kl_divergence(train_mean, train_logvar)
                train_loss_unsup = train_recon_loss + beta*train_total_kld
                
                # Validation set evaluation
                eval_mean, eval_logvar, _, eval_recon1, eval_recon2 = net(eval_x1, eval_x2, s_dropout=False)
                eval_recon = torch.cat((eval_recon1, eval_recon2), 1)
                eval_recon_loss = bce_recon_loss(eval_recon, eval_real)
                eval_total_kld, _, _ = kl_divergence(eval_mean, eval_logvar)
                eval_loss_unsup = eval_recon_loss + beta*eval_total_kld
                
                current_loss = eval_loss_unsup.item()
                
                # Early stopping check
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    best_model_state = net.state_dict().copy()
                    patience_counter = 0
                    best_epoch = epoch + 1
                    
                    # Store the outputs at best epoch
                    best_train_mean = train_mean.clone()
                    best_train_logvar = train_logvar.clone()
                    best_eval_mean = eval_mean.clone()
                    best_eval_logvar = eval_logvar.clone()
                    best_train_loss = train_loss_unsup.clone()
                    best_eval_loss = eval_loss_unsup.clone()
                else:
                    patience_counter += 1
                
                temp_epoch = epoch + 1
                print(f"Epoch: {temp_epoch}, Beta: {beta:.4f}, Loss in training: {train_loss_unsup.item():.4f}, "
                      f"Loss in validation: {eval_loss_unsup.item():.4f}")
                
                # Check if we should stop
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {temp_epoch}. "
                          f"Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
                    break
    
    if save_model:
        print("Saving best model...")
        torch.save(best_model_state if best_model_state is not None else net.state_dict(), path)
        print("Model saved.")
    
    training_time = np.array(time.time() - start_time).round(2)
    print(f"Training time: {training_time} seconds")
    print(f"Returning outputs from best epoch ({best_epoch}) with validation loss: {best_loss:.4f}")
    
    # No final evaluation, return the stored best outputs
    return (best_train_mean, best_train_logvar, best_eval_mean, best_eval_logvar, best_train_loss, best_eval_loss)
# In[ ]:





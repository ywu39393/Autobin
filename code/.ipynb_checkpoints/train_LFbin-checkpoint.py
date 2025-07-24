#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from code.LFbin import BinaryClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
dtype = torch.FloatTensor
from sklearn.metrics import accuracy_score
from code.utils import EarlyStopping, load_data
from sklearn.model_selection import StratifiedKFold
# In[ ]:


# def train_binary_classifier(train_x1, train_disease, train_trt, train_biomarker, train_yevent,
#                             eval_x1, eval_disease, eval_trt, eval_biomarker, eval_yevent,
#                             input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, Learning_Rate, L2, epoch_num, patience, dtype,
#                             path="saved_model/binary_classifier_checkpoint.pt", plot = True, num_of_biomarker = 25):
#     net = BinaryClassifier(input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, num_of_biomarker = num_of_biomarker )
    

#     early_stopping = EarlyStopping(patience=patience, verbose=False, path=path)

#     if torch.cuda.is_available():
#         net.cuda()
#     opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
    
#     #calculate the class weight
#     pos_weight = (train_yevent.shape[0]-train_yevent.sum())/train_yevent.sum()
#     pos_weight = pos_weight.clone().detach().to(dtype=torch.float32)

#     start_time = time.time()
#     train_losses = []
#     val_accuracies = []
#     val_aucs = []
#     val_f1s = []
    
#     for epoch in range(epoch_num):
#         net.train()
#         opt.zero_grad()

#         y_pred = net(train_x1, train_disease, train_trt, train_biomarker, s_dropout=True)
#         loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), train_yevent.squeeze(), pos_weight=pos_weight)

#         loss.backward()
#         opt.step()

#         train_losses.append(loss.item())

#         net.eval()
#         eval_y_pred = net(eval_x1, eval_disease, eval_trt, eval_biomarker, s_dropout=False)
#         eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
#         eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy())
#         eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
        
#         val_accuracies.append(eval_accuracy)
#         val_aucs.append(eval_auc)
#         val_f1s.append(eval_f1)

#         early_stopping(eval_f1, net)
#         if early_stopping.early_stop:
#             print("Early stopping, number of epochs: ", epoch)
#             print('Save model of Epoch {:d}'.format(early_stopping.best_epoch_num))
#             break
#         if (epoch+1) % 100 == 0:
#             net.eval()
#             train_y_pred = net(train_x1, train_disease, train_trt, train_biomarker, s_dropout=False)
#             train_f1 = f1_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
#             print("Training F1: %s," % train_f1, "validation F1: %s." % eval_f1)

#     print("Loading model, best epoch: %s." % early_stopping.best_epoch_num)
#     net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

#     net.eval()
#     train_y_pred = net(train_x1, train_disease, train_trt, train_biomarker, s_dropout=False)
#     train_accuracy = accuracy_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
#     train_auc = roc_auc_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy())
#     train_f1 = f1_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
    
#     net.eval()
#     eval_y_pred = net(eval_x1, eval_disease, eval_trt, eval_biomarker, s_dropout=False)
#     eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
#     eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy())
#     eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())

#     print("Final training Accuracy: %s," % train_accuracy, "final validation Accuracy: %s." % eval_accuracy)
#     print("Final training AUC: %s," % train_auc, "final validation AUC: %s." % eval_auc)    
#     print("Final training F1: %s," % train_f1, "final validation F1: %s." % eval_f1)
#     time_elapse = np.array(time.time() - start_time).round(2)
#     print("Total time elapse: %s." % time_elapse)
#     os.makedirs('plots', exist_ok=True)
#     if plot == True: 
#         # Plotting the metrics
#         epochs = range(1, len(train_losses) + 1)
#         plt.figure(figsize=(18, 6))

#         plt.subplot(1, 3, 1)
#         plt.plot(epochs, train_losses, 'b', label='Training loss')
#         plt.title('Training loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.subplot(1, 3, 2)
#         plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
#         plt.title('Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         plt.subplot(1, 3, 3)
#         plt.plot(epochs, val_aucs, 'g', label='Validation AUC')
#         plt.plot(epochs, val_f1s, 'm', label='Validation F1')
#         plt.title('Validation AUC and F1')
#         plt.xlabel('Epochs')
#         plt.ylabel('Score')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig('plots/training_metrics.png')
#         plt.close()
        
#         plt.figure()
#         conf_matrix = confusion_matrix(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#         plt.title('Confusion Matrix')
#         plt.xlabel('Predicted label')
#         plt.ylabel('True Label')
#         plt.savefig('plots/confusion matrix.png')
#         plt.close()

        
#         # ROC Curve
#         fpr, tpr, _ = roc_curve(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy())
#         roc_auc = auc(fpr, tpr)

#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic')
#         plt.legend(loc="lower right")
#         plt.savefig('plots/AUC plot.png')
#         plt.close()
    

#     return (train_y_pred, eval_y_pred, train_accuracy, eval_accuracy, train_auc, eval_auc, train_f1, eval_f1, early_stopping.best_epoch_num)

def train_binary_classifier_cv(x1, biomarker, yevent,
                            input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, 
                            Learning_Rate, L2, epoch_num, patience, dtype,
                            n_splits=5, base_path="saved_model/", plot=True, save_index=False, num_of_biomarker=25):
    """
    Train a binary classifier using cross-validation for imbalanced data.
    
    Parameters:
    -----------
    x1, biomarker : PyTorch tensors
        Input data features and target
    input_n, level_2_dim : int
        Model architecture parameters
    Dropout_Rate_1, Dropout_Rate_2 : float
        Dropout rates for the model
    Learning_Rate, L2 : float
        Optimizer parameters
    epoch_num : int
        Maximum number of epochs
    patience : int
        Patience for early stopping
    dtype : torch.dtype
        Data type for calculations
    n_splits : int
        Number of cross-validation folds
    base_path : str
        Base path for saving models
    plot : bool
        Whether to plot training metrics
    num_of_biomarker : int
        Number of biomarkers in the model
        
    Returns:
    --------
    tuple
        Results including predictions, metrics and best models
    """
    
    # Create directory for saved models and plots
    os.makedirs('plots', exist_ok=True)
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Convert tensors to numpy for splitting
    x1_np = x1.cpu().numpy()
    biomarker_np = biomarker.cpu().numpy()
    yevent_np = yevent.cpu().numpy()
    
    # Lists to store results from each fold
    fold_train_y_preds = []
    fold_eval_y_preds = []
    fold_train_accuracies = []
    fold_eval_accuracies = []
    fold_train_aucs = []
    fold_eval_aucs = []
    fold_train_f1s = []
    fold_eval_f1s = []
    fold_best_epoch_nums = []
    fold_indices = []

    for fold, (train_idx, eval_idx) in enumerate(skf.split(x1_np, yevent_np)):
        print(f"\nTraining on fold {fold+1}/{n_splits}")
        fold_indices.append((train_idx, eval_idx))
        
        # Save indices if requested
        if save_index:
            index_save_path = f"{base_path}_cv_indices_fold_{fold+1}.npz"
            np.savez(index_save_path, train_idx=train_idx, eval_idx=eval_idx)
            print(f"Saved CV indices for fold {fold+1} to: {index_save_path}")
    
        
        # Convert indices back to tensors
        train_x1 = torch.tensor(x1_np[train_idx], dtype=x1.dtype)
        train_biomarker = torch.tensor(biomarker_np[train_idx], dtype=biomarker.dtype)
        train_yevent = torch.tensor(yevent_np[train_idx], dtype=yevent.dtype)
        
        eval_x1 = torch.tensor(x1_np[eval_idx], dtype=x1.dtype)
        eval_biomarker = torch.tensor(biomarker_np[eval_idx], dtype=biomarker.dtype)
        eval_yevent = torch.tensor(yevent_np[eval_idx], dtype=yevent.dtype)
        # Create model
        net = BinaryClassifier(input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, num_of_biomarker=num_of_biomarker)
        
        # Path for the current fold
        path = f"{base_path}_binary_classifier_fold_{fold+1}.pt"
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=False, path=path)
        
        if torch.cuda.is_available():
            net.cuda()
            train_x1 = train_x1.cuda()
            train_biomarker = train_biomarker.cuda()
            train_yevent = train_yevent.cuda()
            eval_x1 = eval_x1.cuda()
            eval_biomarker = eval_biomarker.cuda()
            eval_yevent = eval_yevent.cuda()
            
        opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
        
        # Calculate class weight for imbalanced data
        pos_weight = (train_yevent.shape[0] - train_yevent.sum()) / train_yevent.sum()
        pos_weight = pos_weight.clone().detach().to(dtype=torch.float32)
        
        start_time = time.time()
        train_losses = []
        val_accuracies = []
        val_aucs = []
        val_f1s = []
        
        for epoch in range(epoch_num):
            net.train()
            opt.zero_grad()
            
            y_pred = net(train_x1, train_biomarker, s_dropout=True)
            loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), train_yevent.squeeze(), pos_weight=pos_weight)
            
            loss.backward()
            opt.step()
            
            train_losses.append(loss.item())
            
            net.eval()
            with torch.no_grad():
                eval_y_pred = net(eval_x1, eval_biomarker, s_dropout=False)
                eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
                eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy())
                eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
            
            val_accuracies.append(eval_accuracy)
            val_aucs.append(eval_auc)
            val_f1s.append(eval_f1)
            
            early_stopping(eval_f1, net)
            if early_stopping.early_stop:
                print(f"Early stopping, number of epochs: {epoch}")
                print(f'Save model of Epoch {early_stopping.best_epoch_num}')
                break
                
            if (epoch+1) % 10 == 0:
                net.eval()
                with torch.no_grad():
                    train_y_pred = net(train_x1, train_biomarker, s_dropout=False)
                    train_f1 = f1_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
                print(f"Fold {fold+1}, Epoch {epoch+1}: Training F1: {train_f1:.4f}, validation F1: {eval_f1:.4f}")
        
        print(f"Loading model, best epoch: {early_stopping.best_epoch_num}")
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
        net.eval()
        with torch.no_grad():
            train_y_pred = net(train_x1, train_biomarker, s_dropout=False)
            train_accuracy = accuracy_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
            train_auc = roc_auc_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy())
            train_f1 = f1_score(train_yevent.detach().cpu().numpy(), train_y_pred.detach().cpu().numpy().round())
            
            eval_y_pred = net(eval_x1, eval_biomarker, s_dropout=False)
            eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
            eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy())
            eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
        
        print(f"Fold {fold+1} - Final metrics:")
        print(f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {eval_accuracy:.4f}")
        print(f"Training AUC: {train_auc:.4f}, Validation AUC: {eval_auc:.4f}")
        print(f"Training F1: {train_f1:.4f}, Validation F1: {eval_f1:.4f}")
        
        time_elapse = np.array(time.time() - start_time).round(2)
        print(f"Time elapsed: {time_elapse} seconds")
        
        # Store results for this fold
        fold_train_y_preds.append(train_y_pred.detach().cpu().numpy())
        fold_eval_y_preds.append(eval_y_pred.detach().cpu().numpy())
        fold_train_accuracies.append(train_accuracy)
        fold_eval_accuracies.append(eval_accuracy)
        fold_train_aucs.append(train_auc)
        fold_eval_aucs.append(eval_auc)
        fold_train_f1s.append(train_f1)
        fold_eval_f1s.append(eval_f1)
        fold_best_epoch_nums.append(early_stopping.best_epoch_num)
        
        if plot:
            # Plotting metrics for this fold
            epochs = range(1, len(train_losses) + 1)
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, 'b', label='Training loss')
            plt.title(f'Fold {fold+1} - Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
            plt.title(f'Fold {fold+1} - Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(epochs, val_aucs, 'g', label='Validation AUC')
            plt.plot(epochs, val_f1s, 'm', label='Validation F1')
            plt.title(f'Fold {fold+1} - Validation AUC and F1')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/fold_{fold+1}_training_metrics.png')
            plt.close()
            
            # Confusion Matrix
            plt.figure()
            conf_matrix = confusion_matrix(eval_yevent.detach().cpu().numpy(), eval_y_pred.detach().cpu().numpy().round())
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Fold {fold+1} - Confusion Matrix')
            plt.xlabel('Predicted label')
            plt.ylabel('True Label')
            plt.savefig(f'plots/fold_{fold+1}_confusion_matrix.png')
            plt.close()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy())
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Fold {fold+1} - Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f'plots/fold_{fold+1}_AUC_plot.png')
            plt.close()
    
    
    # Aggregate results across all folds
    print("\nCross-validation results:")
    print(f"Mean Training Accuracy: {np.mean(fold_train_accuracies):.4f} ± {np.std(fold_train_accuracies):.4f}")
    print(f"Mean Validation Accuracy: {np.mean(fold_eval_accuracies):.4f} ± {np.std(fold_eval_accuracies):.4f}")
    print(f"Mean Training AUC: {np.mean(fold_train_aucs):.4f} ± {np.std(fold_train_aucs):.4f}")
    print(f"Mean Validation AUC: {np.mean(fold_eval_aucs):.4f} ± {np.std(fold_eval_aucs):.4f}")
    print(f"Mean Training F1: {np.mean(fold_train_f1s):.4f} ± {np.std(fold_train_f1s):.4f}")
    print(f"Mean Validation F1: {np.mean(fold_eval_f1s):.4f} ± {np.std(fold_eval_f1s):.4f}")
    
    # Find best fold based on validation F1 score
    best_fold_idx = np.argmax(fold_eval_f1s)
    print(f"\nBest model from fold {best_fold_idx+1} with validation F1: {fold_eval_f1s[best_fold_idx]:.4f}")
    
    if plot:
        # Plot cross-validation results
        plt.figure(figsize=(12, 8))
        
        metrics = ['Accuracy', 'AUC', 'F1']
        train_metrics = [fold_train_accuracies, fold_train_aucs, fold_train_f1s]
        val_metrics = [fold_eval_accuracies, fold_eval_aucs, fold_eval_f1s]
        
        for i, (metric, train_vals, val_vals) in enumerate(zip(metrics, train_metrics, val_metrics)):
            plt.subplot(1, 3, i+1)
            plt.bar(np.arange(n_splits) - 0.2, train_vals, 0.4, label='Training')
            plt.bar(np.arange(n_splits) + 0.2, val_vals, 0.4, label='Validation')
            plt.xlabel('Fold')
            plt.ylabel(metric)
            plt.title(f'Cross-validation {metric}')
            plt.xticks(np.arange(n_splits), [f'{j+1}' for j in range(n_splits)])
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/cross_validation_metrics.png')
        plt.close()
    
    
    # Return comprehensive results
    results = {
        'fold_train_y_preds': fold_train_y_preds,
        'fold_eval_y_preds': fold_eval_y_preds,
        'fold_train_accuracies': fold_train_accuracies,
        'fold_eval_accuracies': fold_eval_accuracies,
        'fold_train_aucs': fold_train_aucs,
        'fold_eval_aucs': fold_eval_aucs,
        'fold_train_f1s': fold_train_f1s,
        'fold_eval_f1s': fold_eval_f1s,
        'fold_best_epoch_nums': fold_best_epoch_nums,
        'best_fold': best_fold_idx,
        'fold_indices': fold_indices,
        'mean_train_accuracy': np.mean(fold_train_accuracies),
        'mean_eval_accuracy': np.mean(fold_eval_accuracies),
        'mean_train_auc': np.mean(fold_train_aucs),
        'mean_eval_auc': np.mean(fold_eval_aucs),
        'mean_train_f1': np.mean(fold_train_f1s),
        'mean_eval_f1': np.mean(fold_eval_f1s),
        'std_train_accuracy': np.std(fold_train_accuracies),
        'std_eval_accuracy': np.std(fold_eval_accuracies),
        'std_train_auc': np.std(fold_train_aucs),
        'std_eval_auc': np.std(fold_eval_aucs),
        'std_train_f1': np.std(fold_train_f1s),
        'std_eval_f1': np.std(fold_eval_f1s)
    }
    
    return results
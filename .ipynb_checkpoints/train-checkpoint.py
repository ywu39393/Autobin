import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold

# Custom modules
from code.KL_PMVAE import KL_PMVAE_2omics
from code.KL_PMVAE_shap import KL_PMVAE_2omics as KL_PMVAE_2omics_shap
from code.LFbin import BinaryClassifier
from code.utils import (
    load_pathway, 
    bce_recon_loss, 
    kl_divergence, 
    get_match_id, 
    load_data,
    EarlyStopping,
    splitExprandSample
)
from code.train_KL_PMVAE import train_KL_PMVAE
from code.train_LFbin import train_binary_classifier_cv
dtype = torch.FloatTensor



def main():
    
    #====================================================================================
    # Parser setting
    #====================================================================================
    parser = argparse.ArgumentParser(description="Autobin data Loader")
    
    parser.add_argument('--clinical_data', type=str, default='data/clinical_data.csv', help='Path to clinical data CSV')
    parser.add_argument('--gene_data', type=str, default='data/gene_data.csv', help='Path to gene data CSV')
    parser.add_argument('--other_data', type=str, default='data/other_data.csv', help='Path to other omics data CSV')
    parser.add_argument('--pathway_data', type=str, default='data/pathway.csv', help='Path to pathway mask CSV')
    parser.add_argument('--response_data', type=str, default='data/response_data.csv', help='Path to response variable CSV')
    parser.add_argument('--latent_dim_data', type=str, default='data/latent_dim.csv', help='Path to latent dimension data CSV')
    parser.add_argument('--pmvae_path', type=str, default='model/pmvae_model.pt', help='Path to PMVAE model')
    parser.add_argument('--lfbin_path', type=str, default='model/lfbin', help='Path to LFBin model directory')

    args = parser.parse_args()



    #====================================================================================
    # Data loading
    #====================================================================================

    patient_id, biomarker = load_data(args.clinical_data, dtype=dtype)
    _, x_gene = load_data(args.gene_data, dtype=dtype)
    _, x_olink = load_data(args.other_data, dtype=dtype)
    pathway_mask = load_pathway(args.pathway_data, dtype=dtype)

    model_path = args.pmvae_path
    latent_path = args.latent_dim_data


    #====================================================================================
    # PMVAE model training
    #====================================================================================

    # Setup parameters
    input_n1 = x_gene.shape[1]
    input_n2 = x_olink.shape[1]

    # Hyperparameters to tune
    # z_dim = [16, 32, 64, 128]  
    # EPOCH_NUM = [800, 1200, 1600, 2000]
    # NUM_CYCLES = [1, 2, 3, 4]  
    # Initial_Learning_Rate = [0.0005, 0.001, 0.005, 0.01, 0.05]  
    # L2_Lambda = [0.0001, 0.001, 0.005, 0.01] 
    # CUTTING_RATIO = [0.5, 0.6, 0.7, 0.8] 

    z_dim = [64]
    EPOCH_NUM = [1600]
    NUM_CYCLES = [2]
    Initial_Learning_Rate = [0.005]
    L2_Lambda = [0.005]
    CUTTING_RATIO = [0.7]

    # Initialize k-fold cross validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Track best parameters across all folds
    best_params = {
        'l2': 0.,
        'lr': 0.,
        'dim': 0,
        'epoch_num': 0,
        'num_cycle': 0,
        'cr': 0.,
        'loss': float('inf')
    }

    start_time = time.time()

    # Nested loops for hyperparameter tuning with cross-validation find the best dimension
    for l2 in L2_Lambda:
        for lr in Initial_Learning_Rate:
            for Z in z_dim:
                for Epoch_num in EPOCH_NUM:
                    for Num_cycles in NUM_CYCLES:
                        for cutting_ratio in CUTTING_RATIO:
                            fold_losses = []

                            # Cross-validation loop
                            for fold, (train_idx, val_idx) in enumerate(kf.split(x_gene)):
                                # Split data for this fold
                                x_train_gene = x_gene[train_idx]
                                x_train_olink = x_olink[train_idx]
                                x_val_gene = x_gene[val_idx]
                                x_val_olink = x_olink[val_idx]

                                # Train model for this fold
                                _, _, _, _, train_loss_unsup, eval_loss_unsup = train_KL_PMVAE(
                                    x_train_gene, x_train_olink,
                                    x_val_gene, x_val_olink,
                                    Z, input_n1, input_n2, pathway_mask,
                                    lr, l2, cutting_ratio, Epoch_num, Num_cycles, dtype,
                                    path=f"saved_models/unsup_checkpoint_fold_{fold}.pt"
                                )

                                fold_losses.append(eval_loss_unsup.item())

                                print(f"Fold {fold+1}/{k_folds}:")
                                print(f"L2: {l2}, LR: {lr}, z_dim: {Z}, "
                                      f"loss in validation: {eval_loss_unsup.item():.4f}, "
                                      f"loss in training: {train_loss_unsup.item():.4f}")

                            # Calculate average loss across folds
                            avg_loss = np.mean(fold_losses)

                            # Update best parameters if better
                            if avg_loss < best_params['loss']:
                                best_params.update({
                                    'l2': l2,
                                    'lr': lr,
                                    'dim': Z,
                                    'epoch_num': Epoch_num,
                                    'num_cycle': Num_cycles,
                                    'cr': cutting_ratio,
                                    'loss': avg_loss
                                })

                            print("\nAverage across folds:")
                            print(f"num_epoch: {Epoch_num}, num_cycles: {Num_cycles}, "
                                  f"cutting_ratio: {cutting_ratio}")
                            print(f"L2: {l2}, LR: {lr}, z_dim: {Z}, "
                                  f"average validation loss: {avg_loss:.4f}")

    end_time = time.time()

    # Print final results
    print("\nBest parameters found:")
    print(f"Optimal num epoch: {best_params['epoch_num']}, "
          f"optimal num cycles: {best_params['num_cycle']}, "
          f"optimal cutting ratio: {best_params['cr']}")
    print(f"Optimal L2: {best_params['l2']}, "
          f"optimal LR: {best_params['lr']}, "
          f"optimal z_dim: {best_params['dim']}")
    print(f"Best average validation loss: {best_params['loss']:.4f}")
    print(f"Total time: {end_time - start_time:.2f} seconds")


    opt_dim = best_params['dim']
    opt_lr = best_params['lr']
    opt_l2 = best_params['l2']
    opt_cr = best_params['cr']
    opt_epoch_num = best_params['epoch_num']
    opt_num_cycle = best_params['num_cycle']
    pmvae_dim = opt_dim

    # Total number of samples
    n_samples = x_gene.size(0)
    # Generate a random permutation of indices
    indices = torch.randperm(n_samples)
    # Compute split index
    split_idx = int(0.8 * n_samples)
    # Split into train and test indices
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    # Index into tensors using .numpy() to convert to numpy for proper pandas indexing
    train_indices_np = train_indices.numpy()
    test_indices_np = test_indices.numpy()
    # Index into tensors
    x_tune_gene = x_gene[train_indices]
    x_tune_olink = x_olink[train_indices]
    x_test_gene = x_gene[test_indices]
    x_test_olink = x_olink[test_indices]
    # Index into patient IDs using numpy indices for proper correspondence
    patient_id_tune = patient_id.iloc[train_indices_np]
    patient_id_test = patient_id.iloc[test_indices_np]

    # Train the model
    train_mean, train_logvar, test_mean, test_logvar, train_loss_unsup, test_loss_unsup = train_KL_PMVAE(
        x_tune_gene, x_tune_olink, x_test_gene, x_test_olink,
        opt_dim, input_n1, input_n2, pathway_mask,
        opt_lr, opt_l2, opt_cr, opt_epoch_num, opt_num_cycle, dtype, 
        save_model=True, path=model_path
    )

    tr_z = train_mean
    tes_z = test_mean

    print("Training sample size: %s," % tr_z.size()[0], "testing sample size: %s." % tes_z.size()[0])

    # Get column names for new data csv file
    z_count = np.array(list(range(1, tr_z.size()[1] + 1, 1))).astype('str')
    z_names = np.char.add('Z_', z_count).tolist()
    # Convert and save to csv file
    processed_tr = pd.DataFrame(tr_z.detach().cpu(), columns=z_names)
    processed_tr = processed_tr.astype(float)
    # Reset index of patient_id DataFrames to ensure proper concatenation
    patient_id_tune_reset = patient_id_tune.reset_index(drop=True)
    patient_id_test_reset = patient_id_test.reset_index(drop=True)
    # Concatenate patient IDs with processed data
    processed_tr = pd.concat([patient_id_tune_reset, processed_tr], axis=1)
    processed_tes = pd.DataFrame(tes_z.detach().cpu(), columns=z_names)
    processed_tes = processed_tes.astype(float)
    processed_tes = pd.concat([patient_id_test_reset, processed_tes], axis=1)
    # Concatenate training and testing data
    processed_all = pd.concat([processed_tr, processed_tes], axis=0, ignore_index=True)
    # Get the patient ID column name (assuming it's the first column)
    patient_id_col = processed_all.columns[0]
    # Reorder the combined data based on patient ID
    processed_all_sorted = processed_all.sort_values(by=patient_id_col).reset_index(drop=True)
    # Save the concatenated and sorted data
    processed_all_sorted.to_csv(latent_path, index=False)


    print('Concatenated and sorted data for PMVAE saved')



    #====================================================================================
    # Lfbin model training
    #====================================================================================

    _, yevent = load_data(args.response_data, dtype=dtype)
    _, x = load_data(args.latent_dim_data, dtype=dtype)
    num_of_clinical = biomarker.shape[1]
    lfbin_path = args.lfbin_path
    
    
    # Example of calling the function with your data 
    # input_n = opt_dim
    # level_2_dim = [8, 16, 32]
    # epoch_num = 1000
    # patience = 50
    # Initial_Learning_Rate = [0.2, 0.1, 0.05, 0.01, 0.0075, 0.005, 0.0025]
    # L2_Lambda = [0.01, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]
    # Dropout_rate_1 = [0.1, 0.3, 0.5, 0.7]
    # Dropout_rate_2 = [0.1, 0.3, 0.5, 0.7]
    input_n = opt_dim
    level_2_dim = [16]
    epoch_num = 1000
    patience = 50
    Initial_Learning_Rate = [0.2]
    L2_Lambda = [ 0.00075]
    Dropout_rate_1 = [0.1]
    Dropout_rate_2 = [0.1]

    best_epoch_num = 0

    # Initialize variables to track best F1 score
    opt_l2 = 0
    opt_lr = 0
    opt_dim = 0
    opt_dr1 = 0
    opt_dr2 = 0
    opt_va_f1 = float(0)  
    opt_tr_f1 = float(0)  
    best_epoch_num = 0

    # Grid search loop
    for l2 in L2_Lambda:
        for lr in Initial_Learning_Rate:
            for dim in level_2_dim:
                for dr1 in Dropout_rate_1:
                    for dr2 in Dropout_rate_2:
                        # Get results from cross-validation
                        results = train_binary_classifier_cv(x, biomarker, yevent, input_n, dim, dr1, dr2, 
                                                             lr, l2, epoch_num, patience, dtype, n_splits=5, 
                                                             base_path= lfbin_path, plot = False, save_index = False, num_of_biomarker = num_of_clinical)

                        # Extract metrics from results
                        mean_train_f1 = results['mean_train_f1']
                        mean_eval_f1 = results['mean_eval_f1']
                        best_epoch_num_tune = results['fold_best_epoch_nums'][results['best_fold']]

                        # Update best parameters if current F1 is better
                        if mean_eval_f1 > opt_va_f1:
                            opt_l2 = l2
                            opt_lr = lr
                            opt_dim = dim
                            opt_dr1 = dr1
                            opt_dr2 = dr2
                            opt_tr_f1 = mean_train_f1
                            opt_va_f1 = mean_eval_f1
                            best_epoch_num = best_epoch_num_tune

                        print("L2: %s," % l2, "LR: %s." % lr, "dim: %s," % dim, "dr1: %s," % dr1, "dr2: %s." % dr2)
                        print("Training F1: %s," % mean_train_f1, "Validation F1: %s." % mean_eval_f1)
                        print("F1 Std - Train: %s," % results['std_train_f1'], "Validation: %s." % results['std_eval_f1'])
                        print("---")

    # Print optimal parameters
    print("Optimal Parameters:")
    print("L2: %s, LR: %s, Dim: %s, DR1: %s, DR2: %s" % (opt_l2, opt_lr, opt_dim, opt_dr1, opt_dr2))
    print("Best Validation F1: %s, Training F1: %s" % (opt_va_f1, opt_tr_f1))
    print("Best Epoch: %s" % best_epoch_num)

    # # Train final model with optimal parameters
    results = train_binary_classifier_cv(x, biomarker, yevent, input_n, opt_dim, opt_dr1, opt_dr2, 
                                                         opt_lr, opt_l2, epoch_num, patience, dtype, n_splits=5, 
                                                         base_path= lfbin_path, plot = True, save_index = True, num_of_biomarker = num_of_clinical)

    best_fold = results['best_fold']

  

    #====================================================================================
    # DeepSHAP Lfbin model Inference
    #====================================================================================
    
    
    """
    Initialize the Lfbin network using the optimal set of hyperparameters
    """
    net = BinaryClassifier(input_n,opt_dim , opt_dr1, opt_dr2,  num_of_biomarker = num_of_clinical)
    """"""
    net.load_state_dict(torch.load(f"{lfbin_path}_binary_classifier_fold_{best_fold+1}.pt", map_location=torch.device('cpu')))
    ##make suere they are on the same device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    loaded_indices = np.load(f'{lfbin_path}_cv_indices_fold_{best_fold+1}.npz')
    train_idx = loaded_indices['train_idx']
    eval_idx = loaded_indices['eval_idx']

    x_tune = x[train_idx]
    x_test = x[eval_idx]
    biomarker_tune = biomarker[train_idx]
    biomarker_test = biomarker[eval_idx]
    yevent_tune = yevent[train_idx]
    yevent_test = yevent[eval_idx]
    patient_id_tune = patient_id.iloc[train_idx,:].reset_index(drop=True)
    patient_id_test = patient_id.iloc[eval_idx, :].reset_index(drop=True)

    x_tune = x_tune.to(device)
    biomarker_tune = biomarker_tune.to(device)
    net = net.to(device)
    """
    Get prognostic indicies for the tuning set and testing set patients, respectively.
    """
    net.eval()
    train_y_pred = net(x_tune, biomarker_tune, s_dropout = False)
    test_y_pred = net(x_test, biomarker_test, s_dropout = False)


    prognosis_index_train = train_y_pred
    prognosis_index_test = test_y_pred

    print("Prognosis_index_training size: %s," %prognosis_index_train.size()[0], "prognosis_index_testing size: %s." %prognosis_index_test.size()[0])

    processed_tr_pre = torch.cat((yevent_tune,biomarker_tune, x_tune, prognosis_index_train), 1)
    processed_tes_pre = torch.cat((yevent_test, biomarker_test, x_test, prognosis_index_test), 1)

    z_count = np.array(list(range(1, x_tune.size()[1]+1, 1))).astype('str')
    z_names = np.char.add('Z_', z_count).tolist()

    dat_biomarker = pd.read_csv(args.clinical_data)
    dat_latent = pd.read_csv(args.latent_dim_data) 
    column_name = ['Response']
    column_name.extend(dat_biomarker.columns[1:])
    column_name.extend(dat_latent.columns[1:])
    column_name.extend(["Prognosis_index"])

    processed_tr = pd.DataFrame(processed_tr_pre.detach().cpu().numpy(), columns = column_name)
    processed_tr = processed_tr.astype(float)
    processed_tr = pd.concat([patient_id_tune, processed_tr], axis=1)

    processed_tes = pd.DataFrame(processed_tes_pre.detach().cpu().numpy(), columns = column_name)
    processed_tes = processed_tes.astype(float)
    processed_tes = pd.concat([patient_id_test, processed_tes], axis=1)

    """
    Divide patients into high- and low-risk groups according to the median prognostic index among the tuning set patients
    """

    condition1_tr = processed_tr["Prognosis_index"]>processed_tr["Prognosis_index"].median()
    condition2_tr = processed_tr["Prognosis_index"]<=processed_tr["Prognosis_index"].median()

    condition1_tes = processed_tes["Prognosis_index"]>processed_tr["Prognosis_index"].median()
    condition2_tes = processed_tes["Prognosis_index"]<=processed_tr["Prognosis_index"].median()

    upper_group_tr = processed_tr[condition1_tr]
    lower_group_tr = processed_tr[condition2_tr]

    upper_group_tes = processed_tes[condition1_tes]
    lower_group_tes = processed_tes[condition2_tes]

    
    """
    Save the high- and low-risk group patient information
    """
    
    upper_group_tr.to_csv("data/higher_PI_train.csv", index=False)
    lower_group_tr.to_csv("data/lower_PI_train.csv", index=False)
    upper_group_tes.to_csv("data/higher_PI_test.csv", index=False)
    lower_group_tes.to_csv("data/lower_PI_test.csv", index=False)

    #define all the select variable name
    clinical_name = dat_biomarker.columns[1:].to_list()
    latent_name = dat_latent.columns[1:]
    
    
    """
    DeepSHAP
    
    """
    # load_data_deepshap: load data for LFbin model
    # input: data: data that stack all biomarkers(treatment, disease condition, clinical biomarkers and latent dimension)
    # dtype: make sure the data type during training was consistant.

    # make sure the output include different class of biomarkes where Z: latent
    def load_data_deepshap(data, dtype):
        z = data[latent_name].values.astype(np.float64)
        biomarker = data[clinical_name].values.astype(np.float64)

        Z = torch.from_numpy(z).type(dtype)
        BIOMARKER = torch.from_numpy(biomarker).type(dtype)

        return(Z, BIOMARKER)
    # used to load all features regarding the type
    def load_feature_deepshap(data):
        feature_data = data.drop(["ID", "response", "Prognosis_index"], axis = 1)

        return feature_data

    """
    These codes were adapted from the work of Withnell et al. https://academic.oup.com/bib/article/22/6/bbab315/6353242
    Please check their original implementation of DeepSHAP for more details at https://github.com/zhangxiaoyu11/XOmiVAE
    """
    # DeepSHAP algorithm, no modification needed
    class Explainer(object):
        """ This is the superclass of all explainers.
        """

        def shap_values(self, X):
            raise Exception("SHAP values not implemented for this explainer!")

        def attributions(self, X):
            return self.shap_values(X)

    class PyTorchDeepExplainer(Explainer):

        def __init__(self, model, data, outputNumber, dim, explainLatentSpace):

            data = list(load_data_deepshap(data, dtype))

            # check if we have multiple inputs
            self.multi_input = False
            if type(data) == list:
                self.multi_input = True
            else:
                data = [data]
            self.data = data
            self.layer = None
            self.input_handle = None
            self.interim = False
            self.interim_inputs_shape = None
            self.expected_value = None  # to keep the DeepExplainer base happy
            if type(model) == tuple:

                self.interim = True
                model, layer = model
                model = model.eval()
                self.layer = layer
                self.add_target_handle(self.layer)

                # if we are taking an interim layer, the 'data' is going to be the input
                # of the interim layer; we will capture this using a forward hook
                with torch.no_grad():
                    _ = model(*data)
                    interim_inputs = self.layer.target_input
                    if type(interim_inputs) is tuple:
                        # this should always be true, but just to be safe
                        self.interim_inputs_shape = [i.shape for i in interim_inputs]
                    else:
                        self.interim_inputs_shape = [interim_inputs.shape]
                self.target_handle.remove()
                del self.layer.target_input
            self.model = model.eval()
            self.multi_output = False
            self.num_outputs = 1
            with torch.no_grad():
                outputs = model(*data)

                #This is where specifies whether we want to explain the mean or z output
                if type(outputs) != list:
                    output = outputs
                else:
                    output = outputs[outputNumber]
                self.outputNum=outputNumber
                # Chosen dimension
                self.dim=None
                self.explainLatent = False
                if explainLatentSpace:
                    self.explainLatent=True
                    self.dimension=dim
                    output = output[:, dim]
                    output = output.reshape(output.shape[0], 1)
                # also get the device everything is running on
                self.device = output.device
                if output.shape[1] > 1:
                    self.multi_output = True
                    self.num_outputs = output.shape[1]
                self.expected_value = output.mean(0).cpu().numpy()

        def add_target_handle(self, layer):

            input_handle = layer.register_forward_hook(get_target_input)
            self.target_handle = input_handle

        def add_handles(self, model, forward_handle, backward_handle):
            """
            Add handles to all non-container layers in the model.
            Recursively for non-container layers
            """
            handles_list = []
            model_children = list(model.children())
            if model_children:
                for child in model_children:
                    handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
            else:  # leaves
                handles_list.append(model.register_forward_hook(forward_handle))
                handles_list.append(model.register_backward_hook(backward_handle))

            return handles_list

        def remove_attributes(self, model):
            """
            Removes the x and y attributes which were added by the forward handles
            Recursively searches for non-container layers
            """
            for child in model.children():
                if 'nn.modules.container' in str(type(child)):
                    self.remove_attributes(child)
                else:
                    try:
                        del child.x
                    except AttributeError:
                        pass
                    try:
                        del child.y
                    except AttributeError:
                        pass

        def gradient(self, idx, inputs):

            self.model.zero_grad()
            X = [x.requires_grad_() for x in inputs]

            output = self.model(*X)

            #Specify the output to change
            if type(output) != list:
                outputs = output
            else:
                outputs = output[self.outputNum]

            #Specify the dimension to explain
            if self.explainLatent==True:

                outputs = outputs[:, self.dimension]
                outputs = outputs.reshape(outputs.shape[0], 1)


            selected = [val for val in outputs[:, idx]]

            grads = []
            if self.interim:
                interim_inputs = self.layer.target_input
                for idx, input in enumerate(interim_inputs):
                    grad = torch.autograd.grad(selected, input,
                                               retain_graph=True if idx + 1 < len(interim_inputs) else None,
                                               allow_unused=True)[0]
                    if grad is not None:
                        grad = grad.cpu().numpy()
                    else:
                        grad = torch.zeros_like(interim_inputs[idx]).cpu().numpy()
                    grads.append(grad)
                del self.layer.target_input
                return grads, [i.detach().cpu().numpy() for i in interim_inputs]
            else:
                for idx, x in enumerate(X):
                    grad = torch.autograd.grad(selected, x,
                                               retain_graph=True if idx + 1 < len(X) else None,
                                               allow_unused=True)[0]
                    if grad is not None:
                        grad = grad.cpu().numpy()
                    else:
                        grad = torch.zeros_like(X[idx]).cpu().numpy()
                    grads.append(grad)
                return grads

        def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=False):

            # X ~ self.model_input
            # X_data ~ self.data

            X = list(load_data_deepshap(X, dtype))

            # check if we have multiple inputs
            if not self.multi_input:
                assert type(X) != list, "Expected a single tensor model input!"
                X = [X]
            else:
                assert type(X) == list, "Expected a list of model inputs!"


            X = [x.detach().to(self.device) for x in X]

            # if ranked output is given then this code is run and only the 'max' value given is explained
            if ranked_outputs is not None and self.multi_output:
                with torch.no_grad():
                    model_output_values = self.model(*X)
                    # Withnell's change to adjust for the additional outputs in VAE model
                    model_output_values = model_output_values[self.outputNum]

                # rank and determine the model outputs that we will explain

                if output_rank_order == "max":
                    _, model_output_ranks = torch.sort(model_output_values, descending=True)
                elif output_rank_order == "min":
                    _, model_output_ranks = torch.sort(model_output_values, descending=False)
                elif output_rank_order == "max_abs":
                    _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
                else:
                    assert False, "output_rank_order must be max, min, or max_abs!"

            else:
                # outputs and srray of 0s so we know we are explaining the first value
                model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                      torch.arange(0, self.num_outputs).int())

            # add the gradient handles

            handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
            if self.interim:
                self.add_target_handle(self.layer)

            # compute the attributions
            output_phis = []

            for i in range(model_output_ranks.shape[1]):

                phis = []
                #phis are shapLundberg values

                if self.interim:
                    for k in range(len(self.interim_inputs_shape)):
                        phis.append(np.zeros((X[0].shape[0], ) + self.interim_inputs_shape[k][1: ]))
                else:
                    for k in range(len(X)):
                        phis.append(np.zeros(X[k].shape))
                #shape is 5 as testing 5 samples
                for j in range(X[0].shape[0]):

                    # tile the inputs to line up with the background data samples
                    tiled_X = [X[l][j:j + 1].repeat(
                                       (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])) for l
                               in range(len(X))]
                    joint_x = [torch.cat((tiled_X[l], self.data[l]), dim=0) for l in range(len(X))]
                    # run attribution computation graph
                    feature_ind = model_output_ranks[j, i]
                    sample_phis = self.gradient(feature_ind, joint_x)
                    # assign the attributions to the right part of the output arrays
                    if self.interim:
                        sample_phis, output = sample_phis
                        x, data = [], []
                        for i in range(len(output)):
                            x_temp, data_temp = np.split(output[i], 2)
                            x.append(x_temp)
                            data.append(data_temp)
                        for l in range(len(self.interim_inputs_shape)):
                            phis[l][j] = (sample_phis[l][self.data[l].shape[0]:] * (x[l] - data[l])).mean(0)
                    else:
                        for l in range(len(X)):
                            phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]).to(self.device) * (X[l][j: j + 1] - self.data[l])).cpu().numpy().mean(0)
                output_phis.append(phis[0] if not self.multi_input else phis)


            # cleanup; remove all gradient handles
            for handle in handles:
                handle.remove()
            self.remove_attributes(self.model)
            if self.interim:
                self.target_handle.remove()

            if not self.multi_output:
                return output_phis[0]
            elif ranked_outputs is not None:
                # Withnell: returns a list... only want first value
                return output_phis, model_output_ranks
            else:
                return output_phis

    # Module hooks


    def deeplift_grad(module, grad_input, grad_output):
        """The backward hook which computes the deeplift
        gradient for an nn.Module
        """
        # first, get the module type
        module_type = module.__class__.__name__

        # first, check the module is supported
        if module_type in op_handler:

            if op_handler[module_type].__name__ not in ['passthrough', 'linear_1d']:
                return op_handler[module_type](module, grad_input, grad_output)
        else:
            print('Warning: unrecognized nn.Module: {}'.format(module_type))
            return grad_input


    def add_interim_values(module, input, output):
        """The forward hook used to save interim tensors, detached
        from the graph. Used to calculate the multipliers
        """
        try:
            del module.x
        except AttributeError:
            pass
        try:
            del module.y
        except AttributeError:
            pass
        module_type = module.__class__.__name__

        if module_type in op_handler:
            func_name = op_handler[module_type].__name__

            # First, check for cases where we don't need to save the x and y tensors
            if func_name == 'passthrough':
                pass
            else:
                # check only the 0th input varies
                for i in range(len(input)):
                    if i != 0 and type(output) is tuple:
                        assert input[i] == output[i], "Only the 0th input may vary!"
                # if a new method is added, it must be added here too. This ensures tensors
                # are only saved if necessary
                if func_name in ['maxpool', 'nonlinear_1d']:
                    # only save tensors if necessary
                    if type(input) is tuple:
                        setattr(module, 'x', torch.nn.Parameter(input[0].detach()))
                    else:
                        setattr(module, 'x', torch.nn.Parameter(input.detach()))
                    if type(output) is tuple:
                        setattr(module, 'y', torch.nn.Parameter(output[0].detach()))
                    else:
                        setattr(module, 'y', torch.nn.Parameter(output.detach()))
                if module_type in failure_case_modules:
                    input[0].register_hook(deeplift_tensor_grad)


    def get_target_input(module, input, output):
        """A forward hook which saves the tensor - attached to its graph.
        Used if we want to explain the interim outputs of a model
        """
        try:
            del module.target_input
        except AttributeError:
            pass
        setattr(module, 'target_input', input)

    # Withnell:
    # From the documentation: "The current implementation will not have the presented behavior for
    # complex Module that perform many operations. In some failure cases, grad_input and grad_output
    # will only contain the gradients for a subset of the inputs and outputs.
    # The tensor hook below handles such failure cases (currently, MaxPool1d). In such cases, the deeplift
    # grad should still be computed, and then appended to the complex_model_gradients list. The tensor hook
    # will then retrieve the proper gradient from this list.


    failure_case_modules = ['MaxPool1d']


    def deeplift_tensor_grad(grad):
        return_grad = complex_module_gradients[-1]
        del complex_module_gradients[-1]
        return return_grad


    complex_module_gradients = []


    def passthrough(module, grad_input, grad_output):
        """No change made to gradients"""
        return None


    def maxpool(module, grad_input, grad_output):
        pool_to_unpool = {
            'MaxPool1d': torch.nn.functional.max_unpool1d,
            'MaxPool2d': torch.nn.functional.max_unpool2d,
            'MaxPool3d': torch.nn.functional.max_unpool3d
        }
        pool_to_function = {
            'MaxPool1d': torch.nn.functional.max_pool1d,
            'MaxPool2d': torch.nn.functional.max_pool2d,
            'MaxPool3d': torch.nn.functional.max_pool3d
        }
        delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
        dup0 = [2] + [1 for i in delta_in.shape[1:]]
        # we also need to check if the output is a tuple
        y, ref_output = torch.chunk(module.y, 2)
        cross_max = torch.max(y, ref_output)
        diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

        # all of this just to unpool the outputs
        with torch.no_grad():
            _, indices = pool_to_function[module.__class__.__name__](
                module.x, module.kernel_size, module.stride, module.padding,
                module.dilation, module.ceil_mode, True)
            xmax_pos, rmax_pos = torch.chunk(pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs, indices, module.kernel_size, module.stride,
                module.padding, list(module.x.shape)), 2)
        org_input_shape = grad_input[0].shape  # for the maxpool 1d
        grad_input = [None for _ in grad_input]
        grad_input[0] = torch.where(torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in),
                               (xmax_pos + rmax_pos) / delta_in).repeat(dup0)
        if module.__class__.__name__ == 'MaxPool1d':
            complex_module_gradients.append(grad_input[0])
            # the grad input that is returned doesn't matter, since it will immediately be
            # be overridden by the grad in the complex_module_gradient
            grad_input[0] = torch.ones(org_input_shape)
        return tuple(grad_input)


    def linear_1d(module, grad_input, grad_output):
        """No change made to gradients."""
        return None


    def nonlinear_1d(module, grad_input, grad_output):
        delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2):]

        delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
        dup0 = [2] + [1 for i in delta_in.shape[1:]]
        # handles numerical instabilities where delta_in is very small by
        # just taking the gradient in those cases
        grads = [None for _ in grad_input]
        grads[0] = torch.where(torch.abs(delta_in.repeat(dup0)) < 1e-6, grad_input[0],
                               grad_output[0] * (delta_out / delta_in).repeat(dup0))
        return tuple(grads)


    op_handler = {}

    # passthrough ops, where we make no change to the gradient
    op_handler['Dropout3d'] = passthrough
    op_handler['Dropout2d'] = passthrough
    op_handler['Dropout'] = passthrough
    op_handler['AlphaDropout'] = passthrough

    op_handler['Conv1d'] = linear_1d
    op_handler['Conv2d'] = linear_1d
    op_handler['Conv3d'] = linear_1d
    op_handler['ConvTranspose1d'] = linear_1d
    op_handler['ConvTranspose2d'] = linear_1d
    op_handler['ConvTranspose3d'] = linear_1d
    op_handler['Linear'] = linear_1d
    op_handler['AvgPool1d'] = linear_1d
    op_handler['AvgPool2d'] = linear_1d
    op_handler['AvgPool3d'] = linear_1d
    op_handler['AdaptiveAvgPool1d'] = linear_1d
    op_handler['AdaptiveAvgPool2d'] = linear_1d
    op_handler['AdaptiveAvgPool3d'] = linear_1d
    op_handler['BatchNorm1d'] = linear_1d
    op_handler['BatchNorm2d'] = linear_1d
    op_handler['BatchNorm3d'] = linear_1d

    op_handler['LeakyReLU'] = nonlinear_1d
    op_handler['ReLU'] = nonlinear_1d
    op_handler['ELU'] = nonlinear_1d
    op_handler['Sigmoid'] = nonlinear_1d
    op_handler["Tanh"] = nonlinear_1d
    op_handler["Softplus"] = nonlinear_1d
    op_handler['Softmax'] = nonlinear_1d

    op_handler['MaxPool1d'] = maxpool
    op_handler['MaxPool2d'] = maxpool
    op_handler['MaxPool3d'] = maxpool


    # In[ ]:

    # getTopShapValues: use to get top SHAP value by define the numberOfTopFeatures and numberOfLatents
    def getTopShapValues(shap_vals, numberOfTopFeatures, numberOfLatents, path, absolute=True):
        multiple_input = False
        if type(shap_vals) == list:
            multiple_input = True
            shap_values = None
            for l in range(len(shap_vals)):
                if shap_values is not None:
                    shap_values = np.concatenate((shap_values, shap_vals[l]), axis=1)
                else:
                    shap_values = shap_vals[l]
            shap_vals = shap_values

        if absolute:
            vals = np.abs(shap_vals).mean(0)
        else:
            vals = shap_vals.mean(0)

        z_count = np.array(list(range(1, numberOfLatents+1, 1))).astype('str')
        z_names = np.char.add('Z_', z_count).tolist()

        # revise the clinical name if necessary
        if multiple_input:
            feature_names = z_names + clinical_name
        else:
            feature_names = z_names

        feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                          columns=['features', 'importance_vals'])
        feature_importance.sort_values(by=['importance_vals'], ascending=False, inplace=True)

        mostImp_shap_values = feature_importance.head(numberOfTopFeatures)
        print(mostImp_shap_values)

        feature_importance.to_csv(path + "/lfbin_imp.csv")
        """
        print(mostImp_shap_values)
        print("least importance absolute values")
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=True, inplace=True)
        leastImp_shap_values = feature_importance.head(numberOfTopFeatures)
        print(leastImp_shap_values)
        """
        return mostImp_shap_values


    # In[ ]:

    # revise the LFbin model and get he Deepshap result
    def SupShapExplainer(train_overall_df, path, dimension, input_n, hidden_dim, dr1, dr2, lfbin_path):
        #initialize LFbin network with the set of optimal hyperparameters found via grid search during training/validation process
        LFbin_model = BinaryClassifier(input_n, hidden_dim, dr1, dr2, num_of_biomarker = num_of_clinical)

        #load trained model
        LFbin_model.load_state_dict(torch.load(lfbin_path, map_location=torch.device('cpu')))

        condition1 = train_overall_df["Prognosis_index"]>train_overall_df["Prognosis_index"].median()
        condition2 = train_overall_df["Prognosis_index"]<=train_overall_df["Prognosis_index"].median()

        samplesize = processed_tr.shape[0]//2-5
        #select certain data
        upper_group_sample = splitExprandSample(condition=condition1, sampleSize=samplesize, expr=train_overall_df)
        lower_group_sample = splitExprandSample(condition=condition2, sampleSize=samplesize, expr=train_overall_df)

        e = PyTorchDeepExplainer(LFbin_model, lower_group_sample, outputNumber=0, dim=dimension, explainLatentSpace=False)
        print("calculating shap values")
        shap_values_obtained = e.shap_values(upper_group_sample)

        print("calculated shap values")
        # modify the number of latents and number of top features accordingly
        most_imp  = getTopShapValues(shap_vals=shap_values_obtained, numberOfTopFeatures=20, numberOfLatents=input_n, path=path, absolute=True)

        return most_imp

    most_important_features = SupShapExplainer(train_overall_df=processed_tr, path='deepshap', dimension=0, input_n = input_n, hidden_dim = opt_dim, dr1= opt_dr1, dr2 = opt_dr2, lfbin_path = f"{lfbin_path}_binary_classifier_fold_{best_fold+1}.pt" )

    #====================================================================================
    # DeepSHAP PMVAE model Inference
    #====================================================================================

    import subprocess

    # Run another Python file
    result = subprocess.run([
    'python', 
    'code/Deepshap_pmvae.py',
    '--index', f'{lfbin_path}_cv_indices_fold_{best_fold+1}.npz',
    '--gene_data', args.gene_data,
    '--other_data', args.other_data,
    '--pathway_data', args.pathway_data,
    '--pmvae_dim', str(pmvae_dim),
    '--pmvae_path', args.pmvae_path
    ], capture_output=True, text=True)

    # Check if it ran successfully
    if result.returncode == 0:
        print("Script executed successfully")
        print("Output:", result.stdout)
    else:
        print("Script failed")
        print("Error:", result.stderr)
    

    
    
if __name__ == '__main__':
    main()
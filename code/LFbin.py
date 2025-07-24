#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dtype = torch.FloatTensor


# In[ ]:


# class BinaryClassifier(nn.Module):
#     def __init__(self, input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, num_of_biomarker):
#         super(BinaryClassifier, self).__init__()
#         self.input_n = input_n
#         self.level_2_dim = level_2_dim
#         self.activation = nn.ReLU()

#         # Binary classification fc layers
#         self.bn_input = nn.BatchNorm1d(self.input_n)
#         self.fc1 = nn.Linear(self.input_n + num_of_biomarker+6, self.level_2_dim)
#         self.bn2 = nn.BatchNorm1d(self.level_2_dim)
#         self.fc2 = nn.Linear(self.level_2_dim, 1)
        
#         # Dropout
#         self.dropout_1 = nn.Dropout(Dropout_Rate_1)
#         self.dropout_2 = nn.Dropout(Dropout_Rate_2)

#     def forward(self, latent_features, c1, c2, c3, s_dropout=False):
#         if s_dropout:
#             latent_features = self.dropout_1(latent_features)
#         latent_features = self.bn_input(latent_features)
#         clinical_layer = torch.cat((latent_features, c1, c2, c3), 1)
#         hidden_layer = self.activation(self.fc1(clinical_layer))
#         if s_dropout:
#             hidden_layer = self.dropout_2(hidden_layer)
#         hidden_layer = self.bn2(hidden_layer)
#         y_pred = torch.sigmoid(self.fc2(hidden_layer))
        
#         return y_pred


class BinaryClassifier(nn.Module):
    def __init__(self, input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, num_of_biomarker):
        super(BinaryClassifier, self).__init__()
        self.input_n = input_n
        self.level_2_dim = level_2_dim
        self.tanh = nn.Tanh()

        # Binary classification fc layers
        self.bn_input = nn.BatchNorm1d(self.input_n)
        self.fc1 = nn.Linear(self.input_n + num_of_biomarker, self.level_2_dim)
        self.bn2 = nn.BatchNorm1d(self.level_2_dim)
        self.fc2 = nn.Linear(self.level_2_dim, 1)
        
        # Dropout
        self.dropout_1 = nn.Dropout(Dropout_Rate_1)
        self.dropout_2 = nn.Dropout(Dropout_Rate_2)

    def forward(self, latent_features, c1, s_dropout=False):
        if s_dropout:
            latent_features = self.dropout_1(latent_features)
        latent_features = self.bn_input(latent_features)
        clinical_layer = torch.cat((latent_features, c1), 1)
        hidden_layer = self.tanh(self.fc1(clinical_layer))
        if s_dropout:
            hidden_layer = self.dropout_2(hidden_layer)
        hidden_layer = self.bn2(hidden_layer)
        y_pred = torch.sigmoid(self.fc2(hidden_layer))
        
        return y_pred
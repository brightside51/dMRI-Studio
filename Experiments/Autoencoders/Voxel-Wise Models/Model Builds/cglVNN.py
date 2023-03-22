# Library Imports
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from torchsummary import summary

# Full Voxel-Wise CVAE Model Class Importing
sys.path.append('../Model Builds')
from Encoder import Encoder
from Decoder import Decoder

##############################################################################################
# ---------------------------------- Voxel-Wise cglVNN Build ---------------------------------
##############################################################################################

# Conditional Generative Linear Voxel Net Model Class
class cglVNN(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        num_labels: int = 5,                    # Number of Input Parameter Settings
        num_hidden: int = 2,                    # Number of NN Hidden Layers
        var_hidden: int =64                     # Deviance / Expansion of Hidden Layers
    ):

        # Class Variable Logging
        super(cglVNN, self).__init__()
        self.num_labels = num_labels
        self.num_hidden = num_hidden
        self.var_hidden = var_hidden
        var_neuron = int(np.floor(self.var_hidden / self.num_hidden))

        # Neural Network Architecture Definition
        net = []; in_neuron = 1 + (2 * self.num_labels); out_neuron = var_neuron
        for i in range(1, (2 * self.num_hidden) + 1):
            if i == self.num_hidden: var_neuron = -var_neuron
            elif i == 2 * self.num_hidden: out_neuron = 1
            #print(f"{in_neuron} -> {out_neuron}")
            net.append( nn.Sequential(
                            nn.Linear(in_neuron, out_neuron),
                            nn.LeakyReLU(inplace = True)))
            in_neuron = out_neuron; out_neuron = out_neuron + var_neuron
        self.net = nn.Sequential(*net)

    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        X_real: np.ndarray or torch.Tensor,     # 1D Voxel Input
        y_real: np.ndarray or torch.Tensor,     # 1D Image Label Input
        y_target: np.ndarray or torch.Tensor,   # 1D Target Label Input
    ):
    
        # Forward Propagation in Neural Network Architecture
        X_real = torch.Tensor(X_real)#.to(self.settings.device)
        y_real = torch.Tensor(y_real)#.to(self.settings.device)
        y_target = torch.Tensor(y_target)#.to(self.settings.device)
        return self.net(torch.cat([X_real, y_real, y_target], dim = 1))
    
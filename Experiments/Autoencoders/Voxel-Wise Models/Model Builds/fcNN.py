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
# ----------------------------------- Voxel-Wise fcNN Build ----------------------------------
##############################################################################################

# Linear Fully Connected Neural Network Model Class
class fcNN(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_params: int = 500,                   # Number of Input Parameter Settings
        out_params: int = 1344,                 # Number of Output Parameter Settings
        num_hidden: int = 2                     # Number of NN Hidden Layers
    ):

        # Class Variable Logging
        super(fcNN, self).__init__()
        self.in_params = in_params; self.out_params = out_params; self.num_hidden = num_hidden
        assert(self.out_params > self.in_params
               ),"ERROR: Neural Network wrongly built!"
        net = []; num_neuron = self.in_params

        # Neural Network Architecture Definition
        num_fc = int(np.floor((self.out_params - self.in_params) / (self.num_hidden + 1)))
        for i in range(self.num_hidden + 1):
            if i == self.num_hidden: num_fc += 1
            net.append( nn.Sequential(
                            nn.Linear(num_neuron, num_neuron + num_fc),
                            nn.LeakyReLU(inplace = True)))
            num_neuron += num_fc
        assert(num_neuron == self.out_params), "ERROR: Neural Network wrongly built!"
        self.net = nn.Sequential(*net)
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor,      # 1D Image Input
    ):
    
        # Forward Propagation in Neural Network Architecture
        #X = torch.Tensor(X).to(self.settings.device)
        return self.net(X)
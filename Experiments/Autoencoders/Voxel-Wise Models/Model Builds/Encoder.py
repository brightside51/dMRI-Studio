# Library Imports
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

##############################################################################################
# --------------------------------------- Encoder Build --------------------------------------
##############################################################################################

# Encoder Model Class
class Encoder(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser    # Model Settings & Parametrizations
        #num_labels: int = 5,                # Number of Labels contained in Dataset
        #num_channel: int = 64,              # Number of Output Channels for Encoder
        #num_layers: int = 3,                # Number of Main Convolutional Layers
        #latent_dim: int = 64                # Latent Space Dimensionality
    ):

        # Class Variable Logging
        super(Encoder, self).__init__()
        self.settings = settings

        # Encoder Downsampling Architecture Definition
        net = []; in_channel = 1 + settings.num_labels
        for i in reversed(range(settings.num_layers)):
            out_channel = int(settings.num_channel / (2 ** i))      # Current Main Layer's Output Channels
            #k = 2 * (i + 1)                                        # Kernel Size Value (6 is too high for Voxel-Wise CVAE)
            #print(f"{in_channel} -> {out_channel}")
            net.append(nn.Sequential(                               # Main Layer Block Repeatable Architecture
                nn.Conv1d(      in_channels = in_channel,
                                out_channels = out_channel,
                                kernel_size = 1, stride = 2, padding = 0),
                nn.LeakyReLU(   inplace = True)))
            in_channel = out_channel                                # Next Main Layer's Input Channels
        self.net = nn.Sequential(*net)
        
        # Mean and LogVariance Computation Linear Layers
        self.mean_layer = nn.Linear(    in_features = settings.num_channel,
                                        out_features = settings.latent_dim)
        self.logvar_layer = nn.Linear(  in_features = settings.num_channel,
                                        out_features = settings.latent_dim)

    # --------------------------------------------------------------------------------------------

    # Encoder Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor,      # 1D Image Input
        y: np.ndarray or torch.Tensor       # Image Labels Input
    ):

        # Net Input Handling
        X = torch.Tensor(X).view(-1, 1, 1).to(self.settings.device)                             # Input Features | [batch_size, 1,              1]
        y = torch.Tensor(y).view(-1, self.settings.num_labels, 1).to(self.settings.device)      # Input Labels   | [batch_size, num_labels,     1]
        input = torch.cat((X, y), dim = 1)                                                      # Encoder Output | [batch_size, 1+num_channel,  1]
        
        # Forward Propagation in Encoder Architecture
        output = self.net(input)                                                                # Encoder Output | [batch_size, num_channel,    1]
        z_mean = self.mean_layer(output.view(-1, self.settings.num_channel))                    # Latent Mean    | [batch_size, latent_dim]
        z_logvar = self.logvar_layer(output.view(-1, self.settings.num_channel))                # Latent LogVar  | [batch_size, latent_dim]

        # Display Settings for Experimental Model Version
        if self.settings.model_version == 0:
            print(f"Encoder Input  | {list(input.shape)}")
            print(f"Encoder Output | {list(output.shape)}")
            print(f"Latent Mean    | {list(z_mean.shape)}")
            print(f"Latent LogVar  | {list(z_logvar.shape)}\n")
        return z_mean, z_logvar

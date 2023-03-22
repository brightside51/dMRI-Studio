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
# --------------------------------------- Decoder Build --------------------------------------
##############################################################################################

# Decoder Model Class
class Decoder(nn.Module):

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
        super(Decoder, self).__init__()
        self.settings = settings

        # Decoder Upsampling Architecture Definition
        net = []; in_channel =  settings.num_channel
        self.linear = nn.Linear(in_features = settings.latent_dim + settings.num_labels,
                                out_features = settings.num_channel)
        for i in range(settings.num_layers):
            out_channel = int(settings.num_channel / (2 ** (i + 1)))    # Current Main Layer's Output Channels
            if i == settings.num_layers - 1: out_channel = 1            # Last Layer's Single Voxel Output Channel
            #k = 2 * (i + 1)                                            # Kernel Size Value (6 is too high for Voxel-Wise CVAE)
            #print(f"{in_channel} -> {out_channel}")
            net.append(nn.Sequential(                                   # Main Layer Block Repeatable Architecture
                nn.ConvTranspose1d( in_channels = in_channel,
                                    out_channels = out_channel,
                                    kernel_size = 1, stride = 2, padding = 0),
                nn.LeakyReLU(       inplace = True)))
            in_channel = out_channel                                    # Next Main Layer's Input Channels
        net.append(nn.Sigmoid()); self.net = nn.Sequential(*net)
            
    # --------------------------------------------------------------------------------------------

    # Decoder Application Function
    def forward(
        self,
        z: np.ndarray or torch.Tensor,      # Latent Dimension Representation
        y: np.ndarray or torch.Tensor       # Image Labels Input
    ):

        # Net Input Handling
        z = torch.Tensor(z).to(self.settings.device)        # Latent Representation | [batch_size, latent_dim]
        y = torch.Tensor(y).to(self.settings.device)        # Input Labels          | [batch_size, num_labels]
        input = torch.cat((z, y), dim = 1)                  # Decoder Input         | [batch_size, latent_dim + num_labels]

        # Forward Propagation in Decoder Architecture
        output = self.linear(input).view(-1, self.settings.num_channel, 1)
        output = self.net(output).view(-1, 1)

        # Display Settings for Experimental Model Version
        if self.settings.model_version == 0:
            print(f"Decoder Input  | {list(input.shape)}")
            print(f"Decoder Output | {list(output.shape)}\n")
        return output

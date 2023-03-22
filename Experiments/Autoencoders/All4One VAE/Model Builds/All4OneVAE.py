# Library Imports
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from torchsummary import summary

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from v3DMUDI import v3DMUDI

# Full 2D CcGAN Model Class Importing
sys.path.append('../Model Builds')
from Encoder import Encoder
from Decoder import Decoder

##############################################################################################
# ------------------------------------- All4One VAE Build ------------------------------------
##############################################################################################

# VAE Model Class
class All4One(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        latent_dim: int = 64,               # Latent Space Dimensionality
        num_channel: int = 1,               # Number of Channels in each Image (Default: 1 for 2D Dataset)
        img_shape: int = 96,                # Square Image Side Length (1/4th of pre-Convolution No. Channels)
        in_channel: int = 64,               # Number of Input Channels in ResNet Main Block Intermediate Layers' Blocks
        expansion: int = 1,                 # Expansion Factor for Stride Value in ResNet Main Block Intermediate Layers
        num_blocks: list = [2, 2, 2, 2]     # Number of Blocks in ResNet Main Block Intermediate Layers
    ):

        # Encoder & Decoder Construction
        super(All4One, self).__init__()
        self.encoder = Encoder(in_channel, num_channel, latent_dim, expansion, num_blocks)
        self.decoder = Decoder(num_channel, img_shape, latent_dim, expansion, num_blocks)

    # Latent Space Reparametrization
    @staticmethod
    def reparam(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    # All4One VAE Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor       # 3D Latent Representation Input
    ):

        # Forward Propagation in VAE Architecture
        mu, var = self.encoder(X)
        z = self.reparam(mu, var)
        X_fake = self.decoder(z)
        return mu, var, z, X_fake
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

# Full 2D CcGAN Model Class Importing
sys.path.append('../Model Builds')
from Encoder import Encoder
from Decoder import Decoder

##############################################################################################
# ----------------------------------- Voxel-Wise CVAE Build ----------------------------------
##############################################################################################

# Decoder Model Class
class VWCVAE(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser    # Model Settings & Parametrizations
    ):

        # Class Variable Logging
        super(VWCVAE, self).__init__()
        self.settings = settings
        self.encoder = Encoder(settings)      # Encoder Architecture Definition
        self.decoder = Decoder(settings)      # Decoder Architecture Definition
        
    # --------------------------------------------------------------------------------------------
    
    # Latent Dimension Reparametrization Gimmick Function
    def reparametrize(
        self,
        z_mean: torch.Tensor,           # Latent Dimension Mean
        z_logvar: torch.Tensor          # Latent Dimension Logarithmic Variance
    ):

        eps = torch.randn(  z_mean.size(0),                             # Epsilon Sampling from
                            z_mean.size(1)).to(self.settings.device)    # Standard Normal Distribution
        z = z_mean + (eps * torch.exp(z_logvar / 2.0))                  # Latent Dimension Representation
        return z

    # --------------------------------------------------------------------------------------------

    # Encoder Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor,      # 1D Image Input
        y: np.ndarray or torch.Tensor       # Image Labels Input
    ):

        # Forward Propagation in Encoder Architecture
        X = torch.Tensor(X)                             # Input Features        | [batch_size, 1]
        y = torch.Tensor(y).to(self.settings.device)    # Input Labels          | [batch_size, num_labels]
        z_mean, z_logvar = self.encoder(X, y)           # Encoder Output        | [batch_size, latent_dim]
        z = self.reparametrize(z_mean, z_logvar)        # Latent Representation | [batch_size, latent_dim]
        output = self.decoder(z, y)                     # Decoder Output        | [batch_size, 1]

        # Display Settings for Experimental Model Version
        if self.settings.model_version == 0:
            print(f"Model Features Input  | {list(X.shape)}")
            print(f"Model Labels Input    | {list(y.shape)}")
            print(f"Latent Representation | {list(z.shape)}")
            print(f"Model Output          | {list(output.shape)}")
        return output
# Library Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import alive_progress

# Functionality Import
from pathlib import Path
from tabulate import tabulate
from alive_progress import alive_bar
from torch.nn.utils import spectral_norm
from torchsummary import summary


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Linear Spectral Normalization Layer Class
class LinearSpectralNorm(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_channels: int = 64,
        bias: bool = True
    ):

        # Layer Architecture Construction
        super().__init__()
        self.num_channels = num_channels
        self.layer = spectral_norm(nn.Linear(in_channel, out_channel, bias))

    # Layer Application Function
    def forward(
        self,
        z: torch.Tensor
    ):

        # Layer Walkthrough
        out = self.layer(z)
        out = out.view(-1, self.num_channels * 16, 4, 4)
        return out

# --------------------------------------------------------------------------------------------

# 2D Convolutional Spectral Normalization Layer Class
class Conv2DSpectralNorm(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):

        # Layer Architecture Construction
        super().__init__()
        self.layer = spectral_norm( nn.Conv2d(in_channel, out_channel, kernel_size,
                                    stride, padding, dilation, groups, bias))

    # Layer Application Function
    def forward(
        self,
        z: torch.Tensor
    ):

        # Layer Walkthrough
        out = self.layer(z)
        return out

# --------------------------------------------------------------------------------------------

# Self Attention Layer Class
class SelfAttention(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int
    ):

        # Block Architecture Construction
        super().__init__()
        self.in_channel = in_channel
        self.conv2DSN_theta = Conv2DSpectralNorm(   in_channel, in_channel // 8,
                                                    kernel_size = 1, stride = 1, padding = 0)
        self.conv2DSN_phi = Conv2DSpectralNorm(     in_channel, in_channel // 8,
                                                    kernel_size = 1, stride = 1, padding = 0)
        self.conv2DSN_gamma = Conv2DSpectralNorm(   in_channel, in_channel // 2,
                                                    kernel_size = 1, stride = 1, padding = 0)
        self.conv2DSN_attent = Conv2DSpectralNorm(  in_channel // 2, in_channel,
                                                    kernel_size = 1, stride = 1, padding = 0)
        self.MaxPool2D = nn.MaxPool2d(2, stride = 2, padding = 0)
        self.SoftMax = nn.Softmax(dim = -1)
    
    # Layer Application Function
    def forward(
        self,
        X: torch.Tensor                 # Tensor containing Data
    ):

        # Theta Path
        theta = self.conv2DSN_theta(X)
        theta = theta.view(-1, X.shape[1] // 8, X.shape[2] * X.shape[3])

        # Phi Path
        phi = self.conv2DSN_phi(X)
        phi = self.MaxPool2D(phi)
        phi = phi.view(-1, X.shape[1] // 8, (X.shape[2] * X.shape[3]) // 4)

        # Gamma Path
        gamma = self.conv2DSN_gamma(X)
        gamma = self.MaxPool2D(gamma)
        gamma = gamma.view(-1, X.shape[1] // 2, (X.shape[2] * X.shape[3]) // 4)

        # Attention Map
        attent = torch.bmm(theta.permute(0, 2, 1), phi)
        attent = self.SoftMax(attent)
        attent = torch.bmm(gamma, attent.permute(0, 2, 1))
        attent = attent.view(-1, X.shape[1] // 2, X.shape[2], X.shape[3])
        attent = self.conv2DSN_attent(attent)

        return X + (nn.Parameter(torch.zeros(1)) * attent)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 2D Conditional Batch Normalization Layer Class
class c2DBatchNorm(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        num_feats: int,                 #
        dim_embedding: int = 128,       #
        momentum: float = 0.001         #
    ):

        # Layer Architecture Construction
        super().__init__()
        self.num_feats = num_feats
        self.layer = nn.BatchNorm2d(num_feats, momentum = momentum, affine = False)
        self.embedding = nn.Linear(dim_embedding, num_feats, bias = False)

    # Layer Application Function
    def forward(
        self,
        X: torch.Tensor,                # Tensor containing Data
        y: torch.Tensor                 # Tensor containing Labels
    ):

        # Layer Walkthrough
        out = self.layer(X)
        gamma = beta = self.embedding(y).view(-1, self.num_feats, 1, 1)
        out = out + (gamma * out) + beta
        return out

# --------------------------------------------------------------------------------------------

# Main / Repeatable Generator Block Construction Class
class GeneratorBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        dim_embedding: int = 128
    ):

        # Block Architecture Construction
        super().__init__()
        self.c2DBN_1 = c2DBatchNorm(            in_channel, dim_embedding)
        self.conv2DSN_1 = Conv2DSpectralNorm(   in_channel, out_channel,
                                                kernel_size = 3, stride = 1, padding = 1)
        self.c2DBN_2 = c2DBatchNorm(            out_channel, dim_embedding)
        self.conv2DSN_2 = Conv2DSpectralNorm(   out_channel, out_channel,
                                                kernel_size = 3, stride = 1, padding = 1)
        self.conv2DSN_X = Conv2DSpectralNorm(   in_channel, out_channel,
                                                kernel_size = 1, stride = 1, padding = 0)

    # Layer Application Function
    def forward(
        self,
        X: torch.Tensor,                # Tensor containing Data
        y: torch.Tensor                 # Tensor containing Labels
    ):

        # Block Walkthrough (Data Processing)
        X_0 = X                              # Copy of Original Data
        X_0 = F.interpolate(X_0, scale_factor = 2, mode = 'nearest')
        X_0 = self.conv2DSN_X(X_0)

        # Block Walkthrough
        X = self.c2DBN_1(X, y)
        X = nn.ReLU(inplace = True)(X)
        X = F.interpolate(X, scale_factor = 2, mode = 'nearest')
        X = self.conv2DSN_1(X)
        X = self.c2DBN_2(X, y)
        X = nn.ReLU(inplace = True)(X)
        X = self.conv2DSN_2(X)

        out = X + X_0
        return out

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Weight Initialization Function
def weightInit(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None: module.bias.data.fill_(0.)

# Generator Model Class
class Generator(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        dim_z: int = 256,               # Z Space Dimensionality
        dim_embedding: int = 128,       # Embedding Space Dimensionality
        num_channels: int = 64,         #
        momentum: float = 0.0001,       #
        eps: float = 1e-5,              #
    ):

        # Class Variable Logging
        super().__init__()
        self.dim_z = dim_z
        self.num_channels = num_channels

        # Generator Architecture
        #self.gen = nn.Sequential(); genList = dict()
        self.linearSN = LinearSpectralNorm( self.dim_z,  self.num_channels * 16 * 4 * 4,
                                            self.num_channels)                  # (4 x 4) Image
        self.genBlock1 = GeneratorBlock(    num_channels * 16,                  #    |
                                            num_channels * 16, dim_embedding)   # (8 x 8) Image
        self.genBlock2 = GeneratorBlock(    num_channels * 16,                  #    |
                                            num_channels * 8, dim_embedding)    # (16 x 16) Image
        self.genBlock3 = GeneratorBlock(    num_channels * 8,                   #     |
                                            num_channels * 4, dim_embedding)    # (32 x 32) Image
        self.selfAttention = SelfAttention( num_channels * 4)                   # (32 x 32) Image
        self.genBlock4 = GeneratorBlock(    num_channels * 4,                   #     |
                                            num_channels * 2, dim_embedding)    # (64 x 64) Image
        self.genBlock5 = GeneratorBlock(    num_channels * 2,                   #      |
                                            num_channels * 1, dim_embedding)    # (128 x 128) Image
        self.genPost = nn.Sequential(                                           # Image Post-Processing
            nn.BatchNorm2d(     num_channels, eps = eps,
                                momentum = momentum, affine = True),
            nn.ReLU(            inplace = True),
            Conv2DSpectralNorm( num_channels, 1,
                                kernel_size = 3, stride = 1, padding = 1),
            nn.Tanh())
        
        # Weight & Parameter Initialization
        self.visualizer()               # Parameter Numbers Visualization
        self.apply(weightInit)          # Weight Initialization Function     

    # Model Visualizer Function
    def visualizer(self):
        
        # Number of Total & Trainable Parameters
        num_total = sum(p.numel()   for p in self.parameters())     # Number of Total Parameters
        num_train = sum(p.numel()   for p in self.parameters()      # Number of Trainable Parameters 
                                    if p.requires_grad)             # (those that Require Autograd)
        print(f"Generator | Total Parameters: {num_total}\n          | Trainable Parameters: {num_train}")

    # Layer Application Function
    def forward(
        self,
        z: torch.Tensor,                # Tensor containing Z Space Data
        y: torch.Tensor                 # Tensor containing Labels
    ):

        # Block Walkthrough
        out = self.linearSN(z)                              # (4 x 4) Image
        out = out.view(-1, self.num_channels * 16, 4, 4)    # (4 x 4) Image
        out = self.genBlock1(out, y)                        # (8 x 8) Image
        out = self.genBlock2(out, y)                        # (16 x 16) Image
        out = self.genBlock3(out, y)                        # (32 x 32) Image
        out = self.selfAttention(out)                       # (32 x 32) Image
        out = self.genBlock4(out, y)                        # (64 x 64) Image
        out = self.genBlock5(out, y)                        # (128 x 128) Image
        out = self.genPost(out)                             # Image Post-Processing
        return out

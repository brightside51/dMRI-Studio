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

# Weight Initialization Function
def weightInit(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None: module.bias.data.fill_(0.)

# Main / Repeatable Discriminator Block Construction Class
class DiscriminatorBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool = True         # Boolean Control Variable for Downsampling
    ):

        # Block Architecture Construction
        super().__init__()
        if in_channel != out_channel: self.mismatch = True
        else: self.mismatch = False
        self.downsample = downsample
        self.conv2DSN_1 = Conv2DSpectralNorm(   in_channel, out_channel,
                                                kernel_size = 3, stride = 1, padding = 1)
        self.conv2DSN_2 = Conv2DSpectralNorm(   out_channel, out_channel,
                                                kernel_size = 3, stride = 1, padding = 1)
        self.downsampleLayer = nn.AvgPool2d(2)
        self.conv2DSN_X = Conv2DSpectralNorm(   in_channel, out_channel,
                                                kernel_size = 1, stride = 1, padding = 0)
        
    # Block Application Function
    def forward(
        self,
        X: torch.Tensor                 # Tensor containing Data
    ):

        # Block Walkthrough (Data Processing)
        X_0 = X.detach().clone()        # Copy of Original Data
        if self.downsample or self.mismatch:
            X_0 = self.conv2DSN_X(X_0)
            if self.downsample: X_0 = self.downsampleLayer(X_0)

        # Block Walkthrough
        X = nn.ReLU(inplace = True)(X)
        X = self.conv2DSN_1(X)
        X = nn.ReLU(inplace = True)(X)
        X = self.conv2DSN_2(X)
        if self.downsample: X = self.downsampleLayer(X)

        out = (X + X_0)
        return out

# --------------------------------------------------------------------------------------------

# Optimal Discriminator Block Construction Class
class OptimalBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):

        # Block Architecture Construction
        super().__init__()
        self.X0Block = nn.Sequential(
            nn.AvgPool2d(2),
            Conv2DSpectralNorm( in_channel, out_channel,
                                kernel_size = 1, stride = 1, padding = 0))
        self.XBlock = nn.Sequential(
            Conv2DSpectralNorm( in_channel, out_channel,
                                kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            Conv2DSpectralNorm( out_channel, out_channel,
                                kernel_size = 3, stride = 1, padding = 1),
            nn.AvgPool2d(2))

    # Block Application Function
    def forward(
        self,
        X: torch.Tensor                 # Tensor containing Data
    ):

        # Block Walkthrough
        X_0 = self.X0Block(X)
        X = self.XBlock(X)
        out = X + X_0
        return out

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

# Discriminator Model Class
class Discriminator(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        #num_labels: int = 7,           # Number of Labels provided in y
        dim_embedding: int = 128,       # Embedding Space Dimensionality
        num_channels: int = 64,         # Number of Neural Net Channels
    ):

        # Class Variable Logging
        super().__init__()
        self.num_channels = num_channels

        # Discriminator Architecture
        self.main = nn.Sequential(
            OptimalBlock(       1,                  num_channels),          # (128 x 128) Image
            DiscriminatorBlock( num_channels,       num_channels * 2),      # (64 x 64) Image
            SelfAttention(      num_channels * 2),                          # (64 x 64) Image
            DiscriminatorBlock( num_channels * 2,   num_channels * 4),      # (32 x 32) Image
            DiscriminatorBlock( num_channels * 4,   num_channels * 8),      # (16 x 16) Image
            DiscriminatorBlock( num_channels * 8,   num_channels * 16),     # (8 x 8) Image
            DiscriminatorBlock( num_channels * 16,   num_channels * 16,     # (4 x 4) Image
                                downsample = False),
            nn.ReLU(inplace = True))                                        # (4 x 4) Image
        self.linearSN = self.LinearSpectralNorm(num_channels * 16 * 4 * 4, 1)
        self.embedding = self.LinearSpectralNorm(dim_embedding, num_channels * 16 * 4 * 4, bias = False)

        # Weight & Parameter Initialization
        self.visualizer()               # Parameter Numbers Visualization
        self.apply(weightInit)          # Weight Initialization Function    
        nn.init.xavier_uniform_(self.embedding.weight) 

    # Model Visualizer Function
    def visualizer(self):
        
        # Number of Total & Trainable Parameters
        num_total = sum(p.numel()   for p in self.parameters())     # Number of Total Parameters
        num_train = sum(p.numel()   for p in self.parameters()      # Number of Trainable Parameters 
                                    if p.requires_grad)             # (those that Require Autograd)
        print(f"Discriminator | Total Parameters: {num_total}\n              | Trainable Parameters: {num_train}")

    # Linear Spectral Normalization Layer Function
    def LinearSpectralNorm(
        self,
        in_channel: int,
        out_channel: int,
        bias: bool = True
    ):
        return spectral_norm(nn.Linear(in_channel, out_channel, bias = bias))

    # Layer Application Function
    def forward(
        self,
        X: torch.Tensor,                # Tensor containing Data
        h: torch.Tensor                 # Tensor containing Embedded Labels
    ):

        # Block Walkthrough
        out = self.main(X)                              # (128 x 128) -> (4 x 4) Image
        out = out.view(-1, self.num_channels * 16 * 4 * 4)
        out1 = torch.squeeze(self.linearSN(out))        # 1st Output Section (Linear)
        h = self.embedding(h)                           # Embedded Labels
        out2 = torch.sum(torch.mul(out, h), dim = [1])  # 2nd Output Section (Projection)
        return (out1 + out2).unsqueeze(-1)
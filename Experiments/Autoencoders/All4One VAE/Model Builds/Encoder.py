# Library Imports
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

# Main Encoder Block Construction Class
class EncoderBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,        # Number of Encoder Block's Convolutional Input Channels
        stride: int = 1,
        padding: int = 1
    ):

        # Main Block's Downsampling Architecture Definition
        super(EncoderBlock, self).__init__()
        out_channel = in_channel * stride
        self.block = nn.Sequential(
            nn.Conv2d(      in_channel, out_channel, kernel_size = 3,
                            stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d( out_channel),
            nn.ReLU(),
            nn.Conv2d(      out_channel, out_channel, kernel_size = 3,
                            stride = 1, padding = padding, bias = False),
            nn.BatchNorm2d( out_channel))

        # Main Block's Shorcut Architecture Definition
        if stride == 1: self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(      in_channel, out_channel, kernel_size = 1,
                                stride = stride, bias = False),
                nn.BatchNorm2d( out_channel))

    # Main Block Application Function
    def forward(self, X):

        # Main Block Architecture Walkthrough
        out = self.block(X)
        out = out + self.shortcut(X)
        return F.relu(out)

# --------------------------------------------------------------------------------------------

# Encoder Model Class
class Encoder(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int = 64,               # Number of Input Channels in ResNet Main Block Intermediate Layers' Blocks
        num_channel: int = 1,               # Number of Channels in each Image (Default: 1 for 2D Dataset)
        latent_dim: int = 64,               # Latent Space Dimensionality
        expansion: int = 1,                 # Expansion Factor for Stride Value in ResNet Main Block Intermediate Layers
        num_blocks: list = [2, 2, 2, 2]     # Number of Blocks in ResNet Main Block Intermediate Layers
    ):

        # Class Variable Logging
        super(Encoder, self).__init__()
        assert(len(num_blocks) == 4), "Number of Blocks provided Not Supported!"
        self.in_channel = in_channel; self.num_channel = num_channel
        self.latent_dim = latent_dim; self.expansion = expansion

        # Encoder Downsampling Architecture Definition
        self.net = nn.Sequential(
            nn.Conv2d(      self.num_channel, 64, kernel_size = 3,
                            stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d( 64),
            nn.ReLU(),
            self.main_layer(out_channel = 64, num_blocks = num_blocks[0], stride = 1),
            self.main_layer(out_channel = 128, num_blocks = num_blocks[1], stride = 2),
            self.main_layer(out_channel = 256, num_blocks = num_blocks[2], stride = 2),
            self.main_layer(out_channel = 512, num_blocks = num_blocks[3], stride = 2))
        self.linear = nn.Linear(512, 2 * self.latent_dim)

    # Encoder Repeatable Layer Definition Function
    def main_layer(
        self,
        out_channel: int,
        num_blocks: int,
        stride: int = 2
    ):

        # Layer Architecture Creation
        stride = [stride] + [1] * (num_blocks - 1); layer = []
        for s in stride:
            layer.append(EncoderBlock(self.in_channel, stride = s))
            self.in_channel = out_channel
        return nn.Sequential(*layer)
    
    # Encoder Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor       # 3D Image Input
    ):

        # Forwad Propagation in Encoder Architecture
        X = torch.Tensor(X)
        out = self.net(X)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        mu = out[:, :self.latent_dim]
        var = out[:, self.latent_dim:]
        return mu, var
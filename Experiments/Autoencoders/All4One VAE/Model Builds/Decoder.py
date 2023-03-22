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
# --------------------------------------- Decoder Build --------------------------------------
##############################################################################################

# 2D Resizing Convolution
class ResizeConv2d(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        scale_factor: int,
        mode: str ='nearest'
    ):

        # 2D Resizing Convolution 
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor; self.mode = mode
        self.conv = nn.Conv2d(  in_channel, out_channel,
                                kernel_size, stride = 1, padding = 1)

    # Resizing Convolutional Block Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor       # 3D Image Input
    ):
        
        # Resizing Convolutional Block Architecture Walkthrough
        X = torch.Tensor(X)
        out = F.interpolate(X, scale_factor = self.scale_factor,
                            mode = self.mode)
        return self.conv(out)

# --------------------------------------------------------------------------------------------

# Main Decoder Block Construction Class
class DecoderBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel: int,        # Number of Decoder Block's Convolutional Input Channels
        stride: int = 1,
        padding: int = 1
    ):

        # Main Block's Upsampling Architecture Definition
        super(DecoderBlock, self).__init__()
        out_channel = int(in_channel / stride)
        self.block1 = nn.Sequential(
            nn.Conv2d(      in_channel, in_channel, kernel_size = 3,
                            stride = 1, padding = padding, bias = False),
            nn.BatchNorm2d( in_channel),
            nn.ReLU())

        # Main Block's Shorcut Architecture Definition
        if stride == 1:
            self.block2 = nn.Sequential(
                nn.Conv2d(      in_channel, out_channel, kernel_size = 3,
                                stride = 1, padding = padding, bias = False),
                nn.BatchNorm2d( out_channel))
            self.shortcut = nn.Sequential()
        else:
            self.block2 = nn.Sequential(
                ResizeConv2d(   in_channel, out_channel,
                                kernel_size = 3, scale_factor = stride),
                nn.BatchNorm2d( out_channel))
            self.shortcut = self.block2

    # Main Block Application Function
    def forward(
        self,
        X: np.ndarray or torch.Tensor       # 3D Image Input
    ):

        # Main Block Architecture Walkthrough
        out = self.block1(X)
        out = self.block2(out)
        out = out + self.shortcut(X)
        return F.relu(out)

# --------------------------------------------------------------------------------------------

# Decoder Model Class
class Decoder(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        num_channel: int = 1,               # Number of Channels in each Image (Default: 1 for 2D Dataset)
        img_shape: int = 96,                # Square Image Side Length (1/4th of pre-Convolution No. Channels)
        latent_dim: int = 64,               # Latent Space Dimensionality
        expansion: int = 1,                 # Expansion Factor for Stride Value in ResNet Main Block Intermediate Layers
        num_blocks: list = [2, 2, 2, 2]     # Number of Blocks in ResNet Main Block Intermediate Layers
    ):

        # Class Variable Logging
        super(Decoder, self).__init__()
        assert(len(num_blocks) == 4), "Number of Blocks provided Not Supported!"
        self.num_blocks = num_blocks
        self.in_channel = img_shape * (2 ** (len(self.num_blocks)))
        self.channel = self.in_channel; self.num_channel = num_channel
        self.img_shape = img_shape; self.latent_dim = latent_dim

        # Decoder Upsampling Architecture Definition
        self.linear = nn.Linear(self.latent_dim, self.in_channel)
        self.net = nn.Sequential(
            self.main_layer(out_channel = int(self.channel / 2), num_blocks = num_blocks[3], stride = 2),
            self.main_layer(out_channel = int(self.channel / 2), num_blocks = num_blocks[2], stride = 2),
            self.main_layer(out_channel = int(self.channel / 2), num_blocks = num_blocks[1], stride = 2),
            self.main_layer(out_channel = int(self.channel), num_blocks = num_blocks[0], stride = 1),
            nn.Sigmoid(),
            ResizeConv2d(   self.img_shape * 2, self.num_channel,
                            kernel_size = 3, scale_factor = img_shape / 64))

    # Decoder Repeatable Layer Definition Function
    def main_layer(
        self,
        out_channel: int,
        num_blocks: int,
        stride: int = 2
    ):

        # Layer Architecture Creation
        stride = [stride] + [1] * (num_blocks - 1); layer = []
        for s in reversed(stride):
            layer.append(DecoderBlock(self.channel, stride = s))
        self.channel = out_channel
        return nn.Sequential(*layer)

    # Decoder Application Function
    def forward(
        self,
        z: np.ndarray or torch.Tensor       # 3D Latent Representation Input
    ):

        # Forward Propagation in Decoder Architecture
        z = torch.Tensor(z)
        out = self.linear(z)
        out = out.view(z.size(0), self.in_channel, 1, 1)
        out = F.interpolate(out, scale_factor = 2 ** (len(self.num_blocks) - 1))
        out = self.net(out)
        out = out.view( z.size(0), self.num_channel,
                        self.img_shape, self.img_shape)
        return out

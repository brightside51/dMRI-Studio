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


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Main / Repeatable ResNet Block Construction Class
class SimpleBlock(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        in_channel:int,
        out_channel: int,
        stride: int = 1,
        expansion: int = 1
    ):

        # Main Block's Common Section Architecture
        super(SimpleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(  in_channel, out_channel, kernel_size = 3,
                        stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),          # WARNING: They use torch.functional's ReLU for some reason here!
            nn.Conv2d(  out_channel, out_channel, kernel_size = 3,
                        stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),          # WARNING: They use torch.functional's ReLU for some reason here!
            nn.Conv2d(  out_channel, expansion * out_channel,
                        kernel_size = 1, bias = False),
            nn.BatchNorm2d(expansion * out_channel)
        )

        # Main Block's Shortcut Section Architecture
        if stride != 1 or in_channel != expansion * out_channel:
             self.shortcut = nn.Sequential(
                nn.Conv2d(  in_channel, expansion * out_channel, kernel_size = 1,
                            stride = stride, bias = False),
                nn.BatchNorm2d(expansion * out_channel)
             )
        else: self.shortcut = nn.Sequential()

    # Main Block Application Function
    def forward(self, X):

        # Main Block Architecture Walkthrough
        out = self.block(X)
        out = out + self.shortcut(X)
        return F.relu(out)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# Label Embedding CNN Class (X -> h -> y)
class LabelEmbedding(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        num_blocks: list = [3, 4, 6, 3],    # Number of Blocks in ResNet Main Block Intermediate Layers
        in_channel: int = 64,               # Number of Input Channels in ResNet Main Block Intermediate Layers' Blocks
        expansion: int = 1,                 # Expansion Factor for Stride Value in ResNet Main Block Intermediate Layers
        dim_embedding: int = 128,           # Embedding Space Dimensionality (WIP)
        num_labels: int = 7                 # Number of Labels in Dataset
    ):

        # Class Variable Logging
        super(LabelEmbedding, self).__init__()
        assert(len(num_blocks) == 4), "Number of Blocks provided Not Supported!"
        self.num_blocks = num_blocks
        self.in_channel = in_channel
        self.expansion = expansion
        self.dim_embedding = dim_embedding

        # Main ResNet50 Architecture Construction
        self.mainNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.main_layer(64, self.num_blocks[0]),
            self.main_layer(128, self.num_blocks[1]),
            self.main_layer(256, self.num_blocks[2]),
            self.main_layer(512, self.num_blocks[3]),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    # --------------------------------------------------------------------------------------------

        # 1st SubNetwork for Label Embedding (X -> h)
        self.t1Net = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.dim_embedding),
            nn.BatchNorm1d(self.dim_embedding),
            nn.ReLU())

        # 2nd SubNetwork for Label Embedding (h -> y)
        self.t2Net = nn.Sequential(
            nn.Linear(self.dim_embedding, num_labels),
            nn.ReLU())

    # --------------------------------------------------------------------------------------------

    # ResNet Repeatable Layer Definition Function
    def main_layer(
        self,
        out_channel: int,
        num_blocks: int,
        stride: int = 2
    ):

        # Layer Architecture Creation
        stride = [stride] + [1] * (num_blocks - 1); layer = []
        for s in stride:
            layer.append(SimpleBlock(self.in_channel, out_channel, s, expansion = self.expansion))
            self.in_channel = out_channel * self.expansion
        return nn.Sequential(*layer)
            
    # --------------------------------------------------------------------------------------------

    # Label Embedding Application Function
    def forward(self,
        X: np.ndarray or torch.Tensor       # 3D Image Input
    ):

        # Forwad Propagation in CNN Architecture
        X = torch.Tensor(X)                 # Numpy Array to Tensor Conversion
        h = self.mainNet(X)                 # Main ResNet Application
        h = h.view(h.size(0), -1)           # Linearization of ResNet Features
        h = self.t1Net(h)                   # 1st SubNewtork Application (X -> h)
        y = self.t2Net(h)                   # 2nd SubNewtork Application (h -> y)
        return h, y


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Label Embedding SubNetwork Class (y -> h)
class t3Net(nn.Module):
    
    # Constructor / Initialization Function
    def __init__(self,
        dim_embedding: int = 128,           # Embedding Space Dimensionality
        num_blocks: int = 4,                # Number of Blocks in T3 SubNewtork
        num_labels: int = 7                 # Number of Labels in Dataset provided
    ):

        # Class Variable Logging
        super(t3Net, self).__init__()
        self.dim_embedding = dim_embedding      # h Variable Dimension
        self.num_blocks = num_blocks            # Number of Blocks in Embedding MLP
        self.num_labels = num_labels            # Number of Labels in Dataset provided
        self.mlp = nn.Sequential()              # Empty Embedding MLP Variable
        norm = True                             # Default Block Group Normalization

        # MLP Architecture Definition
        for i in range(num_blocks + 1):
            if i == 0: in_channel = self.num_labels         # 1st Block Entry Features
            else: in_channel = self.dim_embedding           # Intermediate Block Input Features
            if i == num_blocks - 1: norm = False            # No Group Normalization on Last Block
            self.mlp.add_module(f'Block #{i + 1}',
                                self.main_block(in_channel,
                                self.dim_embedding, norm = norm))

    # --------------------------------------------------------------------------------------------

    # MLP Repeatable Main Block Definition
    def main_block(self,
        in_channel: int,            # Input Features for Linear Layer
        out_channel: int,           # Output Features for Linear Layer
        norm: bool = True,          # Group Normalization Boolean Control Variable
    ):

        # Main Block Architecture Definition
        if norm:
            block = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.GroupNorm(8, out_channel), nn.ReLU())
        else:
            block = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.ReLU())
        return block
    
    # --------------------------------------------------------------------------------------------
    
    # Label Embedding Application Function
    def forward(self,
        y: pd.DataFrame       # 3D Image Input Labels
    ):

        # Label Embedding using MLP
        assert(y.ndim == 2), f"ERROR: Input Labels not Correctly Dimensioned!"
        return self.mlp(torch.Tensor(np.array(y)))          # h Embedded Labels Variable 

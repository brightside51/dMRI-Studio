# Library Imports
import sys
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import time
import alive_progress
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
torch.autograd.set_detect_anomaly(True)
from alive_progress import alive_bar

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from v3DMUDI import v3DMUDI

# Model Class Access
sys.path.append('../Model Builds')
from LabelEmbedding import LabelEmbedding, t3Net
from Generator import Generator
from Discriminator import Discriminator

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Label Embedding Models Training, Validation & Testing Script Class
class LitT12Net(pl.LightningModule):

    ##############################################################################################
    # ----------------------------------- Model & Dataset Setup ----------------------------------
    ##############################################################################################


    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    ):

        # Class Variable Logging
        super().__init__()
        self.settings = settings
        self.lr_decay_epochs = [80, 140]                # Epochs for Learning Rate Decay

        # Model Initialization
        self.model = LabelEmbedding(        in_channel = 64,
                                            expansion= settings.expansion,
                                            dim_embedding = settings.dim_embedding,
                                            num_labels = settings.num_labels)
        self.optimizer = torch.optim.SGD(   self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            momentum = 0.9,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.past_epochs = 0
        
        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/Embedding Net (V{self.settings.model_version}).pth")
        if self.settings.model_version != 0 and self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING T12 Net Model (Version {self.settings.model_version})")
            checkpoint = torch.load(self.model_filepath); self.checkpoint_fix = dict()
            for sd, sd_value in checkpoint.items():
                if sd == 'ModelSD' or sd == 'OptimizerSD':
                    self.checkpoint_fix[sd] = OrderedDict()
                    for key, value in checkpoint[sd].items():
                        if key[0:7] == 'module.':
                            self.checkpoint_fix[sd][key[7:]] = value
                        else: self.checkpoint_fix[sd][key] = value
                else: self.checkpoint_fix[sd] = sd_value
            
            # Application of Checkpoint's State Dictionary
            self.model.load_state_dict(self.checkpoint_fix['ModelSD'])
            self.optimizer.load_state_dict(self.checkpoint_fix['OptimizerSD'])
            self.past_epochs = self.checkpoint_fix['Training Epochs']
            torch.set_rng_state(self.checkpoint_fix['RNG State'])
            del checkpoint#, checkpoint_fix
        self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(  self.optimizer,     # Learning Rate Decay
                                                    gamma = self.settings.lr_decay)     # in Chosen Epochs
        self.model = nn.DataParallel(self.model.to(self.settings.device))

    # Optimizer Initialization Function
    def configure_optimizers(self): return super().configure_optimizers()

    # Foward Functionality
    def forward(self, X): return self.model(X)

    # --------------------------------------------------------------------------------------------
    
    # Train Set DataLoader Download
    def train_dataloader(self):
        TrainTrainLoader = v3DMUDI.loader(  Path(f"{self.settings.data_folderpath}"),
                                            dim = 2, version = self.settings.data_version,
                                            #set_ = 'Test', mode_ = 'Train')
                                            set_ = 'Train', mode_ = 'Train')
        self.train_batches = len(TrainTrainLoader)
        return TrainTrainLoader
    
    # Validation Set DataLoader Download
    def val_dataloader(self):
        TrainValLoader = v3DMUDI.loader(Path(f"{self.settings.data_folderpath}"),
                                        dim = 2, version = self.settings.data_version,
                                        #set_ = 'Test', mode_ = 'Train')
                                        set_ = 'Train', mode_ = 'Val')
        self.val_batches = len(TrainValLoader)
        return TrainValLoader

    # Test Set DataLoader Download
    def test_dataloader(self):
        TestValLoader = v3DMUDI.loader( Path(f"{self.settings.data_folderpath}"),
                                        dim = 2, version = self.settings.data_version,
                                        set_ = 'Test', mode_ = 'Val')
        self.test_batches = len(TestValLoader)
        return TestValLoader

    ##############################################################################################
    # ------------------------------------- Training Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Training
    def on_train_start(self):
        
        # Model Training Mode Setup
        self.model.train()
        self.automatic_optimization = False
        if self.settings.model_version == 0:
            self.current_epoch = self.settings.num_epochs - 1

        # TensorBoard Logger Initialization
        self.train_loss_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}/T12 Net', 'Training Performance')

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self): self.train_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Data Handling
        X_batch, ygt_batch = batch
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        ygt_batch = ygt_batch.type(torch.float).to(self.settings.device)

        # Forward Propagation
        h_batch, y_batch = self.model(X_batch)                  # T12 Model (X -> h -> y)
        loss = self.criterion(y_batch, ygt_batch)               # Loss Computation

        # Backwards Propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del X_batch, ygt_batch, h_batch, y_batch
        return loss

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx): self.train_loss = self.train_loss + loss['loss'].item()

    # Functionality called upon the End of a Training Epoch
    def training_epoch_end(self, outputs):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # TensorBoard Logger Update
        num_epochs = self.past_epochs + self.current_epoch
        self.train_loss = self.train_loss / self.train_batches
        self.train_loss_logger.experiment.add_scalar("Training Loss", self.train_loss, num_epochs)
        
        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)

    ##############################################################################################
    # ------------------------------------ Validation Script -------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Validation
    def on_validation_start(self):

        # Model Validation Mode Setup
        self.model.eval()

        # TensorBoard Logger Initialization
        if self.current_epoch == 0:
            self.val_loss_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}/T12 Net', 'Validation Performance')

    # Functionality called upon the Start of Training Epoch
    def on_validation_epoch_start(self): self.val_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def validation_step(self, batch, batch_idx):
        
        # Label Handling + Noise Addition
        X_batch, ygt_batch = batch
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        ygt_batch = ygt_batch.type(torch.float).to(self.settings.device)

        # Forward Pass
        h_batch, y_batch = self.model(X_batch)                  # T12 Model (X -> h -> y)
        loss = self.criterion(y_batch, ygt_batch)               # Loss Computation
        self.val_loss = self.val_loss + loss.item()
        del X_batch, ygt_batch, h_batch, y_batch
        return loss

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Batch Validation Step
    #def on_validation_batch_end(self, loss, batch, batch_idx, dataloader_idx): self.val_loss = self.val_loss + loss['loss'].item()

    # Functionality called upon the End of a Validation Epoch
    def validation_epoch_end(self, out):
        
        # TensorBoard Logger Update
        num_epochs = self.past_epochs + self.current_epoch
        self.val_loss = self.val_loss / self.val_batches
        self.val_loss_logger.experiment.add_scalar("Validation Loss", self.val_loss, num_epochs)

    ##############################################################################################
    # ------------------------------------- Testing Script ---------------------------------------
    ##############################################################################################

    
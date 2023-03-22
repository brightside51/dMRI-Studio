# Library Imports
import sys
import numpy as np
import argparse
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

# T3 Embedding Model Training, Validation & Testing Script Class
class LitT3Net(pl.LightningModule):

    ##############################################################################################
    # ----------------------------------- Model & Dataset Setup ----------------------------------
    ##############################################################################################

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
        embedNet: LabelEmbedding,                       # Trained T1 & T2 Conjoint Model
    ):

        # Class Variable Logging
        super().__init__()
        self.settings = settings
        self.embedNet = embedNet.model                  # Embedding Net (T1 + T2) Model
        self.t2Net = embedNet.model.module.t2Net        # T2 Model Contained in Label Embedding Variable
        self.lr_decay_epochs = [150, 250, 350]          # Epochs for Learning Rate Decay

        # Model Initialization
        self.model = t3Net( dim_embedding = self.settings.dim_embedding,
                            num_labels = self.settings.num_labels)
        self.optimizer = torch.optim.SGD(   self.model.parameters(),                  # T3 Model Optimizer
                                            #self.t2Net.parameters(),               # using T2's Parameters 
                                            lr = self.settings.base_lr,
                                            momentum = 0.9,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.past_epochs = 0
        
        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/T3 Net (V{self.settings.model_version}).pth")
        if self.settings.model_version != 0 and self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING T3 Net Model (Version {self.settings.model_version})")
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
        self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(  self.optimizer, # Learning Rate Decay
                                                    gamma = self.settings.lr_decay) # in Chosen Epochs
        self.model = nn.DataParallel(self.model.to(self.settings.device))

    # Optimizer Initialization Function
    def configure_optimizers(self): return super().configure_optimizers()

    # Foward Functionality
    def forward(self, y): return self.model(y)

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

        # TensorBoard Loggers Initialization
        self.train_loss_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}/T3 Net', 'Training Performance')
        self.val_loss_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}/T3 Net', 'Validation Performance')

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self):
        self.train_y_loss = 0
        self.train_h_loss = 0
        self.train_r_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):
        
        # Label Handling
        X_batch, ygt_batch = batch
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        ygt_batch = ygt_batch.type(torch.float).to(self.settings.device)

        # Label Noise Addition
        gamma_batch = np.random.normal(0, 0.2, ygt_batch.shape)
        gamma_batch = torch.from_numpy(gamma_batch).type(torch.float).to(self.settings.device)
        ygt_noise_batch = torch.clamp(ygt_batch + gamma_batch, 0.0, 1.0)

        # Forward Propagation
        h1_batch, y2_batch = self.embedNet(X_batch)                 # T1 (X -> h) + T2 Model (h -> y)
        h3_batch = self.model(ygt_noise_batch)                      # T3 Model (y -> h)
        yr_batch = self.t2Net(h3_batch)                             # T2 Model (h -> Reconstructed y)

        # Loss Computation
        h_loss = self.criterion(h3_batch, h1_batch)                 # Embedding Space Loss Computation
        y_loss = self.criterion(y2_batch, ygt_batch)                # Label Loss Computation
        r_loss = self.criterion(yr_batch, ygt_noise_batch)          # Label Reconstruction Loss Computation

        # Backward Propagation
        self.optimizer.zero_grad()
        y_loss.backward(retain_graph = True)
        h_loss.backward(retain_graph = True)
        r_loss.backward(retain_graph = True)
        self.optimizer.step()
        del X_batch, ygt_batch, gamma_batch, ygt_noise_batch, h1_batch, y2_batch, h3_batch, yr_batch
        return {"y_loss": y_loss, "h_loss": h_loss, 'r_loss': r_loss}

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx):
        self.train_y_loss = self.train_y_loss + loss['y_loss'].item()
        self.train_h_loss = self.train_h_loss + loss['h_loss'].item()
        self.train_r_loss = self.train_r_loss + loss['r_loss'].item()

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # TensorBoard Logger Update
        num_epochs = self.past_epochs + self.current_epoch
        self.train_y_loss = self.train_y_loss / self.train_batches
        self.train_h_loss = self.train_h_loss / self.train_batches
        self.train_r_loss = self.train_r_loss / self.train_batches
        self.train_loss_logger.experiment.add_scalar("Training Loss", self.train_y_loss, num_epochs)
        self.train_loss_logger.experiment.add_scalar("Embedding Space Discrepancy", self.train_h_loss, num_epochs)
        self.train_loss_logger.experiment.add_scalar("Label Reconstruction Loss", self.train_r_loss, num_epochs)
        
        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)

    ##############################################################################################
    # ------------------------------------ Validation Script -------------------------------------
    ##############################################################################################


# Library Imports
import sys
import io
import numpy as np
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import tensorflow as tf
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from v3DMUDI import v3DMUDI

# Full 2D CcGAN Model Class Importing
sys.path.append('../Model Builds')
from Encoder import Encoder
from Decoder import Decoder
from All4OneVAE import All4One

# --------------------------------------------------------------------------------------------

# VAE Model Training, Validation & Testing Script Class
class LitAll4One(pl.LightningModule):

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
        self.model = All4One(               latent_dim = settings.latent_dim,
                                            num_channel = settings.num_channel,
                                            img_shape = settings.img_shape,
                                            expansion = settings.expansion)
        self.optimizer = torch.optim.Adam(  self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            weight_decay = self.settings.weight_decay, )
        self.recon_criterion = nn.MSELoss(); self.past_epochs = 0

        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/All4One (V{self.settings.model_version}).pth")
        if self.settings.model_version != 0 and self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING All4One 2D VAE (Version {self.settings.model_version})")
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
            del checkpoint
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

        # TensorBoard Logger Initialization
        self.train_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Training Performance')

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_kl_loss = 0
        self.train_recon_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Data Handling
        X_batch, ygt_batch = batch
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        #ygt_batch = ygt_batch.type(torch.float).to(self.settings.device)

        # Forward Propagation & Loss Computation
        mu_batch, var_batch, z_batch, X_fake_batch = self.model(X_batch)
        kl_loss =  (-0.5 * (1 + var_batch - mu_batch ** 2 - torch.exp(var_batch)).sum(dim = 1)).mean(dim = 0)        
        recon_loss = self.recon_criterion(X_fake_batch, X_batch)
        loss = (recon_loss * self.settings.kl_alpha) + kl_loss

        # Backwards Propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del X_batch, ygt_batch, mu_batch, var_batch, z_batch, X_fake_batch
        return {'loss': loss, 'kl_loss': kl_loss, 'recon_loss': recon_loss}

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx):

        # Loss Values Update
        self.train_loss = self.train_loss + loss['loss'].item()
        self.train_kl_loss = self.train_kl_loss + loss['kl_loss'].item()
        self.train_recon_loss = self.train_recon_loss + loss['recon_loss'].item()

        # Last Batch's Example Original Image Saving
        if batch_idx == self.train_batches - 1:
            self.X_example, self.y_example = batch
            self.y_example = self.y_example[-1, :]
            self.X_example = self.X_example[-1, :, :, :]
            self.X_example = self.X_example.view(1, self.settings.num_channel,
                        self.settings.img_shape, self.settings.img_shape)

    # --------------------------------------------------------------------------------------------

    # Example Original vs Reconstructed Example Image Plotting Function
    def img_plot(self, num_epochs: int = 0):

        # Original vs Reconstruced Image Sampling
        mu_example, var_example, z_example, self.X_fake_example = self.model(self.X_example)
        self.X_example = self.X_example.view(self.settings.img_shape, self.settings.img_shape)
        self.X_fake_example = self.X_fake_example.view(self.settings.img_shape, self.settings.img_shape)
        del mu_example, var_example, z_example

        # Original Example Image Subplot
        figure = plt.figure(num_epochs, figsize = (60, 60))
        plt.tight_layout(); plt.title(f'Epoch #{num_epochs}')
        plt.subplot(2, 1, 1, title = 'Original')
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.imshow(self.X_example.detach().numpy(), cmap = plt.cm.binary)

        # Reconstructed Example Image Subplot
        plt.subplot(2, 1, 2, title = 'Reconstruction')
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(self.X_fake_example.detach().numpy(), cmap = plt.cm.binary)
        return figure

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Loss Value Updating (Batch Division)
        num_epochs = self.past_epochs + self.current_epoch
        self.train_loss = self.train_loss / self.train_batches
        self.train_kl_loss = self.train_kl_loss / self.train_batches
        self.train_recon_loss = self.train_recon_loss / self.train_batches

        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Example Image Plotting
        if num_epochs == 0:
            self.train_logger.experiment.add_graph(self.model, self.X_example)
        self.train_logger.experiment.add_scalar("Training Loss", self.train_loss, num_epochs)
        self.train_logger.experiment.add_scalar("Kullback Leibler Divergence", self.train_kl_loss, num_epochs)
        self.train_logger.experiment.add_scalar("Image Reconstruction Loss", self.train_recon_loss, num_epochs)
        plot = self.img_plot(num_epochs)
        self.train_logger.experiment.add_figure("Original vs Reconstruction", plot, num_epochs)

        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)

    ##############################################################################################
    # ------------------------------------ Validation Script -------------------------------------
    ##############################################################################################

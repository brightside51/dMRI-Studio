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
from nilearn.image import load_img
from nilearn.masking import unmask
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from v1DMUDI import v1DMUDI

# Full fcNN Model Class Importing
sys.path.append('../Model Builds')
from fcNN import fcNN

# --------------------------------------------------------------------------------------------

# fcNN Model Training, Validation & Testing Script Class
class LitfcNN(pl.LightningModule):

    ##############################################################################################
    # ---------------------------------------- Model Setup ---------------------------------------
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
        self.model = fcNN(                  in_params = self.settings.in_params,
                                            out_params = self.settings.out_params,
                                            num_hidden = self.settings.num_hidden)
        self.optimizer = torch.optim.Adam(  self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.recon_criterion = nn.MSELoss()
        self.past_epochs = 0

        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/fcNN (V{self.settings.model_version}).pth")
        #if self.settings.model_version != 0 and self.model_filepath.exists():
        if self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING Fully Connected Neural Network (Version {self.settings.model_version})")
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

    ##############################################################################################
    # -------------------------------------- Dataset Setup ---------------------------------------
    ##############################################################################################
    
    # Train Set DataLoader Download
    def train_dataloader(self):
        TrainLoader = v1DMUDI.loader(  Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            mode_ = 'Train')
        self.train_batches = len(TrainLoader)
        return TrainLoader

    # Test Set DataLoader Download
    def test_dataloader(self):
        TestLoader = v1DMUDI.loader( Path(f"{self.settings.data_folderpath}"),
                                        version = self.settings.data_version,
                                        mode_ = 'Test')
        self.test_batches = len(TestLoader)
        return TestLoader

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction
    def reconstruct(
        self,
        mode: str,
        #sel_patient: int = 0,           # Selected Example Patient
        sel_params: int = 0,            # Selected Parameter Setting Index
        sel_slice: int = 25             # Selected Slice
    ):
        
        # Dataset & Real 3D Image Access
        num_epochs = self.past_epochs + self.current_epoch
        if mode == 'Train':
            if self.current_epoch == 0:
                self.data = v1DMUDI.load(self.settings.data_folderpath, self.settings.data_version)
                self.pX_train, self.pMask_train = self.data.get_patient(0)
            pX = self.pX_train; pMask = self.pMask_train; sel_patient = 0
        elif mode == 'Test':
            if self.current_epoch == 0:
                self.data = v1DMUDI.load(self.settings.data_folderpath, self.settings.data_version)
                self.pX_test, self.pMask_test = self.data.get_patient(4)
            pX = self.pX_test; pMask = self.pMask_test; sel_patient = 4

        # --------------------------------------------------------------------------------------------

        # Fake 3D Image Generation
        pX_real = unmask(pX, pMask); pX_real = pX_real.get_fdata().T
        pX_fake = self.model(torch.Tensor(np.array(pX)[self.data.idx_train.astype(int), :]).T)
        pX_fake = unmask(pX_fake.detach().numpy().T, pMask); pX_fake = pX_fake.get_fdata().T
        assert(np.all(pX_real.shape == pX_fake.shape)), "ERROR: Unmasking went Wrong!"
        loss = self.recon_criterion(torch.Tensor(pX_fake), torch.Tensor(pX_real))

        # --------------------------------------------------------------------------------------------

        # Original Training Example Image Subplot
        figure = plt.figure(num_epochs, figsize = (60, 60))
        patient_id = self.data.patient_info['Patient'].iloc[sel_patient]
        plt.tight_layout(); plt.title(f'Epoch #{num_epochs} | Patient #{patient_id}'
                        + f' | Parameter Combo #{sel_params} | Slice #{sel_slice}')
        plt.subplot(2, 1, 1, title = 'Original Image')
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.imshow(pX_real[sel_params, sel_slice, :, :], cmap = plt.cm.binary)

        # Reconstructed Training Example Image Subplot
        plt.subplot(2, 1, 2, title = 'Reconstructed Image')
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(pX_fake[sel_params, sel_slice, :, :], cmap = plt.cm.binary)
        return figure, loss
    
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
    def on_train_epoch_start(self): self.train_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Data Handling
        X_train_batch, X_val_batch = batch
        X_train_batch = X_train_batch.type(torch.float).to(self.settings.device)
        X_batch = torch.cat((X_train_batch, X_val_batch), 1).type(torch.float).to(self.settings.device)

        # Forward Propagation & Loss Computation
        X_fake_batch = self.model(X_train_batch)
        loss = self.criterion(X_fake_batch, X_batch)

        # Backwards Propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del X_batch, X_train_batch, X_val_batch, X_fake_batch
        return loss

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx): self.train_loss = self.train_loss + loss['loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Reconstructed Images
        num_epochs = self.past_epochs + self.current_epoch
        self.train_loss = self.train_loss / self.train_batches
        train_plot, train_recon_loss = self.reconstruct(mode = 'Train',
                                                        sel_params = 0,
                                                        sel_slice = 25)
        
        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        if num_epochs == 0: self.train_logger.experiment.add_graph(self.model, torch.rand(1, self.settings.in_params))
        self.train_logger.experiment.add_scalar("Loss", self.train_loss, num_epochs)
        self.train_logger.experiment.add_scalar("Reconstruction Loss", train_recon_loss, num_epochs)
        self.train_logger.experiment.add_figure(f'Image Reconstruction', train_plot, num_epochs)

        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)

    ##############################################################################################
    # -------------------------------------- Testing Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Training
    def on_test_start(self):

        # TensorBoard Logger Initialization
        self.model.eval()
        self.test_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Testing Performance')
    
    # Functionality called upon the Start of Training Epoch
    def on_test_epoch_start(self): self.test_loss = 0

    # --------------------------------------------------------------------------------------------

    # Test Step / Batch Loop 
    def test_step(self, batch, batch_idx):

        # Data Handling
        X_train_batch, X_val_batch = batch
        X_train_batch = X_train_batch.type(torch.float).to(self.settings.device)
        X_batch = torch.cat((X_train_batch, X_val_batch), 1).type(torch.float).to(self.settings.device)

        # Forward Propagation & Loss Computation
        X_fake_batch = self.model(X_train_batch)
        loss = self.criterion(X_fake_batch, X_batch)
        del X_batch, X_train_batch, X_val_batch, X_fake_batch
        return loss

    # Functionality called upon the End of a Batch Test Step
    def on_test_batch_end(self, loss, batch, batch_idx, dataloader_idx):
        self.test_loss = self.test_loss + loss.item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Training Epoch
    def on_test_epoch_end(self):

        # Epoch Update for Losses & Reconstructed Images
        num_epochs = self.past_epochs + self.current_epoch
        self.test_loss = self.test_loss / self.test_batches
        test_plot, test_recon_loss = self.reconstruct(  mode = 'Test',
                                                        sel_params = 0,
                                                        sel_slice = 25)
        
        # TensorBoard Logger Update for Scalar Values & Image Visualizer
        self.test_logger.experiment.add_scalar("Loss", self.test_loss, num_epochs)
        self.test_logger.experiment.add_scalar("Reconstruction Loss", test_recon_loss, num_epochs)
        self.test_logger.experiment.add_figure(f'Image Reconstruction', test_plot, num_epochs)
        
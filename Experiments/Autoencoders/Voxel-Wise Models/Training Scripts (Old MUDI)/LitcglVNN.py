# Library Imports
import sys
import io
import numpy as np
import pandas as pd
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
from h1DMUDI import h1DMUDI

# Full fcNN Model Class Importing
sys.path.append('../Model Builds')
from cglVNN import cglVNN

# --------------------------------------------------------------------------------------------

# cglVNN Model Training, Validation & Testing Script Class
class LitcglVNN(pl.LightningModule):

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
        self.model = cglVNN(                num_labels = self.settings.num_labels,
                                            num_hidden = self.settings.num_hidden,
                                            var_hidden = self.settings.var_hidden)
        self.optimizer = torch.optim.Adam(  self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.past_epochs = 0

        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/cglVNN (V{self.settings.model_version}).pth")
        if self.settings.model_version != 0 and self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING Conditional Generative Linear Voxel Neural Network (Version {self.settings.model_version})")
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
    def forward(self, X_train, y_train, y_val): return self.model(X_train, y_train, y_val)

    ##############################################################################################
    # -------------------------------------- Dataset Setup ---------------------------------------
    ##############################################################################################
    
    # Train Set DataLoader Download
    def train_dataloader(self):
        TrainTrainLoader = h1DMUDI.loader(  Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Train')
        TrainValLoader = h1DMUDI.loader(    Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Val')
        assert(len(TrainTrainLoader) == len(TrainValLoader)
               ), f"ERROR: DataLoaders wrongly built!"
        self.train_batches = len(TrainTrainLoader)
        self.val_batches = len(TrainValLoader)
        return {'train': TrainTrainLoader, 'val': TrainValLoader}

    # Test Set DataLoader Download
    def test_dataloader(self):
        TestTrainLoader = h1DMUDI.loader(   Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Train')
        TestValLoader = h1DMUDI.loader(     Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Val')
        assert(len(TestTrainLoader) == len(TestValLoader)
               ), f"ERROR: DataLoaders wrongly built!"
        self.test_batches = len(TestTrainLoader)
        return {'train': TestTrainLoader, 'val': TestValLoader}

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction
    def reconstruct(
        self,
        t: int = 800,                   # Selected Parameter Setting Index
        sel_slice: int = 25,            # Selected Slice Index
    ):
        
        # Image to Image Mapping
        assert(0 <= t <= self.data.num_params), f"ERROR: Selected Target Parameter not Valid!"
        best_loss = torch.ones(1); worst_loss = torch.zeros(1)
        for p in range(0, self.data.num_params):
            
            # Fake 3D Image Set Generation
            pX_real = self.pX.iloc[p:p+1, :].T                          # Real | Input Image [No. Voxels]
            py_real = pd.concat(    [self.data.params.iloc[p:p+1, :]]   # Real | Input Parameter Combo
                                    * pX_real.shape[0],                 #      | Repeated for No. Voxels
                                    ignore_index = False)               #      | [No. Voxels, No. Labels]
            py_fake = pd.concat(    [self.data.params.iloc[t:t+1, :]]   # Fake | Output Parameter Combo
                                    * pX_real.shape[0],                 #      | Repeated for No. Voxels
                                    ignore_index = False)               #      | [No. Voxels, No. Labels]
            pX_fake = self.model(   torch.Tensor(np.array(pX_real)),    # Fake | Output Image [No. Voxels]
                                    torch.Tensor(np.array(py_real)),    #      | Patient Image for Param p ->
                                    torch.Tensor(np.array(py_fake)))    #      | -> Patient Image for Selected Param

            # Current Parameter Setting Loss Computation
            p_loss = self.criterion(torch.Tensor(pX_fake), torch.Tensor(np.array(self.pX.iloc[t:t+1, :].T)))
            if p_loss < best_loss: best_idx = p; best_loss = p_loss; pX_fake_best = pX_fake
            if p_loss > worst_loss: worst_idx = p; worst_loss = p_loss; pX_fake_worst = pX_fake

        """
        # Voxel to Voxel Mapping
        for p in range(self.pX.shape[1]):
            
            # Fake 3D Image Set Generation
            pX_real = self.pX.iloc[:, p:p+1].T                          # Real | Input Image [No. Params]
        """

        # Original & Fake Image Unmasking
        pX_real = unmask(self.pX.iloc[t:t+1, :], self.pMask); pX_real = pX_real.get_fdata().T
        pX_fake_best = unmask(pX_fake_best.detach().numpy().T, self.pMask); pX_fake_best = pX_fake_best.get_fdata().T
        pX_fake_worst = unmask(pX_fake_worst.detach().numpy().T, self.pMask); pX_fake_worst = pX_fake_worst.get_fdata().T
        assert(np.all(pX_real.shape == pX_fake_best.shape)), "ERROR: Unmasking went Wrong!"
        assert(np.all(pX_real.shape == pX_fake_worst.shape)), "ERROR: Unmasking went Wrong!"

        # --------------------------------------------------------------------------------------------

        # Training Example Original Image Subplot
        figure = plt.figure(self.num_epochs, figsize = (60, 60))
        patient_id = self.data.patient_info['Patient'].iloc[self.ex_patient]
        plt.tight_layout(); plt.title(f'Epoch #{self.num_epochs} | Patient #{patient_id}'
                        + f' | Parameter Combo #{t} | Slice #{sel_slice}')
        plt.subplot(1, 3, 1, title = f'Target Image (Parameter #{t})')
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.imshow(pX_real[0, sel_slice, :, :], cmap = plt.cm.binary)

        # Training Example Reconstructed Images Subplot
        plt.subplot(1, 3, 2, title = f'Best Reconstruction (Parameter #{best_idx} -> {t})')
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(pX_fake_best[0, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(1, 3, 3, title = f'Worst Reconstruction (Parameter #{worst_idx} -> {t})')
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(pX_fake_worst[0, sel_slice, :, :], cmap = plt.cm.binary)
        return figure, best_loss, worst_loss

    ##############################################################################################
    # ------------------------------------- Training Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Training
    def on_train_start(self):
        
        # Model Training Mode Setup
        self.model.train(); self.skip = 0
        self.automatic_optimization = False
        self.train_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Training Performance')
        self.train_logger.experiment.add_graph(self.model, (torch.rand(1, 1), 
                                                            torch.rand(1, self.settings.num_labels),
                                                            torch.rand(1, self.settings.num_labels)))

        # Training & Validation Set Example Patient Dataset 
        self.data = h1DMUDI.load(self.settings.data_folderpath, self.settings.data_version)
        self.ex_patient = 0; self.pX, self.pMask = self.data.get_patient(self.ex_patient)

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self):
        self.num_epochs = self.past_epochs + self.current_epoch
        self.tt_loss = 0; self.tv_loss = 0
        #self.vt_loss = 0; self.vv_loss = 0
        self.train_loss = 0
        #self.val_loss = 0

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Full Training & Validation Batch Input Data Handling
        X_train_batch, y_train_batch = batch['train']; X_val_batch, y_val_batch = batch['val']
        X_train_batch = X_train_batch.type(torch.float).to(self.settings.device)
        y_train_batch = y_train_batch.type(torch.float).to(self.settings.device)
        X_val_batch = X_val_batch.type(torch.float).to(self.settings.device)
        y_val_batch = y_val_batch.type(torch.float).to(self.settings.device)
        assert(X_train_batch.shape[0] % self.data.train_params == 0
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"
        assert(X_val_batch.shape[0] % self.data.val_params == 0
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"

        # Full Batch Label Data Handling
        y_batch = torch.cat((   y_train_batch[0 : self.data.train_params, :],
                                y_val_batch[0 : self.data.val_params, :]), dim = 0)
        y_batch = y_batch.type(torch.float).to(self.settings.device)
        assert(y_batch.shape[0] == self.data.num_params
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"
        tt_loss = torch.zeros(1); tv_loss = torch.zeros(1)
        #vt_loss = torch.zeros(1); vv_loss = torch.zeros(1)

        # Voxel Reconstruction Loop (all Parameters)
        for i in range(self.data.num_params):

            # Forward Propagation
            X_batch = torch.empty((0, 1))
            y_target = y_batch[i, :].repeat(X_train_batch.shape[0], 1)
            X_target = self.model(X_train_batch, y_train_batch, y_target)

            # Train -> Train Set Loss Computation
            if i < self.data.train_params:
                for b in range(self.data.settings.batch_size):
                    idx = (b * self.data.train_params) + i
                    X_batch = torch.cat((X_batch, X_train_batch[idx, :].repeat(self.data.train_params, 1)), dim = 0)
                tt_loss = tt_loss + (self.criterion(X_target, X_batch) / self.data.train_params)
                
            # Train -> Validation Set Loss Computation
            else:
                for b in range(self.data.settings.batch_size):
                    idx = (b * self.data.val_params) + (i - self.data.train_params)
                    X_batch = torch.cat((X_batch, X_val_batch[idx, :].repeat(self.data.train_params, 1)), dim = 0)
                tv_loss = tv_loss + (self.criterion(X_target, X_batch) / self.data.train_params)
            
        # Backwards Propagation
        self.optimizer.zero_grad()
        tt_loss.backward(retain_graph = True)
        tv_loss.backward(retain_graph = True)
        self.optimizer.step()

        del idx, X_train_batch, X_val_batch, y_train_batch, y_val_batch, X_batch, y_batch, X_target, y_target
        return {"tt_loss": tt_loss, "tv_loss": tv_loss, 'train_loss': tt_loss + tv_loss}

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx):
        self.tt_loss = self.tt_loss + loss['tt_loss'].item()
        self.tv_loss = self.tv_loss + loss['tv_loss'].item()
        #self.vt_loss = self.vt_loss + loss['vt_loss'].item()
        #self.vv_loss = self.vv_loss + loss['vv_loss'].item()
        self.train_loss = self.train_loss + loss['train_loss'].item()
        #self.val_loss = self.val_loss + loss['val_loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Reconstructed Images
        self.tt_loss = self.tt_loss / self.train_batches
        self.tv_loss = self.tv_loss / self.val_batches
        #self.vt_loss = self.vt_loss / self.train_batches
        #self.vv_loss = self.vv_loss / self.val_batches
        self.train_loss = self.train_loss / self.train_batches
        #self.val_loss = self.val_loss / self.train_batches
        train_plot, recon_loss_best, recon_loss_worst = self.reconstruct(t = 0, sel_slice = 25)

        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        self.train_logger.experiment.add_scalar("Train -> Train Reconstruction Loss", self.tt_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Train -> Val Interpolation Loss", self.tv_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Val -> Train Interpolation Loss", self.vt_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Val -> Val Reconstruction Loss", self.vv_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Total Training Loss", self.train_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Total Validation Loss", self.val_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Best Image Reconstruction Loss", recon_loss_best, self.num_epochs)
        self.train_logger.experiment.add_scalar("Worst Image Reconstruction Loss", recon_loss_worst, self.num_epochs)
        self.train_logger.experiment.add_figure(f'Image Reconstructions', train_plot, self.num_epochs)

        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': self.num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)
        
    ##############################################################################################
    # -------------------------------------- Testing Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Testing
    def on_test_start(self):
        
        # Model Testing Mode Setup
        self.model.eval()
        self.automatic_optimization = False
        self.test_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Testing Performance')
        self.test_logger.experiment.add_graph(self.model, ( torch.rand(1), 
                                                            torch.rand(self.settings.num_labels),
                                                            torch.rand(self.settings.num_labels)))
        
        # Training & Validation Set Example Patient Dataset 
        self.data = h1DMUDI.load(self.settings.data_folderpath, self.settings.data_version)
        self.ex_patient = 4; self.pX, self.pMask = self.data.get_patient(self.ex_patient)

    # Functionality called upon the Start of Testing Epoch
    def on_test_epoch_start(self):
        self.num_epochs = self.past_epochs + self.current_epoch
        self.tt_loss = 0; self.vt_loss = 0
        self.tv_loss = 0; self.vv_loss = 0
        self.train_loss = 0; self.val_loss = 0

    # --------------------------------------------------------------------------------------------

    # Testing Step / Batch Loop 
    def test_step(self, batch, batch_idx):

        # Full Batch Input Data Handling
        X_train_batch, y_train_batch = batch['train']
        X_val_batch, y_val_batch = batch['val']
        X_batch = torch.cat((X_train_batch, X_val_batch), dim = 0)
        y_batch = torch.cat((y_train_batch, y_val_batch), dim = 0)
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        y_batch = y_batch.type(torch.float).to(self.settings.device)

        # Voxel Reconstruction Loop (all Parameters)
        assert(y_batch.shape == self.data.params.shape
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"
        tt_loss = torch.zeros(1); tv_loss = torch.zeros(1)
        vt_loss = torch.zeros(1); vv_loss = torch.zeros(1)
        for i in range(self.data.num_params):
            
            # Forward Propagation
            y_target = y_batch[i, :].repeat(self.data.num_params, 1)
            X_target = self.model(X_batch, y_batch, y_target)

            # Loss Computation
            t_loss = self.criterion(X_target[0:self.data.train_params, :],
                            X_batch[i, :].repeat(self.data.train_params, 1))
            v_loss = self.criterion(X_target[self.data.train_params::, :],
                            X_batch[i, :].repeat(self.data.val_params, 1))
            if i < self.data.train_params:
                tt_loss = tt_loss + (t_loss / self.data.train_params)
                vt_loss = vt_loss + (v_loss / self.data.val_params)
            else:
                tv_loss = tv_loss + (t_loss / self.data.train_params)
                vv_loss = vv_loss + (v_loss / self.data.val_params)
            del X_target, y_target, t_loss, v_loss
        del X_batch, X_train_batch, X_val_batch, y_train_batch, y_val_batch, y_batch
        return {"tt_loss": tt_loss, "vt_loss": vt_loss, 'train_loss': tt_loss + tv_loss,
                'tv_loss': tv_loss, 'vv_loss': vv_loss, 'val_loss': vt_loss + vv_loss,}

    # Functionality called upon the End of a Batch Testing Step
    def on_test_batch_end(self, loss, batch, batch_idx):
        self.tt_loss = self.tt_loss + loss['tt_loss'].item()
        self.vt_loss = self.vt_loss + loss['vt_loss'].item()
        self.tv_loss = self.tv_loss + loss['tv_loss'].item()
        self.vv_loss = self.vv_loss + loss['vv_loss'].item()
        self.train_loss = self.train_loss + loss['train_loss'].item()
        self.val_loss = self.val_loss + loss['val_loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Testing Epoch
    def on_test_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Reconstructed Images
        self.tt_loss = self.tt_loss / self.test_batches
        self.vt_loss = self.vt_loss / self.test_batches
        self.tv_loss = self.tv_loss / self.test_batches
        self.vv_loss = self.vv_loss / self.test_batches
        self.train_loss = self.train_loss / self.test_batches
        self.val_loss = self.val_loss / self.test_batches
        test_plot, recon_loss = self.reconstruct(   mode = 'Test',
                                                    sel_params = 0,
                                                    sel_slice = 25)

        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        self.test_logger.experiment.add_scalar("Train -> Train Reconstruction Loss", self.tt_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Val -> Train Interpolation Loss", self.vt_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Train -> Val Interpolation Loss", self.tv_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Val -> Val Reconstruction Loss", self.vv_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Total Training Loss", self.train_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Total Validation Loss", self.val_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Image Reconstruction Loss", recon_loss, self.num_epochs)
        self.test_logger.experiment.add_figure(f'Image Reconstruction', test_plot, self.num_epochs)

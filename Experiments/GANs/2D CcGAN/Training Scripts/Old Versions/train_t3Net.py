# Library Imports
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import alive_progress
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from alive_progress import alive_bar

# Access to Model Classes
sys.path.append('../Model Builds')
from LabelEmbedding import LabelEmbedding, t3Net
from Generator import Generator
from Discriminator import Discriminator

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Label Embedding / T3 Model Training
def train_t3Net(
    embedNet: LabelEmbedding,                       # Trained T1 & T2 Conjoint Model
    train_set: DataLoader,                          # Training Set's Train DataLoader
    settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    train: bool = True,                             # Boolean Control Variable: False if the purpose is
):                                                  # to just Load the Selected Model Version

    # Learning Rate Decay Function
    def alpha_decay_t3(
        optimizer,
        epoch: int,
        decay_epochs: list = [150, 250, 350]
    ):
        
        # Learning Rate Decrease
        lr = settings.base_lr
        for i in range(len(decay_epochs)):
            if epoch >= decay_epochs[i]:
                lr = lr * settings.lr_decay
        for group in optimizer.param_groups:
            group['lr'] = lr
    
    # --------------------------------------------------------------------------------------------

    # Model Architecture Initialization / Existing Model Checkpoint Loading
    model = t3Net(dim_embedding = settings.dim_embedding)
    current_epoch = 0; criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = settings.base_lr,
                                momentum = 0.9,
                                weight_decay = settings.weight_decay)
    
    # Existing Model Checkpoint Loading
    model_filepath = Path(f"{settings.save_folderpath}/V{settings.model_version}/T3 Net (V{settings.model_version}).pth")
    if settings.model_version != 0 and model_filepath.exists():

        # Checkpoint Fixing (due to the use of nn.DataParallel)
        checkpoint = torch.load(model_filepath); checkpoint_fix = dict()
        for sd, sd_value in checkpoint.items():
            if sd == 'ModelSD' or sd == 'OptimizerSD':
                checkpoint_fix[sd] = OrderedDict()
                for key, value in checkpoint[sd].items():
                    if key[0:7] == 'module.':
                        checkpoint_fix[sd][key[7:]] = value
                    else: checkpoint_fix[sd][key] = value
            else: checkpoint_fix[sd] = sd_value
        
        # Application of Checkpoint's State Dictionary
        model.load_state_dict(checkpoint_fix['ModelSD'])
        optimizer.load_state_dict(checkpoint_fix['OptimizerSD'])
        current_epoch = checkpoint_fix['Training Epochs']
        torch.set_rng_state(checkpoint_fix['RNG State'])
        del checkpoint, checkpoint_fix
    model = nn.DataParallel(model.to(settings.device))

    # --------------------------------------------------------------------------------------------

    # Model Training Mode
    if not(train): print(f"DOWNLOAD: Embedding Net T3 Model (V{settings.model_version})")
    else:

        # Connection to the T1 & T2 Model Optimizer
        embedNet.eval()
        t2Net = embedNet.module.t2Net                                  # T2 Model
        optimizer = torch.optim.SGD(t2Net.parameters(),         # T3 Model Optimizer
                                    lr = settings.base_lr,      #  
                                    momentum = 0.9,
                                    weight_decay = settings.weight_decay)

        # Epoch Loop
        if settings.model_version == 0: settings.num_epochs = 1
        t3_loss_table = np.empty(0, dtype = np.float)
        for epoch in range(current_epoch, current_epoch + settings.num_epochs):
            
            # Training Mode Enabling
            model.train(); train_loss = 0
            alpha_decay_t3(optimizer, epoch)

            # Batch Loop
            with alive_bar( len(train_set), bar = 'blocks',
                        title = f'Epoch #{epoch} | T3 Training   ',
                        force_tty = True) as train_bar:
                for i, (X_batch, ygt_batch) in enumerate(train_set):

                    # Label Handling + Noise Addition
                    ygt_batch = ygt_batch.type(torch.float).to(settings.device)
                    gamma_batch = np.random.normal(0, 0.2, ygt_batch.shape)
                    gamma_batch = torch.from_numpy(gamma_batch).type(torch.float).to(settings.device)
                    ygt_noise_batch = torch.clamp(ygt_batch + gamma_batch, 0.0, 1.0)

                    # Forward Pass
                    h_noise_batch = model(ygt_noise_batch)              # T3 Model (y -> h)
                    y_noise_batch = t2Net(h_noise_batch)                # T2 Model (h -> y)
                    loss = criterion(y_noise_batch, ygt_noise_batch)    # Loss Computation
                    del X_batch, ygt_batch, gamma_batch, ygt_noise_batch, y_noise_batch, h_noise_batch

                    # Backward Pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Training Batch Progress Tracking
                    time.sleep(1); train_bar()
                    train_loss += loss.cpu().item()
                    if settings.model_version == 0 and i == 0: break
                train_loss = train_loss / len(train_set)
                t3_loss_table = np.append(t3_loss_table, train_loss)
                train_bar.text = f"Loss: {train_loss}"
        
        # --------------------------------------------------------------------------------------------

            # Model Progress & State Dictionary Saving 
            print(f"Epoch #{epoch} | T3 Train Loss: {np.round(train_loss, 3)}")
            print("--------------------------------------------------------------------------------------------")
            torch.save({'ModelSD': model.state_dict(),
                        'OptimizerSD': optimizer.state_dict(),
                        'Training Epochs': epoch,
                        'RNG State': torch.get_rng_state()},
                        model_filepath)

        # Training Performance Evaluation - Loss Analysis
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(t3_loss_table, 'g', label = 'T3 Model'); ax.set_xticks([])
        plt.savefig(Path(f"{settings.save_folderpath}/V{settings.model_version}/T3 Loss (V{settings.model_version}).png"))

    return model

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

# Label Embedding / T1 & T2 Models Training
def train_embedNet(
    train_set: DataLoader,                          # Training Set's Train DataLoader
    val_set: DataLoader,                            # Training Set's Validation DataLoader
    settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    train: bool = True,                             # Boolean Control Variable: False if the purpose is
):                                                  # to just Load the Selected Model model_version

    # Learning Rate Decay Function
    def alpha_decay_t12(
        optimizer,
        epoch: int,
        decay_epochs: list = [80, 140]
    ):
        
        # Learning Rate Decrease
        lr = settings.base_lr
        for i in range(len(decay_epochs)):
            if epoch >= decay_epochs[i]:
                lr = lr * settings.lr_decay
        for group in optimizer.param_groups:
            group['lr'] = lr

    # --------------------------------------------------------------------------------------------
    
    # Model Architecture, Loss & Optimization Initialization
    model = LabelEmbedding( in_channel = 64,
                            expansion= settings.expansion,
                            dim_embedding = settings.dim_embedding)
    current_epoch = 0; criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = settings.base_lr,
                                momentum = 0.9,
                                 weight_decay = settings.weight_decay)

    # Existing Model Checkpoint Loading
    model_filepath = Path(f"{settings.save_folderpath}/V{settings.model_version}/Embedding Net (V{settings.model_version}).pth")
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

    # Epoch Loop
    if not(train): print(f"DOWNLOAD: Embedding Net T12 Model (V{settings.model_version})")
    else:
        if settings.model_version == 0: settings.num_epochs = 1
        t12_train_loss_table = np.empty(0, dtype = np.float)
        t12_val_loss_table = np.empty(0, dtype = np.float)
        for epoch in range(current_epoch, current_epoch + settings.num_epochs):
            
            # Training Mode Enabling
            model.train(); train_loss = 0
            alpha_decay_t12(optimizer, epoch)

            # Batch Loop
            with alive_bar( len(train_set), bar = 'blocks',
                            title = f'Epoch #{epoch} | T12 Training  ',
                            force_tty = True) as train_bar:
                for i, (X_batch, ygt_batch) in enumerate(train_set):

                    # Forward Pass
                    X_batch = X_batch.type(torch.float).to(settings.device)
                    h_batch, y_batch = model(X_batch)
                    loss = criterion(y_batch, ygt_batch)
                    del X_batch, ygt_batch, h_batch

                    # Bacward Pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Training Batch Progress Tracking
                    time.sleep(1); train_bar()
                    train_loss += loss.cpu().item()
                    if settings.model_version == 0 and i == 0: break
            train_loss = train_loss / len(train_set)
            t12_train_loss_table = np.append(t12_train_loss_table, train_loss)
            train_bar.text = f"Loss: {train_loss}"
            
        # --------------------------------------------------------------------------------------------

            # Validation Mode Enabling
            model.eval(); val_loss = 0
            with torch.no_grad():

                # Batch Loop
                with alive_bar( len(val_set), bar = 'blocks',
                                title = f'Epoch #{epoch} | T12 Validation',
                                force_tty = True) as val_bar:
                    for i, (X_batch, ygt_batch) in enumerate(val_set):

                        # Forward Pass
                        X_batch = X_batch.type(torch.float).to(settings.device)
                        ygt_batch = ygt_batch.type(torch.float).to(settings.device)
                        h_batch, y_batch = model(X_batch)
                        loss = criterion(y_batch, ygt_batch)
                        del X_batch, ygt_batch, h_batch

                        # Validation Batch Progress Tracking
                        time.sleep(1); val_bar()
                        val_loss += loss.cpu().item()
                        if settings.model_version == 0 and i == 0: break
                val_loss = val_loss / len(train_set)
                t12_val_loss_table = np.append(t12_val_loss_table, val_loss)
                val_bar.text = f"Loss: {val_loss}"
        
    # --------------------------------------------------------------------------------------------

            # Model Progress & State Dictionary Saving 
            print(f"Epoch #{epoch} | T12 Train Loss: {np.round(train_loss, 3)} | T12 Validation Loss: {np.round(val_loss, 3)}")
            print("--------------------------------------------------------------------------------------------")
            torch.save({'ModelSD': model.state_dict(),
                        'OptimizerSD': optimizer.state_dict(),
                        'Training Epochs': epoch,
                        'RNG State': torch.get_rng_state()},
                        model_filepath)
        
        # Training Performance Evaluation - Loss Analysis
        fig, ax = plt.subplots(figsize = (10, 10)); ax.set_xticks([])
        ax.plot(t12_train_loss_table, 'g', label = 'Training')
        ax.plot(t12_val_loss_table, 'r', label = 'Validation')
        ax.legend(loc = 'upper right'); ax.set_title('T12 Loss'); ax.set_xticks([])
        plt.savefig(Path(f"{settings.save_folderpath}/V{settings.model_version}/T12 Loss (V{settings.model_version}).png"))

    return model
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

# Generator & Discriminator Models Training
def train_gan(
    t3Net: t3Net,                                   # Trained T3 Model
    train_set: DataLoader,                          # Training Set's Train DataLoader
    settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    train: bool = True,                             # Boolean Control Variable: False if the purpose...
):                                                  # is to just Load the Selected Model model_version

    # Models Architecture & Optimizer Initialization
    gen = Generator(dim_z = settings.dim_z, dim_embedding = settings.dim_embedding)
    dis = Discriminator(); current_epoch = 0
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr = settings.lr_ccgan, betas = (0.5, 0.999))
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr = settings.lr_ccgan, betas = (0.5, 0.999))

    # Existing Model Checkpoint Loading
    gen_filepath = Path(f"{settings.save_folderpath}/V{settings.model_version}/Generator (V{settings.model_version}).pth")
    dis_filepath = Path(f"{settings.save_folderpath}/V{settings.model_version}/Discriminator (V{settings.model_version}).pth")
    if settings.model_version != 0 and gen_filepath.exists() and dis_filepath.exists():

        # Generator Checkpoint Fixing (due to the use of nn.DataParallel)
        gen_checkpoint = torch.load(gen_filepath); gen_checkpoint_fix = dict()
        for sd, sd_value in gen_checkpoint.items():
            if sd == 'ModelSD' or sd == 'OptimizerSD':
                gen_checkpoint_fix[sd] = OrderedDict()
                for key, value in gen_checkpoint[sd].items():
                    if key[0:7] == 'module.':
                        gen_checkpoint_fix[sd][key[7:]] = value
                    else: gen_checkpoint_fix[sd][key] = value
            else: gen_checkpoint_fix[sd] = sd_value

        # Discriminator Checkpoint Fixing (due to the use of nn.DataParallel)
        dis_checkpoint = torch.load(gen_filepath); dis_checkpoint_fix = dict()
        for sd, sd_value in dis_checkpoint.items():
            if sd == 'ModelSD' or sd == 'OptimizerSD':
                dis_checkpoint_fix[sd] = OrderedDict()
                for key, value in dis_checkpoint[sd].items():
                    if key[0:7] == 'module.':
                        dis_checkpoint_fix[sd][key[7:]] = value
                    else: dis_checkpoint_fix[sd][key] = value
            else: dis_checkpoint_fix[sd] = sd_value

        # Generator Checkpoint Loading
        gen.load_state_dict(gen_checkpoint_fix['ModelSD'])
        gen_optimizer.load_state_dict(gen_checkpoint_fix['OptimizerSD'])
        current_epoch = gen_checkpoint_fix['Training Epochs']

        # Discriminator Checkpoint Loading
        dis.load_state_dict(dis_checkpoint_fix['ModelSD'])
        dis_optimizer.load_state_dict(dis_checkpoint_fix['OptimizerSD'])
        torch.set_rng_state(dis_checkpoint_fix['RNG State'])
        del gen_checkpoint, dis_checkpoint, gen_checkpoint_fix, dis_checkpoint_fix
    
    # Model Transfer to CUDA Device
    t3Net = t3Net.to(settings.device); t3Net.eval()
    gen = nn.DataParallel(gen).to(settings.device)
    dis = nn.DataParallel(dis).to(settings.device)

    # --------------------------------------------------------------------------------------------

    if not(train):
        print(f"DOWNLOAD: Generator Model (V{settings.model_version})")
        print(f"DOWNLOAD: Discriminator Model (V{settings.model_version})")
    else:

        # Data Accessing Betterment
        nRow= nCol = 10
        X_train, y_train = train_set.dataset[:]
        nSamples, nLabels = y_train.shape
        if settings.model_version != 0: assert(settings.batch_size == train_set.batch_size
        ), "ERROR: Batch Size Value not Corresponding"

        # Training Image Selection between the 5th & 95th Label Percentile
        sel_label = np.empty((nRow, nLabels))
        start_label = np.quantile(y_train, 0.05, axis = 0)      # 5th Percentile Label Value
        end_label = np.quantile(y_train, 0.95, axis = 0)        # 95th Percentile Label Value
        for l in range(nLabels): sel_label[:, l] = np.linspace(start_label[l], end_label[l], num = nRow)
        del start_label, end_label

        # Automated Label-Specific Kappa Difference Computation & Gaussian Noise Generation Function
        kappa_list = np.empty(nLabels)
        for l in range(nLabels): kappa_list[l] = np.mean(np.diff(np.sort(np.unique(y_train[:, l].numpy()))))
        if nLabels >= 6: kappa_list[-1] = 0.0
        def eps(samples):
            eps_pos = np.random.normal(0, settings.kernel_eps, (samples, nLabels))
            eps_neg = np.random.normal(0, settings.kernel_eps, (samples, nLabels))
            return (eps_pos - eps_neg) * (kappa_list / 2)

        # --------------------------------------------------------------------------------------------
        
        # Epoch Loop
        if settings.model_version == 0: settings.num_epochs = 1; settings.batch_size = 1
        dis_loss_table = np.empty(0, dtype = np.float)
        gen_loss_table = np.empty(0, dtype = np.float)
        for epoch in range(current_epoch, current_epoch + settings.num_epochs):

            # Discriminator Training Process / Step Loop
            with alive_bar( settings.dis_update, bar = 'blocks',
                        title = f'Epoch #{epoch} | Discriminator ',
                        force_tty = True) as dis_bar:
                for up in range(settings.dis_update):

                    # Random Draw of Batch of Target Labels (w/ Added Gaussian Noise)
                    """[Bug] Since the Target Labels are chosen from the Overall Dataset
                    and not from a List of Unique Labels, there might be some Repeats,
                    though it is also a fact there aren't that many Unique Values"""
                    y_target = y_train[np.random.choice(nSamples, size = settings.batch_size, replace = False), :]   # Random Batch of Target Labels
                    y_vic = y_target + eps(settings.batch_size)                                                      # Initial Addition of Gaussian Noise
                    index_target = np.empty(settings.batch_size, dtype = int)                       # Vicinity-Labelled Real Image Indexes
                    y_fake = np.empty((settings.batch_size, nLabels))                               # Fake Vicinity Labels for Image Generation
                
                    # Batch Loop
                    for i in range(settings.batch_size):
                        
                        # Hard Vicinity-Labelled Real Image Searching
                        index_vic = np.where(np.all(np.abs(y_train - y_vic[i, :]).numpy()           # Index of Real in-Vicinity Training Sample Labels
                                                        <= (kappa_list * 2.0), axis = 1))[0]
                        while len(index_vic) < nLabels:                                             # Redoing of the Vicinity Area Loop...
                            y_vic[i, :] = y_target[i, :] + eps(1)                                   # using different Gaussian Noise values...
                            index_vic = np.where(np.all(np.abs(y_train - y_vic[i, :]).numpy()       # to Ensure that at least 'nlabels' Neighbour...
                                                        <= (kappa_list * 2.0), axis = 1))[0]        # for each Target in the Random Batch is found!
                        index_target[i] = np.random.choice(index_vic, size = 1)                     # Choosing of 1 in-Vicinity Sample per Target

                        # Fake Image Label Generation
                        inf_bound = y_vic[i, :] - kappa_list
                        sup_bound = y_vic[i, :] + kappa_list
                        assert(np.all((inf_bound <= sup_bound).numpy())), "ERROR: Kappa Paremeter wrongly Set!"
                        y_fake[i, :] = np.random.uniform(inf_bound, sup_bound, size = nLabels)      # Random Creation of a Fake Label Sample...
                        assert(np.all(np.abs(y_fake[i, :] - y_vic[i, :].numpy())                    # that must remain within Vicinity...
                                    <= kappa_list)), "ERROR: Kappa Paremeter wrongly Set!"          # of the used batch Target Label 

                    # Hard Vicinity-Labelled Real Image Drawing
                    X_vic = torch.Tensor(X_train[index_target]).type(torch.float).to(settings.device)
                    y_vic = torch.Tensor(y_train[index_target]).type(torch.float).to(settings.device)
                    del index_target, index_vic, inf_bound, sup_bound

                    # Fake Image Generation
                    y_fake = torch.from_numpy(y_fake).type(torch.float).to(settings.device)
                    z_fake = torch.randn(settings.batch_size, settings.dim_z, dtype = torch.float).to(settings.device)
                    X_fake = gen(z_fake, t3Net(y_fake))

                    # Forward Pass
                    w_target = w_fake = torch.ones(settings.batch_size, dtype = torch.float).to(settings.device)
                    out_target = dis(X_vic, t3Net(y_target))		    # Real Sample Discriminator Output
                    out_fake = dis(X_fake, t3Net(y_fake))		        # Fake Sample Discriminator Output

                    # Vanilla Loss Function Computation Switch Case
                    assert(settings.loss == 'vanilla' or settings.loss == 'hinge'
                    ), f"ERROR: Loss Function not Supported!"
                    if settings.loss == 'vanilla':
                        loss_target = torch.nn.Sigmoid()(out_target)
                        loss_fake = torch.nn.Sigmoid()(out_fake)
                        loss_target = torch.log(loss_target + 1e-20)        # Real Sample Loss Value
                        loss_fake = torch.log(loss_fake + 1e-20)            # Fake Sample Loss Value

                    # Hinge Loss Function Computation Switch Case
                    elif settings.loss == 'hinge':                          
                        loss_target = torch.nn.ReLU()(1.0 - out_target)		# Real Sample Loss Value
                        loss_fake = torch.nn.ReLU()(1.0 + out_fake)		    # Fake Sample Loss Value
                    del X_vic, y_vic, z_fake, X_fake, out_target, out_fake, y_fake, y_target

                    # Backward Pass & Step Update
                    w_target = w_target.unsqueeze(-1); loss_target = loss_target.unsqueeze(-1)
                    w_fake = w_fake.unsqueeze(-1); loss_fake = loss_fake.unsqueeze(-1)
                    dis_loss =  torch.mean(w_target.view(-1) * loss_target.view(-1)) + torch.mean(w_fake.view(-1) * loss_fake.view(-1))
                    dis_optimizer.zero_grad()
                    dis_loss.backward()
                    dis_optimizer.step()
                    dis_loss_table = np.append(dis_loss_table, dis_loss.detach().numpy())
                    time.sleep(1); dis_bar()

            # --------------------------------------------------------------------------------------------

            # Generator Training Process / Step Loop
            gen.train()
            with alive_bar( settings.gen_update, bar = 'blocks',
                    title = f'Epoch #{epoch} | Generator     ',
                    force_tty = True) as gen_bar:
                for up in range(settings.gen_update):

                    # Random Draw of Batch of Target Labels (w/ Added Gaussian Noise)
                    """[Bug] Since the Target Labels are chosen from the Overall Dataset
                    and not from a List of Unique Labels, there might be some Repeats,
                    though it is also a fact there aren't that many Unique Values"""
                    y_target = y_train[np.random.choice(nSamples, size = settings.batch_size, replace = False), :]  # Random Batch of Target Labels
                    y_fake = (y_target + eps(settings.batch_size)).type(torch.float).to(settings.device)            # Initial Addition of Gaussian Noise

                    # Fake Image Generation & Forward Pass
                    z_fake = torch.randn(settings.batch_size, settings.dim_z, dtype = torch.float).to(settings.device)
                    X_fake = gen(z_fake, t3Net(y_fake))
                    out_fake = dis(X_fake, t3Net(y_fake))		        # Fake Sample Discriminator Output

                    # Loss Function Computation Switch Case
                    assert(settings.loss == 'vanilla' or settings.loss == 'hinge'
                    ), f"ERROR: Loss Function not Supported!"
                    if settings.loss == 'vanilla':
                        gen_loss = torch.nn.Sigmoid()(out_fake)
                        gen_loss = torch.log(gen_loss + 1e-20)      # Fake Sample Loss Value
                    elif settings.loss == 'hinge':                          
                        gen_loss = - out_fake.mean()		        # Fake Sample Loss Value
                    del z_fake, X_fake, out_fake, y_fake, y_target

                    # Backward Pass & Step Update
                    gen_optimizer.zero_grad()
                    gen_loss.backward()
                    gen_optimizer.step()
                    gen_loss_table = np.append(gen_loss_table, gen_loss.detach().numpy())
                    time.sleep(1); gen_bar()

            # --------------------------------------------------------------------------------------------

            # Model Progress & State Dictionary Saving
            print(f"Epoch #{epoch} | Discriminator Train Loss: {np.round(dis_loss.detach().numpy(), 3)}")
            print(f"Epoch #{epoch} | Generator Train Loss: {np.round(gen_loss.detach().numpy(), 3)}")
            torch.save({'ModelSD': dis.state_dict(),
                        'OptimizerSD': dis_optimizer.state_dict(),
                        'Training Epochs': epoch,
                        'RNG State': torch.get_rng_state()},
                        dis_filepath)
            torch.save({'ModelSD': gen.state_dict(),
                        'OptimizerSD': gen_optimizer.state_dict(),
                        'Training Epochs': epoch,
                        'RNG State': torch.get_rng_state()},
                        gen_filepath)

        # --------------------------------------------------------------------------------------------

        # Training Performance Evaluation - Example Images Visualization
        z_fix = torch.randn(nRow * nCol, settings.dim_z, dtype = torch.float).to(settings.device)
        gen.eval(); y_fix = np.empty((nRow * nCol, nLabels))
        for i in range(nRow):
            current_label = sel_label[i, :]
            for j in range(nCol):
                y_fix[(i * nCol) + j, :] = current_label
        y_fix = torch.from_numpy(y_fix).type(torch.float).to(settings.device)
        with torch.no_grad(): X_fix = gen(z_fix, t3Net(y_fix)).detach().cpu()
        fig, axs = plt.subplots(int(np.ceil(nRow/2)), int(np.ceil(nCol/2)), figsize=(15, 15)); fig.tight_layout()
        for i in range(int(np.ceil(nRow/2))):
            for j in range(int(np.ceil(nCol/2))):
                axs[i, j].imshow(X_fix[int((i * np.ceil(nRow/2)) + j), 0, :, :], cmap = 'gray')
                plt.axis('off'), axs[i, j].xaxis.set_visible(False); axs[i, j].yaxis.set_visible(False)
        plt.savefig(Path(f"{settings.save_folderpath}/V{settings.model_version}/Example Images (V{settings.model_version}).png"))
        del z_fix, X_fix, y_fix
        
        # Training Performance Evaluation - Loss Analysis
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(dis_loss_table, 'g', label = 'Discriminator')
        ax.plot(gen_loss_table, 'r', label = 'Generator')
        ax.legend(loc = 'upper right'); ax.set_title('GAN Loss'); ax.set_xticks([])
        plt.savefig(Path(f"{settings.save_folderpath}/V{settings.model_version}/GAN Loss (V{settings.model_version}).png"))
        
        # Training Performance Evaluation - Other Analytics
        #

    return dis, gen

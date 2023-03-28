# Library Imports
import os
import pickle
import random
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py

# Functionality Import
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from nilearn.image import load_img
from nilearn.masking import unmask
from scipy.ndimage.interpolation import rotate
from sklearn.preprocessing import StandardScaler
from ipywidgets import interactive, IntSlider
from tabulate import tabulate
from alive_progress import alive_bar

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Horizontal 1D MUDI Dataset Initialization Class
class h1DMUDI(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        #info_filepath,         # header_.csv
        settings: argparse.ArgumentParser
    ):
        
        # Parameter & Patient Index Value Access
        super(h1DMUDI).__init__()
        self.settings = settings; self.version = self.settings.version
        self.idx = pd.read_csv( self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.params = pd.read_excel(self.settings.param_filepath)       # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                          # Total Number of Parameters in Dataset
        self.num_patients = len(self.settings.patient_list)             # Total Number of Patients in Dataset
        self.num_train_params = self.settings.num_train_params          # Number of Training Parameters
        self.num_val_params = self.num_params - self.num_train_params   # Number of Validation Parameters

        # ----------------------------------------------------------------------------------------
        
        # Dataset Sample & Label Handling Settings
        self.num_labels = self.settings.num_labels
        if self.settings.gradient_coord:
            self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])                # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])      # from 3D Cartesian or Polar Referential
        assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"
        if self.settings.label_norm:                                                                    # Control Boolean Value for the Normalization of Labels
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)
        
        # ----------------------------------------------------------------------------------------

        # Data Splitting
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        self.idxv_train, self.idxv_test = self.vsplit()
        self.idxh_train, self.idxh_val = self.hsplit()

    ##############################################################################################
    # -------------------------------------- Data Splitting --------------------------------------
    ##############################################################################################

    # Vertical / Patient Data Split
    def vsplit(self):

        # Test Patient(s) Selection Feature
        self.num_test_patients = self.settings.num_test_patients                # Number of Test Set Patients
        self.num_train_patients = self.num_patients - self.num_test_patients    # Number of Training Set Patients
        patient_list = self.settings.patient_list.copy(); patient_list.sort()   # Complete Sorted Patient List
        
        # [Option #1] Selected Test Patient(s)
        if self.settings.sel_test_patients is not None:
            self.test_patients = self.settings.sel_test_patients
            self.train_patients = patient_list.copy()
            self.train_patients.remove(self.test_patients)

        # [Option #2] Non-Specific Test Patient(s)
        else:
            if self.settings.patient_shuffle: random.shuffle(patient_list)      # Patient Shuffling Feature
            self.train_patients = patient_list[0 : self.num_train_patients]
            self.test_patients = patient_list[self.num_train_patients : :]
            self.train_patients.sort(); self.test_patients.sort()

        # Training & Test Patient Sample Indexes
        idx_train = self.idx[np.isin(self.idx[:, 1], self.train_patients), 0]
        idx_test = self.idx[np.isin(self.idx[:, 1], self.test_patients), 0]
        return idx_train, idx_test

    # --------------------------------------------------------------------------------------------

    # Horizontal / Parameter Data Split
    def hsplit(self):

        # [Option #1] Existing Selection of Set Labels
        self.idxh_train_filepath = Path(f"{self.settings.save_folderpath}/1D Training Labels (V{self.version}).txt")    # Filepath for Selected Training Labels
        self.idxh_val_filepath = Path(f"{self.settings.save_folderpath}/1D Validation Labels (V{self.version}).txt")    # Filepath for Selected Validation Labels
        if self.idxh_train_filepath.exists():
            print(f"DOWNLOADING Training Parameters (Version {self.settings.version})")
            idx_train = np.sort(np.loadtxt(self.idxh_train_filepath)).astype(int)
            if self.idxh_val_filepath.exists():
                idx_val = np.sort(np.loadtxt(self.idxh_val_filepath)).astype(int)
            else: idx_val = np.setdiff1d(   np.array(range(0, self.num_params)),
                                            idx_train).astype(int)
        
        # [Option #2] Non-Existing Selection of Set Labels
        else:
            if self.settings.sample_shuffle:                        # Sample Shuffling Feature
                idx_y = np.random.permutation(self.num_params)
            else: idx_y = np.array(range(0, self.num_params))
            idx_train = idx_y[0 : self.num_train_params]
            idx_val = idx_y[self.num_train_params : self.num_params]
        return idx_train, idx_val

    ##############################################################################################
    # --------------------------------- Label & Feature Handling ---------------------------------
    ##############################################################################################

    # Label Scaler Download & Reverse Transformation
    def label_unscale(
        path: Path,
        version: int,
        y: np.array or pd.DataFrame
    ):

        # Label Scaler Download & Reverse Usage
        scaler = torch.load(f"{path}/V{version}/1D Label Scaler (V{version}).pkl")
        return scaler.inverse_transform(y)
    
    # --------------------------------------------------------------------------------------------

    # Dataset Indexing Functionality
    def get_data(
        self,
        idxv: int or np.array,
        idxh: int or np.array
    ): 
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        return torch.Tensor(self.data[idxv, :])

    # DataLoader Construction Functionality
    def get_loader(
        self,
        idxv: int or np.array,
    ):  
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        return DataLoader(TensorDataset(torch.Tensor(self.data[idxv, :][:, self.idxh_train]),
                                        torch.Tensor(self.data[idxv, :][:, self.idxh_val])),
                                        num_workers = self.settings.num_workers, shuffle = False,
                                        batch_size = self.settings.batch_size)

    # --------------------------------------------------------------------------------------------

    # Patient Data Acquisition Functionality
    def get_patient(
        self,
        num_patient: int,
    ):
        
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')

        # Full Patient Data Acquisition
        assert(num_patient in self.settings.patient_list
        ), f"ERROR: Chosen Patient {num_patient} is not contained in the Dataset provided!"
        idxv = self.idx[np.isin(self.idx[:, 1], num_patient), 0]
        pX = torch.Tensor(self.data[idxv, :])

        # Patient Mask Acquisition
        mask_filepath = Path(f"{self.settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()
        ), f"ERROR: Mask for Patient {num_patient} is not contained in the Dataset provided!"
        pMask = load_img(mask_filepath)
        #X = unmask(pX.T, pMask); X = X.get_fdata() 
        return pX, pMask

    ##############################################################################################
    # ------------------------------------- Saving & Loading -------------------------------------
    ##############################################################################################

    # Dataset Saving Functionality
    def save(self):

        # Dataset Loader Saving
        torch.save(self.get_loader(self.idxv_train), f"{self.settings.save_folderpath}/1D TrainLoader (V{self.version}).pkl")
        torch.save(self.get_loader(self.idxv_test), f"{self.settings.save_folderpath}/1D TestLoader (V{self.version}).pkl")
        torch.save(self.scaler, f"{self.settings.save_folderpath}/1D Label Scaler (V{self.version}).pkl")
        if not self.idxh_train_filepath.exists(): np.savetxt(self.idxh_train_filepath, np.array(self.idxh_train))
        if not self.idxh_val_filepath.exists(): np.savetxt(self.idxh_val_filepath, np.array(self.idxh_val))

        # Full Dataset Class Saving
        f = open(f'{self.settings.save_folderpath}/Horizontal 1D MUDI (Version {self.version})', 'wb')
        del self.data; pickle.dump(self, f); f.close

    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loading Function
    def load(
        path: Path,
        version: int = 0,
    ):
        f = open(f'{path}/Horizontal 1D MUDI (Version {version})', 'rb')
        mudi = pickle.load(f); f.close
        return mudi

    # Dataset Loader Loading Function
    def loader(
        path: Path,
        version: int = 0,
        set_: str = 'Train',
    ):
        return torch.load(f"{path}/1D {set_}Loader (V{version}).pkl")
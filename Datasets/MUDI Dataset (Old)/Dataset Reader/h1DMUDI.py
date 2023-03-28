# Library Imports
import os
import pickle
import psutil
import itertools
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itk
import itkwidgets
import time
import alive_progress

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
        settings: argparse.ArgumentParser
    ):
        
        # Parameter Value Access
        super(h1DMUDI).__init__()
        self.settings = settings; self.version = self.settings.version
        self.params = pd.read_excel(self.settings.param_filepath)       # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                          # Total Number of Parameters in Dataset
        #assert(self.num_params == self.data.shape[0]), "ERROR: Number of Parameters is Incoherent" 

        # Patient Information Access
        self.patient_info = pd.read_csv(self.settings.info_filepath)    # List of Patients and Corresponding IDs & Image Sizes inside Full Dataset
        self.patient_info = self.patient_info[:-1]                      # Eliminating the Last Row containing Useless Information from the Patient Information
        self.num_patients = self.patient_info.shape[0]                  # Number of Patients inside Full Dataset
        self.progress = False                                           # Control Boolean Value for Progress Saving (Data can only be saved if Split)

        # ----------------------------------------------------------------------------------------------------------------------------
        
        # Dataset Sample & Label Handling Settings
        self.num_labels = self.settings.num_labels
        if self.settings.gradient_coord: self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])   # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])                      # from 3D Cartesian or Polar Referential
        assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"
        if self.settings.label_norm:                                                                                    # Control Boolean Value for the Normalization of Labels
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)

        # ----------------------------------------------------------------------------------------------------------------------------
        
        # Patient & Parameter Shuffling Feature
        if(self.settings.patient_shuffle): self.patient_info = self.patient_info.iloc[np.random.permutation(len(self.patient_info))]
        self.trainStrat_filepath = Path(f"{self.settings.save_folderpath}/1D Training Labels (V{self.version}).txt")
        self.valStrat_filepath = Path(f"{self.settings.save_folderpath}/1D Validation Labels (V{self.version}).txt")
        if self.trainStrat_filepath.exists():
            print(f"DOWNLOADING Training Parameters (Version {self.settings.version})")
            self.idx_train = np.loadtxt(self.trainStrat_filepath).astype(int)
            if self.valStrat_filepath.exists(): self.idx_val = np.loadtxt(self.valStrat_filepath).astype(int)
            else: self.idx_val = np.setdiff1d(np.array(range(0, self.num_params)), self.idx_train).astype(int)
        else:
            if self.settings.sample_shuffle: idx_y = np.random.permutation(self.num_params)
            else: idx_y = range(0, self.num_params)
            self.idx_train = idx_y[0 : self.settings.train_params]
            self.idx_val = idx_y[self.settings.train_params : self.num_params]
        self.y_train = self.params.iloc[self.idx_train]; self.y_val = self.params.iloc[self.idx_val]

    ##############################################################################################
    # --------------------------------- Feature & Label Handling ---------------------------------
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

    ##############################################################################################
    # ---------------------------------- Data Access & Splitting ---------------------------------
    ##############################################################################################

    # Patient Data Access Function
    def get_patient(
        self,
        patient_number: int,                # Number for the Patient File being Read and Acquired (in Order)
    ):

        # Patient Data Information
        assert(0 <= patient_number < self.num_patients), f"ERROR: Input Patient not Found!"         # Assertion for the Existence of the Requested Patient
        patient_id = self.patient_info['Patient'].iloc[patient_number]                              # Patient ID contained within the Patient List
        patient_filepath = Path(f"{self.settings.patient_folderpath}/p{patient_id}.csv")            # Patient Filepath from detailed Folder
        mask_filepath = Path(f"{self.settings.mask_folderpath}/p{patient_id}.nii")                  # Mask Filepath from detailed Folder
        
        # Patient Data Access Memory Requirements
        assert(patient_filepath.exists()                                                            # Assertion for the Existence of Patient File in said Folder
        ), f"Filepath for Patient {patient_id} is not in the Dataset!"
        assert(mask_filepath.exists()                                                               # Assertion for the Existence of Mask File in said Folder
        ), f"Filepath for Mask {patient_id} is not in the Dataset!"
        file_size = os.path.getsize(patient_filepath)                                               # Memory Space occupied by Patient File
        mask_size = os.path.getsize(mask_filepath)                                                  # Memory Space occupied by Mask File
        available_memory = psutil.virtual_memory().available                                        # Memory Space Available for Computation
        assert(available_memory >= (file_size + mask_size)                                          # Assertion for the Existence of Available Memory Space
        ), f"ERROR: Dataset requires {file_size + mask_size}b, but only {available_memory}b is available!"
        
        # Patient Data Access
        pX = pd.read_csv(patient_filepath); del pX['Unnamed: 0']                                    # Full Patient Data
        pMask = load_img(mask_filepath)                                                             # Patient Mask Data
        #pX = unmask(pX, pMask); pX = pX.get_fdata()                                                # Unmasking of Full Patient Data
        del available_memory, mask_size, file_size
        return pX, pMask
    
    # ----------------------------------------------------------------------------------------------------------------------------

    # Patient Data Splitting Function
    def split_patient(
        self,
        patient_number: int,                # Number for the Patient File being Read and Acquired (in Order)
        train_params: float or int = 500,   # Number / Percentage of Parameters to be used in the Training Section of the Patient
    ):

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        if(self.settings.percentage):
            assert(0 < train_params <= 100
            ), f"ERROR: Training Parameter Number not Supported!"       # Percentage Limits for Number of Training Parameters
            val_params = 1 - train_params                               # Percentage Value for Validation Parameters

        # Computation of Training & Validation Parameter Numbers (Numerical Input)
        else:
            assert(0 < train_params <= self.num_params
            ), f"ERROR: Training Parameter Number not Supported!"       # Numerical Limits for Number of Training Parameters
            val_params = self.num_params - train_params                 # Numerical Value for Validation Parameters

        # ----------------------------------------------------------------------------------------------------------------------------

        # Patient Dataset Splitting into Training & Validation Sets
        pX, pMask = self.get_patient(patient_number)
        py_train = self.y_train; py_val = self.y_val
        pX_train = pX.iloc[self.idx_train]
        pX_val = pX.iloc[self.idx_val]
        del pX, pMask

        # Inclusion of Patient ID Label
        if self.settings.patient_id:
            py_train['Patient'] = self.patient_info['Patient'].iloc[patient_number]
            py_val['Patient'] = self.patient_info['Patient'].iloc[patient_number]

        # 1D Image to Voxel-Wise Sample Conversion
        if self.settings.conversion:
            
            # Feature & Label Conversion
            py_train = pd.concat([py_train] * pX_train.shape[1], ignore_index = False)
            py_val = pd.concat([py_val] * pX_val.shape[1], ignore_index = False)
            pX_train = pd.DataFrame(np.ravel(pX_train), index = py_train.index)
            pX_val = pd.DataFrame(np.ravel(pX_val), index = py_val.index)

        return pX_train, pX_val, py_train, py_val

    # ----------------------------------------------------------------------------------------------------------------------------
    
    # Dataset Splitting Function
    def split(self):

        # Patient Number Variable Logging
        assert(0 < self.settings.test_patients <= self.num_patients             # Limits for Number of Test Set Patients
        ), f"ERROR: Test Patient Number not Supported!"
        self.train_patients = self.num_patients - self.settings.test_patients   # Number of Patients to be used in the Training Set
        self.test_patients = self.settings.test_patients                        # Number of Patients to be used in the Test Set
        self.progress = True                                                    # Control Boolean Value for Progress Saving (Data can only be saved if Split)
        
        # Computation of Training & Validation Parameter Numbers
        if(self.settings.percentage):
            assert(0 < self.settings.train_params / 100.0 <= 1
            ), f"ERROR: Training Set's Parameter Number not Supported!"             # Percentage Limits for Number of Training Set's Parameters
            self.train_params = int(self.settings.train_params * self.num_params)   # Percentage Value for Training Set's Training Parameters
        else:
            assert(0 < self.settings.train_params <= self.num_params
            ), f"ERROR: Training Set's Parameter Number not Supported!"             # Numerical Limits for Number of Training Set's Parameters
            self.train_params = self.settings.train_params                          # Numerical Value for Training Set's Training Parameters
        self.val_params = self.num_params - self.settings.train_params              # Numerical Value for Training Set's Validation Parameters

        # ----------------------------------------------------------------------------------------------------------------------------

        # Full MUDI Dataset Building
        with alive_bar( self.num_patients,
                        title = '1D MUDI Dataset',
                        force_tty = True) as progress_bar:
            
            # Training & Test Sets Building
            if not(self.settings.conversion): self.mode = ('Train', 'Test')
            else: self.mode = ('TrainTrain', 'TrainVal', 'TestTrain', 'TestVal')
            self.loaders = dict.fromkeys(self.mode)
            for m in ('Train', 'Test'):
                if m == 'Train': patient_array = range(0, self.train_patients)
                else: patient_array = range(self.train_patients, self.train_patients + self.test_patients)

                # Set Scaffolds Initialization
                if self.settings.conversion:
                    X_train = np.empty(list(np.array((0, 1))))
                    X_val = np.empty(list(np.array((0, 1))))
                    y_train = np.empty(list(np.array((0, self.num_labels))))
                    y_val = np.empty(list(np.array((0, self.num_labels))))
                else:
                    X_train = np.empty(list(np.array((0, self.train_params))))
                    X_val = np.empty(list(np.array((0, self.val_params))))

                # Set Patient Loop
                for p in patient_array:

                    # Training Patient Data Access & Treatment
                    progress_bar.text = f"\n-> {m} Set | Patient {self.patient_info['Patient'].iloc[p]}"
                    pX_train, pX_val, py_train, py_val = self.split_patient(patient_number = p,
                                                                            train_params = self.train_params)
                    X_train = np.concatenate((X_train, pX_train), axis = 0)
                    X_val = np.concatenate((X_val, pX_val), axis = 0)
                    if self.settings.conversion:
                        y_train = np.concatenate((y_train, py_train), axis = 0)
                        y_val = np.concatenate((y_val, py_val), axis = 0)
                    time.sleep(0.01); progress_bar()
            
                    # Set DataLoader Construction
                    if p == patient_array[-1]:
                        if self.settings.conversion:
                            self.loaders[m + 'Train'] =  DataLoader(    TensorDataset(  torch.Tensor(X_train),
                                                                                        torch.Tensor(y_train)), 
                                                                        num_workers = self.settings.num_workers, shuffle = False,
                                                                        batch_size = self.settings.batch_size * self.train_params)
                            self.loaders[m + 'Val'] =   DataLoader(     TensorDataset(  torch.Tensor(X_val),
                                                                                        torch.Tensor(y_val)),  
                                                                        num_workers = self.settings.num_workers, shuffle = False,
                                                                        batch_size = self.settings.batch_size * self.val_params)
                        else: self.loaders[m] =  DataLoader(            TensorDataset(  torch.Tensor(X_train),
                                                                                        torch.Tensor(X_val)),
                                                                        num_workers = self.settings.num_workers,
                                                                        batch_size = self.settings.batch_size,
                                                                        shuffle = False)
                        del X_train, X_val, pX_train, pX_val, py_train, py_val

    ##############################################################################################
    # ------------------------------------- Saving & Loading -------------------------------------
    ##############################################################################################

    # Dataset Saving Function
    def save(self):
        if self.progress:

            # Full Dataset Saving
            f = open(f'{self.settings.save_folderpath}/Horizontal 1D MUDI (Version {self.version})', 'wb')
            pickle.dump(self, f); f.close

            # Dataset Loader Saving
            for m in self.mode: torch.save(self.loaders[m], f"{self.settings.save_folderpath}/1D {m}Loader (V{self.version}).pkl")
            torch.save(self.scaler, f"{self.settings.save_folderpath}/1D Label Scaler (V{self.version}).pkl")
            np.savetxt(f"{self.settings.save_folderpath}/1D Training Labels (V{self.version}).txt", np.array(self.idx_train))
            np.savetxt(f"{self.settings.save_folderpath}/1D Validation Labels (V{self.version}).txt", np.array(self.idx_val))
            del self.loaders
            
    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loading Function
    def load(
        path: Path,
        version: int = 0,
    ):
        f = open(f'{path}/Horizontal 1D MUDI (Version {version})', 'rb')
        mudi = pickle.load(f); f.close
        return mudi

    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loader Loading Function
    def loader(
        path: Path,
        version: int = 0,
        set_: str = 'Train',
        mode_: str = 'Train',
    ):
        return torch.load(f"{path}/1D {set_}{mode_}Loader (V{version}).pkl")
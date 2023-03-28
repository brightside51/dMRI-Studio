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

        # Dataset Access
        """
        file_size = os.path.getsize(settings.data_filepath)
        available_memory = psutil.virtual_memory().available
        assert(available_memory >= file_size
        ), f"ERROR: Dataset requires {file_size}b, but only {available_memory}b is available!"
        with h5py.File(settings.data_filepath, 'r') as archive:
            self.data = pd.DataFrame(archive.get('data1')).T
        """

        # Parameter Value Access
        super(h1DMUDI).__init__()
        self.version = settings.version
        self.params = pd.read_excel(settings.param_filepath)    # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                  # Total Number of Parameters in Dataset
        #assert(self.num_params == self.data.shape[0]), "ERROR: Number of Parameters is Incoherent" 

        # Patient Information Access
        self.patient_folderpath = settings.patient_folderpath
        self.save_folderpath = settings.save_folderpath
        self.patient_info = pd.read_csv(settings.info_filepath)     # List of Patients and Corresponding IDs & Image Sizes inside Full Dataset
        self.patient_info = self.patient_info[:-1]                  # Eliminating the Last Row containing Useless Information from the Patient Information
        self.num_patients = self.patient_info.shape[0]              # Number of Patients inside Full Dataset
        self.progress = False                                       # Control Boolean Value for Progress Saving (Data can only be saved if Split)
        
        # Dataset Sample & Label Handling Settings
        self.voxel_wise = settings.voxel_wise
        self.patient_id = settings.patient_id
        self.gradient_coord = settings.gradient_coord
        self.num_labels = settings.num_labels
        self.label_norm = settings.label_norm
        if self.gradient_coord: self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])    # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])              # from 3D Cartesian or Polar Referential
        assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"
        if self.label_norm:                                                                                     # Control Boolean Value for the Normalization of Labels
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)

    ##############################################################################################
    # --------------------------------- Feature & Label Handling ---------------------------------
    ##############################################################################################

    # 1D Image to 1D Voxel Data Conversion Function
    # [num_sample, num_feats] -> [num_sample * num_feats, 1]
    def convert(
        self,
        data: np.ndarray,           # 3D Data Array
        label: pd.DataFrame
    ):

        # Conversion from 1D Image Data to 1D Voxel Data
        label = label.iloc[np.repeat(np.arange(len(label)), data.shape[1])]
        data = pd.DataFrame(data.to_numpy().reshape((data.shape[0] * data.shape[1], 1)))
        return data, label

    # ----------------------------------------------------------------------------------------------------------------------------

    # Full 3D Patient Image Unmasking
    #def unmask()
    
    # ----------------------------------------------------------------------------------------------------------------------------

    # Label Scaler Download & Reverse Transformation
    def label_unscale(
        path: Path,
        version: int,
        y: np.array or pd.DataFrame
    ):

        # Label Scaler Download & Reverse Usage
        scaler = torch.load(f"{path}/1D Label Scaler (V{version}).pkl")
        return scaler.inverse_transform(y)
        
    ##############################################################################################
    # ---------------------------------- Data Access & Splitting ---------------------------------
    ##############################################################################################

    # Patient Data Access Function
    def get_patient(
        self,
        patient_number: int,                # Number for the Patient File being Read and Acquired (in Order)
    ):

        # Patient Data Access (including all Requirements)
        assert(0 <= patient_number < self.num_patients), f"ERROR: Input Patient not Found!"         # Assertion for the Existence of the Requested Patient
        patient_id = self.patient_info['Patient'].iloc[patient_number]                              # Patient ID contained within the Patient List
        patient_filepath = Path(f"{self.patient_folderpath}/p{patient_id}.csv")                     # Patient Filepath from detailed Folder
        assert(patient_filepath.exists()                                                            # Assertion for the Existence of Patient File in said Folder
        ), f"Filepath for Patient {patient_id} is not in the Dataset!"
        file_size = os.path.getsize(patient_filepath)                                               # Memory Space occupied by Patient File
        available_memory = psutil.virtual_memory().available                                        # Memory Space Available for Computation
        assert(available_memory >= file_size                                                        # Assertion for the Existence of Available Memory Space
        ), f"ERROR: Dataset requires {file_size}b, but only {available_memory}b is available!"
        pX = pd.read_csv(patient_filepath); del pX['Unnamed: 0']                                    # Full Patient Data
        del available_memory, file_size
        return pX

    # ----------------------------------------------------------------------------------------------------------------------------
    
    # Patient Data Splitting Function
    def split_patient(
        self,
        patient_number: int,                # Number for the Patient File being Read and Acquired (in Order)
        train_params: int = 500,            # Number / Percentage of Parameters to be used in the Training Section of the Patient
        percentage: bool = False,           # Control Variable for the Usage of Percentage Values in train_params
        sample_shuffle: bool = False,       # Ability to Shuffle the Samples inside both Training and Validation Datasets
    ):

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        if(percentage):
            assert(0 < train_params <= 100                              # Percentage Limits for Number of Training Parameters
            ), f"ERROR: Training Parameter Number not Supported!"
            train_params = train_params / 100                           # Percentage Value for Training Parameters
            val_params = 1 - train_params                               # Percentage Value for Validation Parameters

        # Computation of Training & Validation Parameter Numbers (Numerical Input)
        else:
            assert(0 < train_params <= self.num_params                  # Numerical Limits for Number of Training Parameters
            ), f"ERROR: Training Parameter Number not Supported!"
            val_params = self.num_params - train_params                 # Numerical Value for Validation Parameters
            if self.voxel_wise:                                         # Correction of Vaidation Parameter Number for Voxel-Wise Data
                patient_feats = self.patient_info['Voxels'].iloc[patient_number]
                val_params *= patient_feats

        # ----------------------------------------------------------------------------------------------------------------------------

        # Patient Data & Label Access & Handling
        pX = self.get_patient(patient_number); py = self.params                                     # Patient Data Access
        if self.patient_id: py['Patient'] = self.patient_info['Patient'].iloc[patient_number]       # Patient ID Label Handling
        print(pX.shape); print(py.shape)
        if self.voxel_wise: pX, py = self.convert(pX, py)                                           # 1D Image to Voxel Data Conversion
        print(pX.shape); print(py.shape)

        # Patient Data Pre-Processing
        # ...

        # Patient Dataset Splitting into Training & Validation Sets
        pX_train, pX_val, py_train, py_val = train_test_split(  pX, py,
                                                                test_size = val_params,
                                                                shuffle = sample_shuffle,
                                                                random_state = 42)
        return pX_train, pX_val, py_train, py_val       

    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Splitting Function
    def split(
        self,
        settings: argparse.ArgumentParser
    ):

        # Patient Number Variable Logging
        assert(0 < settings.test_patients <= self.num_patients      # Limits for Number of Test Set Patients
        ), f"ERROR: Test Patient Number not Supported!"
        self.train_patients = self.num_patients - settings.test_patients    # Number of Patients to be used in the Training Set
        self.test_patients = settings.test_patients                         # Number of Patients to be used in the Test Set
        self.batch_size = settings.batch_size                               # Sample Batch Size Variable
        self.patient_shuffle = settings.patient_shuffle                     # Ability to Shuffle the Patients that compose both Training / Validation and Test Datasets
        self.sample_shuffle = settings.sample_shuffle                       # Ability to Shuffle the Samples inside both Training / Validation and Test Datasets
        self.num_workers = settings.num_workers                             # Number of Workers in the Usage of DataLoaders
        self.progress = True                                                # Control Boolean Value for Progress Saving (Data can only be saved if Split)
        
        # ----------------------------------------------------------------------------------------------------------------------------
        
        # Patient Shuffling Feature
        if(self.patient_shuffle): self.patient_info = self.patient_info.iloc[np.random.permutation(len(self.patient_info))]

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        if(settings.percentage):
            assert(0 < settings.train_params <= 100                         # Percentage Limits for Number of Training Set's Parameters
            ), f"ERROR: Training Set's Parameter Number not Supported!"
            self.trainTrain_params = settings.train_params                  # Percentage Value for Training Set's Training Parameters
            self.trainVal_params = 100 - settings.train_params              # Percentage Value for Training Set's Validation Parameters
            assert(0 < settings.test_params <= 100                          # Percentage Limits for Number of Test Set's Parameters
            ), f"ERROR: Test Set's Parameter Number not Supported!"
            self.testTrain_params = settings.test_params                    # Percentage Value for Test Set's Training Parameters
            self.testVal_params = 100 - settings.test_params                # Percentage Value for Test Set's Validation Parameters

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        else:
            assert(0 < settings.train_params <= self.num_params             # Numerical Limits for Number of Training Set's Parameters
            ), f"ERROR: Training Set's Parameter Number not Supported!"
            self.trainTrain_params = settings.train_params                  # Numerical Value for Training Set's Training Parameters
            self.trainVal_params = self.num_params - settings.train_params  # Numerical Value for Training Set's Validation Parameters
            assert(0 < settings.test_params <= self.num_params              # Numerical Limits for Number of Test Set's Parameters
            ), f"ERROR: Test Set's Parameter Number not Supported!"
            self.testTrain_params = settings.test_params                    # Numerical Value for Test Set's Training Parameters
            self.testVal_params = self.num_params - settings.test_params    # Numerical Value for Test Set's Validation Parameters

        # ----------------------------------------------------------------------------------------------------------------------------

        # Full MUDI Dataset Building
        with alive_bar( self.num_patients,
                        title = '1D MUDI Dataset',
                        force_tty = True) as progress_bar:

            # Training Set Scaffold Setting
            #self.train_set = dict.fromkeys(('X_train', 'X_val', 'y_train', 'y_val'))
            if self.voxel_wise:
                X_train = np.empty(list(np.array((0, 1))))
                X_val = np.empty(list(np.array((0, 1))))
            else:
                X_train = np.empty(list(np.array((0, settings.num_voxels))))
                X_val = np.empty(list(np.array((0, settings.num_voxels))))
            y_train = np.empty([0, self.num_labels]); y_val = np.empty([0, self.num_labels])

            # Training Set Building / Training Patient Loop
            for p in range(self.train_patients):

                # Training Patient Data Access & Treatment
                progress_bar.text = f"\n-> Training Set | Patient {self.patient_info['Patient'].iloc[p]}"
                pX_train, pX_val, py_train, py_val = self.split_patient(patient_number = p,
                                                                        train_params = self.trainTrain_params,
                                                                        percentage = settings.percentage,
                                                                        sample_shuffle = self.sample_shuffle)
                X_train = np.concatenate((X_train, pX_train), axis = 0); X_val = np.concatenate((X_val, pX_val), axis = 0)
                y_train = np.concatenate((y_train, py_train), axis = 0); y_val = np.concatenate((y_val, py_val), axis = 0)
                print(X_train.shape)
                time.sleep(0.01); progress_bar()
            
            # Training DataLoader Construction
            #self.train_set['X_train'] = X_train; self.train_set['X_val'] = X_val
            #self.train_set['y_train'] = y_train; self.train_set['y_val'] = y_val
            self.trainTrainLoader = DataLoader(TensorDataset(   torch.Tensor(X_train),
                                                                torch.Tensor(y_train)),
                                                                num_workers = self.num_workers,
                                                                batch_size = self.batch_size, shuffle = False)
            self.trainValLoader = DataLoader(TensorDataset(     torch.Tensor(X_val),
                                                                torch.Tensor(y_val)),
                                                                num_workers = self.num_workers,
                                                                batch_size = self.batch_size, shuffle = False)
            del X_train, X_val, y_train, y_val, pX_train, pX_val, py_train, py_val

    ##############################################################################################
    # ------------------------------------- Saving & Loading -------------------------------------
    ##############################################################################################

    # Dataset Saving Function
    def save(self):
        if self.progress:

            # Full Dataset Saving
            f = open(f'{self.save_folderpath}/Vertical 1D MUDI (Version {self.version})', 'wb')
            pickle.dump(self, f); f.close

            # Dataset Loader Saving
            torch.save(self.trainTrainLoader, f"{self.save_folderpath}/1D TrainTrainLoader (V{self.version}).pkl")
            torch.save(self.trainValLoader, f"{self.save_folderpath}/1D TrainValLoader (V{self.version}).pkl")
            #torch.save(self.testTrainLoader, f"{self.save_folderpath}/1D TestTrainLoader (V{self.version}).pkl")
            #torch.save(self.testValLoader, f"{self.save_folderpath}/1D TestValLoader (V{self.version}).pkl")
            torch.save(self.scaler, f"{self.save_folderpath}/1D Label Scaler (V{self.version}).pkl")
    
    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loading Function
    def load(
        path: Path,
        version: int = 0,
    ):
        f = open(f'{path}/Vertical 1D MUDI (Version {version})', 'rb')
        mudi = pickle.load(f)
        f.close
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
    
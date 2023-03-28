# Library Imports
import os
import pickle
import psutil
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from typing import Literal, Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Vertical 1D MUDI Dataset Initialization Class
class v1DMUDI(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        patient_folderpath: Path,       # Path for Folder Containing Patient Data Files
        param_filepath: Path,           # Path for File containing all 1344 Parameter Settings Combination
        info_filepath: Path,            # Path for List of Patients and Corresponding Image Size inside Full Dataset
    ):

        # Parameter Value Access
        super(v1DMUDI).__init__()
        self.params = pd.read_excel(param_filepath)             # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                  # Total Number of Parameters in Dataset

        # Patient Information Access
        self.patient_folderpath = patient_folderpath
        self.patient_info = pd.read_csv(info_filepath)          # List of Patients and Corresponding IDs & Image Sizes inside Full Dataset
        self.patient_info = self.patient_info[:-1]              # Eliminating the Last Row containing Useless Information from the Patient Information
        self.num_patients = self.patient_info.shape[0]          # Number of Patients inside Full Dataset


    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # 1D Image Pre-Processing Function
    def pre_process(
        self,
        img: pd.DataFrame,
    ):

        # Input Variable Assertions
        assert(img.ndim == 2), "ERROR: Input Image Shape not Supported! (2D Arrays only)"
        assert(self.pre_shape < img.shape[1]), "ERROR: Convolution Layer Size must be smaller than Original Image's no. of Voxels!"

        # Dimensionality Reduction
        img = np.array(img.values)
        img_final = self.pca.fit_transform(img)
        return pd.DataFrame(img_final)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Patient Data Access & Splitting Function
    def split_patient(
        self,
        patient_number: int,                # Number for the Patient File being Read and Acquired (in Order)
        train_params: int = 500,            # Number / Percentage of Parameters to be used in the Training Section of the Patient
        percentage: bool = False,           # Control Variable for the Usage of Percentage Values in train_params
        sample_shuffle: bool = False,       # Ability to Shuffle the Samples inside both Training and Validation Datasets
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

        # ----------------------------------------------------------------------------------------------------------------------------

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

        # ----------------------------------------------------------------------------------------------------------------------------

        # Patient Dataset Splitting into Training & Validation Sets
        py = self.params; py['Patient'] = patient_id                # Patient Data Label Handling
        pX = self.pre_process(pX)                                   # Patient Data Preprocessing
        pX_train, pX_val, py_train, py_val = train_test_split(  pX, py,
                                                                test_size = val_params,
                                                                shuffle = sample_shuffle,
                                                                random_state = 42)
        return pX_train, pX_val, py_train, py_val
        

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Dataset Splitting Function
    def split(
        self,
        test_patients: int = 1,             # Number of Patients to be used in the Test Set
        train_params: int = 500,            # Number / Percentage of Parameters for the Training of the Training Set
        test_params: int = 20,              # Number / Percentage of Parameters for the Training of the Test Set
        pre_shape: int = 1200,              # Intermediate Dataset Shape as of Pre-Processing
        percentage: bool = False,           # Control Variable for the Usage of Percentage Values in train_params
        patient_shuffle: bool = False,      # Ability to Shuffle the Patients that compose both Training / Validation and Test Datasets
        sample_shuffle: bool = False,       # Ability to Shuffle the Samples inside both Training / Validation and Test Datasets
    ):

        # Patient Number Variable Logging
        assert(0 < test_patients <= self.num_patients               # Limits for Number of Test Set Patients
        ), f"ERROR: Test Patient Number not Supported!"
        self.train_patients = self.num_patients - test_patients     # Number of Patients to be used in the Training & Validation Sets
        self.test_patients = test_patients                          # Number of Patients to be used in the Test Sets
        self.pre_shape = pre_shape

        # Pre-Processing Dimensionality Reduction + Patient Shuffling Feature
        self.pca = PCA(n_components = self.pre_shape)
        if(patient_shuffle):
            self.patient_info = self.patient_info.iloc[np.random.permutation(len(self.patient_info))]

        # ----------------------------------------------------------------------------------------------------------------------------

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        if(percentage):
            assert(0 < train_params <= 100                              # Percentage Limits for Number of Training Set's Parameters
            ), f"ERROR: Training Set's Parameter Number not Supported!"
            self.trainTrain_params = train_params                       # Percentage Value for Training Set's Training Parameters
            self.trainVal_params = 100 - train_params                   # Percentage Value for Training Set's Validation Parameters
            assert(0 < test_params <= 100                               # Percentage Limits for Number of Test Set's Parameters
            ), f"ERROR: Test Set's Parameter Number not Supported!"
            self.testTrain_params = test_params                         # Percentage Value for Test Set's Training Parameters
            self.testVal_params = 100 - test_params                     # Percentage Value for Test Set's Validation Parameters

        # Computation of Training & Validation Parameter Numbers (Percentage Input)
        else:
            assert(0 < train_params <= self.num_params                  # Numerical Limits for Number of Training Set's Parameters
            ), f"ERROR: Training Set's Parameter Number not Supported!"
            self.trainTrain_params = train_params                       # Numerical Value for Training Set's Training Parameters
            self.trainVal_params = self.num_params - train_params       # Numerical Value for Training Set's Validation Parameters
            assert(0 < test_params <= self.num_params                   # Numerical Limits for Number of Test Set's Parameters
            ), f"ERROR: Test Set's Parameter Number not Supported!"
            self.testTrain_params = test_params                         # Numerical Value for Test Set's Training Parameters
            self.testVal_params = self.num_params - test_params         # Numerical Value for Test Set's Validation Parameters

        # ----------------------------------------------------------------------------------------------------------------------------

        # Training & Validation Sets Building
        self.train_set = dict.fromkeys(('X_train', 'X_val', 'y_train', 'y_val'))
        X_train = np.empty([0, self.pre_shape]); X_val = np.empty([0, self.pre_shape])
        y_train = np.empty([0, 7]); y_val = np.empty([0, 7])
        for p in range(self.train_patients):

            # Training Patient Data Access & Treatment
            print(f"Adding Patient {self.patient_info['Patient'].iloc[p]}'s Data to the Training Set...")       # Display of the Patient being Added to the Test Set
            pX_train, pX_val, py_train, py_val = self.split_patient(patient_number = p,
                                                                    train_params = self.trainTrain_params,
                                                                    percentage = percentage,
                                                                    sample_shuffle = sample_shuffle)
            X_train = np.concatenate((X_train, pX_train), axis = 0); X_val = np.concatenate((X_val, pX_val), axis = 0)
            y_train = np.concatenate((y_train, py_train), axis = 0); y_val = np.concatenate((y_val, py_val), axis = 0)
        
        self.train_set['X_train'] = pd.DataFrame(X_train); self.train_set['X_val'] = pd.DataFrame(X_val)
        self.train_set['y_train'] = pd.DataFrame(y_train); self.train_set['y_val'] = pd.DataFrame(y_val)
        del X_train, X_val, y_train, y_val, pX_train, pX_val, py_train, py_val

        # ----------------------------------------------------------------------------------------------------------------------------

        # Test Set Building
        self.test_set = dict.fromkeys(('X_train', 'X_val', 'y_train', 'y_val'))                             # Creation of Empty Dictionary to Fit Patient Data
        X_train = np.empty([0, self.pre_shape]); X_val = np.empty([0, self.pre_shape])
        y_train = np.empty([0, 7]); y_val = np.empty([0, 7])
        for p in range(self.train_patients, self.train_patients + self.test_patients):

            # Training Patient Data Access & Treatment
            print(f"Adding Patient {self.patient_info['Patient'].iloc[p]}'s Data to the Test Set...")       # Display of the Patient being Added to the Test Set
            pX_train, pX_val, py_train, py_val = self.split_patient(patient_number = p,
                                                                    train_params = self.testTrain_params,
                                                                    percentage = percentage,
                                                                    sample_shuffle = sample_shuffle)
            X_train = np.concatenate((X_train, pX_train), axis = 0); X_val = np.concatenate((X_val, pX_val), axis = 0)
            y_train = np.concatenate((y_train, py_train), axis = 0); y_val = np.concatenate((y_val, py_val), axis = 0)
        
        self.test_set['X_train'] = pd.DataFrame(X_train); self.test_set['X_val'] = pd.DataFrame(X_val)
        self.test_set['y_train'] = pd.DataFrame(y_train); self.test_set['y_val'] = pd.DataFrame(y_val)
        del X_train, X_val, y_train, y_val, pX_train, pX_val, py_train, py_val

        # ----------------------------------------------------------------------------------------------------------------------------

        # Split Datasets' Content Report
        if(percentage):
            print(tabulate([[self.train_patients, f"{(self.trainTrain_params / 100) * self.num_params} ({self.trainTrain_params}%)", f"{(self.trainVal_params / 100) * self.num_params} ({self.trainVal_params}%)"],
                            [self.test_patients, f"{(self.testTrain_params / 100) * self.num_params} ({self.testTrain_params}%)", f"{(self.testVal_params / 100) * self.num_params} ({self.testVal_params}%)"]],
                            headers = ['No. Patients', 'Training Parameters', 'Validation Parameters'],
                            showindex = ['Training Set', 'Test Set'], tablefmt = 'fancy_grid'))
        else:
            print(tabulate([[self.train_patients, f"{self.trainTrain_params} ({np.round((self.trainTrain_params / self.num_params) * 100, 2)}%)", f"{self.trainVal_params} ({np.round((self.trainVal_params / self.num_params) * 100, 2)}%)"],
                            [self.test_patients, f"{self.testTrain_params} ({np.round(self.testTrain_params / self.num_params, 2)}%)", f"{self.testVal_params} ({np.round(self.testVal_params / self.num_params, 2)}%)"]],
                            headers = ['No. Patients', 'Training Parameters', 'Validation Parameters'],
                            showindex = ['Training Set', 'Test Set'], tablefmt = 'fancy_grid'))

    
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Dataset Saving Function
    def save(
        self,
        path: Path,
        version: int = 0,
    ):
        f = open(f'{path}/Vertical 1D MUDI (Version {version})', 'wb')
        pickle.dump(self, f)
        f.close

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

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
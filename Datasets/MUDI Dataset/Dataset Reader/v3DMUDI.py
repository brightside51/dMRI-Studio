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

# Vertical 3D MUDI Dataset Initialization Class
class v3DMUDI(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):

        # Parameter Value Access
        super(v3DMUDI).__init__()
        self.params = pd.read_excel(settings.param_filepath)    # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                  # Total Number of Parameters in Dataset
        self.version = settings.version

        # Patient Information Access
        self.patient_folderpath = settings.patient_folderpath
        self.mask_folderpath = settings.mask_folderpath
        self.save_folderpath = settings.save_folderpath
        self.patient_info = pd.read_csv(settings.info_filepath)     # List of Patients and Corresponding IDs & Image Sizes inside Full Dataset
        self.patient_info = self.patient_info[:-1]                  # Eliminating the Last Row containing Useless Information from the Patient Information
        self.num_patients = self.patient_info.shape[0]              # Number of Patients inside Full Dataset
        self.dim = settings.dim                                     # Sample Dimensionality Variable
        self.progress = False                                       # Control Boolean Value for Progress Saving (Data can only be saved if Split)
        
        # Dataset Label Handling Settings
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
    # ----------------------------------- Image Pre-Processing -----------------------------------
    ##############################################################################################

    # Pre-Processing Alternative Method: Interpolation
    def prep_interpolation(
        self,
        data: np.array,
    ):

        # Patient Data Interpolation
        assert(data.ndim >= 3), "ERROR: Pre-Processing Input Data has the Wrong Dimmensions"
        final_data = np.empty((data.shape[0], self.final_shape[0], self.final_shape[1], self.final_shape[2]))
        ratio = np.divide(data.shape[1::], self.final_shape)
        for sample, x, y, z in itertools.product(   range(data.shape[0]),
                                                    range(self.final_shape[0]),
                                                    range(self.final_shape[1]),
                                                    range(self.final_shape[2])):
            final_data[sample][x][y][z] = data[sample][int(x * ratio[0])][int(y * ratio[1])][int(z * ratio[2])]
        return final_data
    
    # ----------------------------------------------------------------------------------------------------------------------------

    # Pre-Processing Alternative Method: Zero-Padding
    def prep_zeroPadding(
        self,
        data: np.array,
    ):

        # Input Data Assertions
        assert(data.ndim >= 3), "ERROR: Pre-Processing Input Data has the Wrong Dimmensions"
        assert(len(self.final_shape) == 3), "ERROR: Pre-Processing Output has the Wrong Dimmensions"
        assert(len(np.where((self.final_shape >= data.shape[1::]) == False)[0]) == 0
        ), "ERROR: Pre-Processed Output Data Shape < Original Input Data Shape"
        #assert(len(np.where((self.final_shape[3 - self.dim::] >= data.shape[4 - self.dim::]) == False)[0]) == 0
        #), "ERROR: Pre-Processed Output Data Shape < Original Input Data Shape"

        # Zero-Padding Implementation
        padding = (np.hstack((0, np.subtract(self.final_shape, data.shape[1::]))) / 2).astype(np.float32)
        padding = padding.reshape((1, -1)).T + np.array([0, 0])
        padding[:, 0] = np.ceil(padding[:, 0]); padding[:, 1] = np.floor(padding[:, 1])
        data = np.pad(data, padding.astype(np.int32), 'constant')
        return data

    # ----------------------------------------------------------------------------------------------------------------------------

    # Pre-Processing Alternative Method: Convolutional Layer
    """
    class preProcess(nn.Module):

        # Constructor / Initialization Function
        def __init__(
            self,
            data: pd.DataFrame,
            pre_shape: int = 512,
        ):

            # Parameter Value Access
            super(preProcess).__init__()
            self.data = data.T
            assert(data.ndim == 4), "ERROR: Input Image Shape not Supported! (4D Arrays only)"
            assert(self.pre_shape < (data.shape[1] * data.shape[2] * data.shape[3])
            ), "ERROR: Convolution Layer Size must be smaller than Original Image's no. of Voxels!"

            # Convolutional Layer Structure
            print(self.data.shape)
            out = self.conv_layer(data.shape[0], )
            print(out.shape)

        # ----------------------------------------------------------------------------------------------------------------------------
        
        # Convolutional Layer 
        def conv_layer(
            self,
            in_channels: int,
            out_channels:int,
        ):

            return nn.Sequential(
                nn.Conv3d(  in_channels, out_channels,
                            kernel_size = (3, 3, 3),
                            padding = 0),
                nn.MaxPool3d((2, 2, 2)),
                nn.Dropout(p = 0.15), )
                #nn.LeakyReLU(),
                #nn.MaxPool3d((2, 2, 2)),)
    """
        
    ##############################################################################################
    # -------------------------------------- Label Handling --------------------------------------
    ##############################################################################################

    # 3D to 2D Data Conversion Function
    def convert(
        self,
        data: np.ndarray,           # 3D Data Array
        label: pd.DataFrame
    ):

        # Selected Slice Variable Logging
        assert(0 < self.num_slices <= data.shape[1]
        ), "ERROR: Number of Selected Slices in 2D Conversion not Supported"
        inf_thresh = int(np.ceil((data.shape[1] - self.num_slices) / 2))
        sup_thresh = data.shape[1] - int(np.floor((data.shape[1] - self.num_slices) / 2))

        # [num_sample, num_slice, img_size1, img_size2] ->
        # -> [num_sample * num_slice, 1, imgsize1, imgsize2]

        # Conversion from 3D Image Data to 2D Image Data
        data = data[:, inf_thresh : sup_thresh, :, :]
        data = data.reshape((data.shape[0] * (sup_thresh - inf_thresh),
                            1, data.shape[2], data.shape[3]))
        label = label.iloc[np.repeat(np.arange(len(label)), self.num_slices)]
        return data, label

    # ----------------------------------------------------------------------------------------------------------------------------

    # [Not Used] Cartesian to Polar Coordinate Conversion Function
    def cart2polar(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        z: pd.DataFrame
    ):

        # Cartesian to Polar Coordinate Conversion
        r = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
        theta = np.arctan2(z, np.sqrt((x ** 2) + (y ** 2))) * 180 / np.pi
        phi = np.arctan2(y, x) * 180 / np.pi

    # ----------------------------------------------------------------------------------------------------------------------------

    # Label Scaler Download & Reverse Transformation
    def label_unscale(
        path: Path,
        version: int,
        y: np.array or pd.DataFrame
    ):

        # 
        scaler = torch.load(f"{path}/Label Scaler (V{version}).pkl")
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
        mask_filepath = Path(f"{self.mask_folderpath}/p{patient_id}.nii")                           # Mask Filepath from detailed Folder
        assert(patient_filepath.exists()                                                            # Assertion for the Existence of Patient File in said Folder
        ), f"Filepath for Patient {patient_id} is not in the Dataset!"
        assert(mask_filepath.exists()                                                               # Assertion for the Existence of Mask File in said Folder
        ), f"Filepath for Mask {patient_id} is not in the Dataset!"
        file_size = os.path.getsize(patient_filepath)                                               # Memory Space occupied by Patient File
        mask_size = os.path.getsize(mask_filepath)                                                  # Memory Space occupied by Mask File
        available_memory = psutil.virtual_memory().available                                        # Memory Space Available for Computation
        assert(available_memory >= (file_size + mask_size)                                          # Assertion for the Existence of Available Memory Space
        ), f"ERROR: Dataset requires {file_size + mask_size}b, but only {available_memory}b is available!"
        pX = pd.read_csv(patient_filepath); del pX['Unnamed: 0']                                    # Full Patient Data
        pMask = load_img(mask_filepath)                                                             # Patient Mask Data
        pX = unmask(pX, pMask); pX = pX.get_fdata()                                                 # Unmasking of Full Patient Data
        del available_memory, mask_size, file_size
        pX = np.transpose(pX, (3, 2, 0, 1))                                                         # Full Patient Data Reshapping
        return pX, pMask
    
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
            if self.dim == 2: val_params *= self.num_slices             #

        # ----------------------------------------------------------------------------------------------------------------------------

        # Patient Data Access & Label Handling
        pX, pMask = self.get_patient(patient_number); py = self.params                              # Patient Data Access
        if self.patient_id: py['Patient'] = self.patient_info['Patient'].iloc[patient_number]       # Patient ID Label Handling

        # Patient Data Pre-Processing
        if(self.pre_processing == 'Zero Padding'): pX = self.prep_zeroPadding(pX)                   # Zero Padding Pre-Processing
        elif(self.pre_processing == 'Interpolation'): pX = self.prep_interpolation(pX)              # Interpolation Pre-Processing
        #elif(self.pre_processing == 'CNN'): pX = self.prep_cnn(pX)                                 # CNN Pre-Processing
        if self.dim == 2: pX, py = self.convert(pX, py)                                             # 3D to 2D Conversion

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
        assert(settings.pre_processing == 'Zero Padding' or settings.pre_processing == 'Interpolation' or settings.pre_processing == 'CNN'
        ), "ERROR: Pre-Processing Method not Supported!"
        assert(len(settings.img_shape) == 3), "ERROR: Pre-Processing Output has the Wrong Dimmensions"
        self.train_patients = self.num_patients - settings.test_patients    # Number of Patients to be used in the Training Set
        self.test_patients = settings.test_patients                         # Number of Patients to be used in the Test Set
        self.batch_size = settings.batch_size                               # Sample Batch Size Variable
        self.pre_processing = settings.pre_processing                       # Chosen Pre-Processing Method
        self.final_shape = settings.img_shape                               # Pre-Processed 3D Image Shape
        self.num_slices = settings.num_slices                               # Number of Selected Slices in 2D Conversion
        self.patient_shuffle = settings.patient_shuffle                     # Ability to Shuffle the Patients that compose both Training / Validation and Test Datasets
        self.sample_shuffle = settings.sample_shuffle                       # Ability to Shuffle the Samples inside both Training / Validation and Test Datasets
        self.num_workers = settings.num_workers                             # Number of Workers in the Usage of DataLoaders
        self.progress = True                                        # Control Boolean Value for Progress Saving (Data can only be saved if Split)

        # Patient Shuffling Feature
        if(self.patient_shuffle): self.patient_info = self.patient_info.iloc[np.random.permutation(len(self.patient_info))]

        # ----------------------------------------------------------------------------------------------------------------------------

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
                        title = f'{self.dim}D MUDI Dataset',
                        force_tty = True) as progress_bar:
            
            # Training Set Scaffold Setting
            self.train_set = dict.fromkeys(('X_train', 'X_val', 'y_train', 'y_val'))
            if self.dim == 2:
                #self.final_shape = np.array((1, self.final_shape[1], self.final_shape[2]))
                X_train = np.empty(list(np.hstack((0, np.array((1, self.final_shape[1], self.final_shape[2]))))))
                X_val = np.empty(list(np.hstack((0, np.array((1, self.final_shape[1], self.final_shape[2]))))))
            else:
                X_train = np.empty(list(np.hstack((0, self.final_shape))))
                X_val = np.empty(list(np.hstack((0, self.final_shape))))
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
                time.sleep(0.01); progress_bar()
            
            # Training DataLoader Construction
            self.train_set['X_train'] = X_train; self.train_set['X_val'] = X_val
            self.train_set['y_train'] = y_train; self.train_set['y_val'] = y_val
            self.trainTrainLoader = DataLoader(TensorDataset(   torch.Tensor(X_train),
                                                                torch.Tensor(y_train)),
                                                                batch_size = self.batch_size, shuffle = False)
            self.trainValLoader = DataLoader(TensorDataset(     torch.Tensor(X_val),
                                                                torch.Tensor(y_val)),
                                                                batch_size = self.batch_size, shuffle = False)
            del X_train, X_val, y_train, y_val, pX_train, pX_val, py_train, py_val

            # ----------------------------------------------------------------------------------------------------------------------------

            # Test Set Scaffold Setting
            self.test_set = dict.fromkeys(('X_train', 'X_val', 'y_train', 'y_val'))                             # Creation of Empty Dictionary to Fit Patient Data
            if self.dim == 2:
                #self.final_shape = np.array((1, self.final_shape[1], self.final_shape[2]))
                X_train = np.empty(list(np.hstack((0, np.array((1, self.final_shape[1], self.final_shape[2]))))))
                X_val = np.empty(list(np.hstack((0, np.array((1, self.final_shape[1], self.final_shape[2]))))))
            else:
                X_train = np.empty(list(np.hstack((0, self.final_shape))))
                X_val = np.empty(list(np.hstack((0, self.final_shape))))
            y_train = np.empty([0, self.num_labels]); y_val = np.empty([0, self.num_labels])

            # Test Set Building / Test Patient Loop
            for p in range(self.train_patients, self.train_patients + self.test_patients):

                # Test Patient Data Access & Treatment
                progress_bar.text = f"-> Test Set | Patient {self.patient_info['Patient'].iloc[p]}"
                pX_train, pX_val, py_train, py_val = self.split_patient(patient_number = p,
                                                                        train_params = self.testTrain_params,
                                                                        percentage = settings.percentage,
                                                                        sample_shuffle = self.sample_shuffle)
                X_train = np.concatenate((X_train, pX_train), axis = 0); X_val = np.concatenate((X_val, pX_val), axis = 0)
                y_train = np.concatenate((y_train, py_train), axis = 0); y_val = np.concatenate((y_val, py_val), axis = 0)
                time.sleep(0.01); progress_bar()

            # Test DataLoader Construction
            self.test_set['X_train'] = X_train; self.test_set['X_val'] = X_val
            self.test_set['y_train'] = y_train; self.test_set['y_val'] = y_val
            self.testTrainLoader = DataLoader(TensorDataset(torch.Tensor(X_train),
                                                            torch.Tensor(y_train)),
                                                            num_workers = self.num_workers,
                                                            batch_size = self.batch_size, shuffle = False)
            self.testValLoader = DataLoader(TensorDataset(  torch.Tensor(X_val),
                                                            torch.Tensor(y_val)),
                                                            num_workers = self.num_workers,
                                                            batch_size = self.batch_size, shuffle = False)
            del X_train, X_val, y_train, y_val, pX_train, pX_val, py_train, py_val

        # ----------------------------------------------------------------------------------------------------------------------------

        # Split Datasets' Content Report
        if settings.percentage:
            print(tabulate([[self.train_patients, f"{(self.trainTrain_params / 100) * self.num_params} ({self.trainTrain_params}%)", f"{(self.trainVal_params / 100) * self.num_params} ({self.trainVal_params}%)"],
                            [self.test_patients, f"{(self.testTrain_params / 100) * self.num_params} ({self.testTrain_params}%)", f"{(self.testVal_params / 100) * self.num_params} ({self.testVal_params}%)"]],
                            headers = ['No. Patients', 'Training Parameters', 'Validation Parameters'],
                            showindex = ['Training Set', 'Test Set'], tablefmt = 'fancy_grid'))
        else:
            print(tabulate([[self.train_patients, f"{self.trainTrain_params} ({np.round((self.trainTrain_params / self.num_params) * 100, 2)}%)", f"{self.trainVal_params} ({np.round((self.trainVal_params / self.num_params) * 100, 2)}%)"],
                            [self.test_patients, f"{self.testTrain_params} ({np.round(self.testTrain_params / self.num_params, 2)}%)", f"{self.testVal_params} ({np.round(self.testVal_params / self.num_params, 2)}%)"]],
                            headers = ['No. Patients', 'Training Parameters', 'Validation Parameters'],
                            showindex = ['Training Set', 'Test Set'], tablefmt = 'fancy_grid'))

    ##############################################################################################
    # ------------------------------------- Saving & Loading -------------------------------------
    ##############################################################################################

    # Dataset Saving Function
    def save(self):
        if self.progress:

            # Full Dataset Saving
            f = open(f'{self.save_folderpath}/Vertical {self.dim}D MUDI (Version {self.version})', 'wb')
            pickle.dump(self, f); f.close

            # Dataset Loader Saving
            torch.save(self.trainTrainLoader, f"{self.save_folderpath}/{self.dim}D TrainTrainLoader (V{self.version}).pkl")
            torch.save(self.trainValLoader, f"{self.save_folderpath}/{self.dim}D TrainValLoader (V{self.version}).pkl")
            torch.save(self.testTrainLoader, f"{self.save_folderpath}/{self.dim}D TestTrainLoader (V{self.version}).pkl")
            torch.save(self.testValLoader, f"{self.save_folderpath}/{self.dim}D TestValLoader (V{self.version}).pkl")
            torch.save(self.scaler, f"{self.save_folderpath}/Label Scaler (V{self.version}).pkl")
    
    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loading Function
    def load(
        path: Path,
        dim: int = 3,
        version: int = 0,
    ):
        f = open(f'{path}/Vertical {dim}D MUDI (Version {version})', 'rb')
        mudi = pickle.load(f)
        f.close
        return mudi

    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loader Loading Function
    def loader(
        path: Path,
        dim: int = 3,
        version: int = 0,
        set_: str = 'Train',
        mode_: str = 'Train',
    ):
        return torch.load(f"{path}/{dim}D {set_}{mode_}Loader (V{version}).pkl")

    ##############################################################################################
    # ----------------------------------- Sample Visualization -----------------------------------
    ##############################################################################################

    # 3D Interactive Plotting Function
    def plot(
        data,
        patient_number: int,
        sample_number: int,
        slice_number: int,
    ):

        # Patient Sample & Slice for Visualization
        img = data[sample_number]; img = img[slice_number]
        #img = data[slice_number, :, :, sample_number].T
        plt.figure(figsize = (10, 20)); plt.imshow(img, cmap = 'gray'); plt.axis('off')
        plt.title(f"Patient #{patient_number} | Sample #{sample_number} | Slice #{slice_number}")

    # ----------------------------------------------------------------------------------------------------------------------------

# Dataset Creation & Visualization Example
if __name__ == '__main__':

    # Dataset Initialization & Saving Example
    main_folderpath = '../../../Datasets/MUDI Dataset/'
    mudi = v3DMUDI( main_folderpath + 'Patient Data',
                    main_folderpath + 'Patient Mask',
                    main_folderpath + 'Raw Data/parameters_new.xlsx',
                    main_folderpath + 'Raw Data/header1_.csv',
                    dim = 2)

    # Data Access & Sample + Slice Slider Interactive Construction
    patient_number: int = 0; data, mask = mudi.get_patient(patient_number)
    sample_slider = IntSlider(value = 0, min = 0, max = data.shape[0], description = 'Sample', continuous_update = False)
    slice_slider = IntSlider(value = 0, min = 0, max = data.shape[1], description = 'Slice', continuous_update = False)
    interactive(mudi.plot, sample_number = sample_slider, slice_number = slice_slider)

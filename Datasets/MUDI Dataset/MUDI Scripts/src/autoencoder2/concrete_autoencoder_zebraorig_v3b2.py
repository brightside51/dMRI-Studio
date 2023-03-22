from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profiler import PassThroughProfiler
from torch import nn
import pandas as pd

from torch import reshape as tshape
from torch import cat as tcat
from torch import exp as texp
from torch import log as tlog
from torch import abs as tabs
from torch import erf as terf
from torch import sqrt as tsqrt
from torch import matmul as tmat

import sys
sys.path.insert(0,'/cubric/data/sapap9/v2b_runs_indices_maarten/src/autoencoder2')

from argparse2 import file_path
from logger import logger


class qmrizebra(nn.Module):
    def __init__(
        self,
        #pdrop, #mridata, 
        #input_size1: int,
        input_size2: int,
        output_size2: int,
        n_hidden_layers2: int,
        negative_slope: float = 0.2,
    ):
        super(qmrizebra, self).__init__()
        
        self.mridata = pd.read_csv('/home/sapap9/PythonCode/MUDI/data/parameters_new.txt', sep=" ", header=None)
        self.mridata.columns = ["x", "y", "z", "bval","TI","TD"]
        self.mridata.TD = self.mridata.TD - min(self.mridata.TD) # TD = TE - min(TE)
        self.mridata["TR"] = 7500.0
        
        self.npars = 7
        
        indices2 = np.arange(2 + n_hidden_layers2)
        data_indices2 = np.array([indices2[0], indices2[-1]])
        data2 = np.array([input_size2, output_size2])

        layer_sizes = np.interp(indices2, data_indices2, data2).astype(int)
        n_layers = len(layer_sizes)#+1
        
        # Construct the network
        layers = OrderedDict()
        for i in range(1, n_layers):
            n = i - 1
            layers[f"linear_{n}"] = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            layers[f"relu_{n}"] = nn.ReLU(True)
            if i == n_layers - 1:  # Last layer
                #layers[f"linear_{n}"] = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                layers[f"softplus_{n}"] = nn.Softplus()
            #else:
                

        logger.debug("encoder2 layers: %s", layers)

        self.encoder2 = nn.Sequential(layers)
        
        # Add learnable normalisation factors to convert output neuron activations to tissue parameters
        normlist = []
        for pp in range(output_size2):
            normlist.append(nn.Linear(1, 1, bias = False))
        self.sgmnorm = nn.ModuleList(normlist)
        
        ### Set the possible maximum and minimum value of the output parameters
        #adc_min = 0.1 # in um^2/ms -> limit in healthy brain is ~0.65 in WM, and 0.6 in dense tumours; more or less equivalent to dpar_min/dper_min; 0.5 changed to 0.1
        theta_min = 0.0 # in rad
        phi_min = 0.0 # in rad
        """dir1_min = -1.0 
        dir2_min = -1.0
        dir3_min = -1.0"""
        dpar_min = 0.01 # um^2/ms from Grussu et al.
        kperp_min = 0 # from Grussu et al. It should be dperp in um^2/ms , but to make sure dper <= dpar"""
        t2star_min = 0.01 # in ms (https://mri-q.com/why-is-t1--t2.html), a bit lower than T2 bound for proteins
        t1_min = 100 # in ms at 3T from Grussu et al.
        s0_min = 0.5 # from Grussu et al.
        #ie_min = 0.1 # to adjust the final value
        
        #adc_max = 3.5 # in um^2/ms -> limit for CSF in radiopaedia is 3.4
        theta_max = np.pi # in rad
        phi_max = 2*np.pi-0.01 # in rad, a bit lower than 2pi
        """dir1_max = 1.0 
        dir2_max = 1.0 
        dir3_max = 1.0"""
        dpar_max = 3.2 # um^2/ms from Grussu et al.
        kperp_max = 1 # from Grussu et al. It should be dperp in um^2/ms , but to make sure dper <= dpar"""
        t2star_max = 2000 # in ms, CSF value (https://mri-q.com/why-is-t1--t2.html)
        t1_max = 5000 # in ms, from the ZEBRA paper
        s0_max = 5 # from Grussu et al.
        #ie_max = 1.0 # to adjust the final value
        
        self.param_min = np.array([theta_min, phi_min, dpar_min, kperp_min, t2star_min, t1_min, s0_min])
        self.param_max = np.array([theta_max, phi_max, dpar_max, kperp_max, t2star_max, t1_max, s0_max])
        self.param_name = ['theta', 'phi', 'dpar', 'kperp', 't2star', 't1', 's0']
    
    def getnorm(self, x):
        """ Get the output from getneurons and normalise it
        
            u_out = mynet.getnorm(u_in)
            
            * mynet: initialised qmrizebra
            
            * u_in: Tensor storing the output neuronal activations
                    Nvoxels x Nparams_to_estimate for a mini-batch
                    
            * u:out = normalised u_in
        """
        
        if x.dim()==1:
            
            # 1D tensor corresponding to a voxel
            normt = torch.tensor(np.zeros((self.npars)))
            normt = normt.type_as(x)
            for pp in range(self.npars):
                bt = np.zeros((self.npars))
                bt[pp] = 1.0
                bt = torch.tensor(bt)
                bt = bt.type_as(x)
                #self.register_buffer("bt",torch.tensor(bt))
                con_one = torch.tensor([1.0])
                con_one = con_one.type_as(x)
                #self.register_buffer("con_one",torch.tensor([1.0])) 
                bt = self.sgmnorm[pp](con_one)*bt
                normt = normt + bt
                
            # Normalise
            normt = tabs(normt)
            x = x*normt
            
        elif x.dim()==2:
            
            # Tensor with Nvoxels x Nparams
            normt = torch.tensor(np.zeros([x.shape[0],self.npars]))
            normt = normt.type_as(x)
            for pp in range(self.npars):
                bt = np.zeros(([x.shape[0],self.npars]))
                bt[:,pp] = 1.0
                bt = torch.tensor(bt)
                bt = bt.type_as(x)
                con_one = torch.tensor([1.0])
                con_one = con_one.type_as(x)
                bt = self.sgmnorm[pp](con_one)*bt
                normt = normt + bt
        
            # Normalise
            normt = tabs(normt)
            x = x*normt
            
        else:
            raise RuntimeError('getnorm() only accepts 1D or 2D inputs')
            
        return x
    
    def getparams(self, x):
        """ Get tissue parameters from initialised qmrizebra
        
            p = mynet.getparams(x_in)
            
            * mynet: initialised qmrizebra
            
            * xin: Tensor storing MRI measurements (a voxel or a mini-batch)
                    voxels x Nmeasurements in a mini-batch
                    
            * p:   Tensor storing the predicted parameters
                   Nvoxels x Nparams_to_estimate for a mini-batch            
        """
        
        x = self.encoder2(x) # the last layer cannot be 0.0 due to the softplus
        
        ## Normalise neuronal activations in [0; 1.0] in 4 steps (A, B, C, D):
        x = tlog(x) # 1. Log activations
        con_two = torch.tensor([2.0])
        con_two = con_two.type_as(x)
        x = x - tlog(tlog(con_two)) # 2. We remove possible negative values considering that the minumum of x is log(2.0)
        x = self.getnorm(x) # 3. Normalisation using a multiplying learnable factor
        x = 2.0*(1.0 / (1.0 + texp(-x)) - 0.5) # 4. Sigmoid function
        
        ## Map normalised neuronal activations to MRI tissue parameter ranges \
        
        # Single voxels
        if x.dim()==1:
            for pp in range(0,self.npars):
                x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]
                
        # Mini-batch
        elif x.dim()==2:
            t_ones = np.ones((x.shape[0],1))
            
            max_val = torch.tensor( np.concatenate( ( self.param_max[0]*t_ones , self.param_max[1]*t_ones , self.param_max[2]*t_ones , self.param_max[3]*t_ones, self.param_max[4]*t_ones, self.param_max[5]*t_ones , self.param_max[6]*t_ones), axis=1  ) )
                             
            min_val = torch.tensor( np.concatenate( ( self.param_min[0]*t_ones , self.param_min[1]*t_ones , self.param_min[2]*t_ones , self.param_min[3]*t_ones, self.param_min[4]*t_ones, self.param_min[5]*t_ones , self.param_min[6]*t_ones), axis=1   ) )
            
            max_val = max_val.type_as(x)
            min_val = min_val.type_as(x)                 
            x = (max_val - min_val)*x + min_val
                             
        return x
    
    # Decoder: Estimate the signal from MRI parameters estimated with the encoder
    def getsignals(self, x):
        """ Use the ZEBRA model to obtain the diffusion signal.
        
            x_out = mynet.getsignals(p_in)
            
            * mynet: initialised qmrizebra
            
            * p_in:   Tensor storing the predicted parameters
                   Nvoxels x Nparams_to_estimate for a mini-batch 
            
            * x_out: Tensor storing the predicted MRI signals according to ZEBRA
                    Nvoxels x Nparams_to_estimate for a mini-batch 
        """
        mriseq = torch.transpose(torch.tensor(self.mridata.values),0,1)
        Nmeas = mriseq.shape[1]
                             
        ## Compute MRI signals from input parameter x (microstructural tissue parameters)
        #phi = torch.atan2(torch.sqrt(torch.square(mriseq[0,:])+ torch.square(mriseq[1,:])),mriseq[2,:])
        #theta = torch.atan2(mriseq[1,:],mriseq[0,:])#+np.pi
        x1 = mriseq[0,:]
        x2 = mriseq[1,:]
        x3 = mriseq[2,:]
        bval = mriseq[3,:]
        bval = bval / 1000.0 # from s/mm^2 to ms/um^2
        TI = mriseq[4,:]
        TD = mriseq[5,:]
        TR = mriseq[6,:]
        # we assume that b_delta = 1
        b_delta = 1.0
        
        if x.dim()==1:
            b_D = b_delta / 3.0 * bval * (x[2] - x[3]*x[2]) - bval / 3.0 * (x[3]*x[2] + 2.0 * x[2]) - bval * b_delta * (torch.square(torch.dot([x1,x2,x3],[torch.cos(x[1])*torch.sin(x[0]),torch.sin(x[0])*torch.sin(x[1]),torch.cos(x[0])])) * (x[2] - x[3]*x[2]))
            s_tot = x[6] * texp(b_D) * tabs(1.0 - 2.0 * texp(-TI/x[5]) + texp(-TR/x[5])) * texp(-TD/x[4])
            #s_tot = x[7] * texp(-b_D) * tabs(1.0 - 2.0 * texp(-TI/x[6]) + texp(-TR/x[6])) * texp(-TD/x[5])
            #s_tot = x[3] * (1 - 2 * texp(-TI/x[2]) + texp(-TR/x[2])) * texp(-bval * x[0]) * texp(-TD/x[1])
            #s_tot = x[4] * tabs(1 - 2 * texp(-TI/x[3]) + texp(-TR/x[3])) * texp(-bval * x[0] * x[1]) * texp(-TD/x[2]) *terf(tsqrt(bval*(x[0] - x[0]*x[1])))/tsqrt(bval*(x[0] - x[0]*x[1]))

            #s_tot = x[7] * texp(-b_D) * tabs(1.0 - 2.0 * texp(-TI/x[6]) + texp(-TR/x[6])) * texp(-TD/x[5]) * terf(tsqrt(bval*(x[3] - x[4]*x[3])))/tsqrt(bval*(x[3] - x[4]*x[3]))
            #x = torch.abs(s_tot*1.0)
            x = 1.0*s_tot
            return x
            
        elif x.dim()==2:
            Nvox = x.shape[0]
            #theta = tshape(theta, (1,Nmeas))
            #phi = tshape(phi, (1,Nmeas))
            x1 = tshape(x1, (1,Nmeas))
            x2 = tshape(x2, (1,Nmeas))
            x3 = tshape(x3, (1,Nmeas))
            bval = tshape(bval, (1,Nmeas))
            TI = tshape(TI, (1,Nmeas))
            TD = tshape(TD, (1,Nmeas))
            TR = tshape(TR, (1,Nmeas))
            
            b_D = b_delta / 3.0 * tmat(tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)).type_as(x), bval.type_as(x))
            b_D = b_D - 1.0 / 3.0 * tmat(tshape(x[:,3]*x[:,2] + 2.0 * x[:,2], (Nvox,1)).type_as(x), bval.type_as(x))
            #angles_dprod = tmat(tshape(x[:,0], (Nvox,1)), theta.type_as(x)) + tmat(tshape(x[:,1], (Nvox,1)), phi.type_as(x))
            angles_dprod = tmat(tshape(torch.sin(x[:,0])*torch.cos(x[:,1]), (Nvox,1)), x1.type_as(x)) + tmat(tshape(torch.sin(x[:,1])*torch.sin(x[:,0]), (Nvox,1)), x2.type_as(x)) + tmat(tshape(torch.cos(x[:,0]), (Nvox,1)), x3.type_as(x))
            b_D_term3 = tmat(tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)).type_as(x), bval.type_as(x)) * torch.square(angles_dprod).type_as(x)
            b_D = b_D - b_delta * b_D_term3

            """b_D = b_delta / 3.0 * tmat(tshape(x[:,4] - x[:,3]*x[:,4], (Nvox,1)).type_as(x), bval.type_as(x))
            b_D = b_D - 1.0 / 3.0 * tmat(tshape(x[:,4]*x[:,3] + 2.0 * x[:,3], (Nvox,1)).type_as(x), bval.type_as(x))

            angles_dprod = tmat(tshape(x[:,0], (Nvox,1)), x1.type_as(x)) + tmat(tshape(x[:,1], (Nvox,1)), x2.type_as(x)) + tmat(tshape(x[:,2], (Nvox,1)), x3.type_as(x))
            
            b_D_term3 = tmat(tshape(x[:,3] - x[:,4]*x[:,3], (Nvox,1)).type_as(x), bval.type_as(x)) *  torch.square(angles_dprod)#.type_as(x)
            b_D = b_D - b_delta * b_D_term3"""
            
            #ones_ten = Tensor(np.ones((1,Nmeas)))
            ones_ten = torch.ones(1,Nmeas)
            ones_ten = ones_ten.type_as(x)
            #s_tot = tmat(tshape(x[:,6],(Nvox,1)), Tensor(np.ones((1,Nmeas))).to(device))
            s_tot = tmat(tshape(x[:,6],(Nvox,1)), ones_ten)
            #s_tot = tmat(tshape(x[:,3],(Nvox,1)), ones_ten)
            
            s_tot = s_tot * texp(b_D)#.type_as(x)
            s_tot = s_tot * tabs(1.0 - 2.0 * texp(tmat(tshape(1.0/x[:,5],(Nvox,1)).type_as(x),-TI.type_as(x))) + texp(tmat(tshape(1.0/x[:,5],(Nvox,1)).type_as(x),-TR.type_as(x))))
            #s_tot = s_tot * tabs(1.0 - 2.0 * texp(tmat(tshape(1.0/x[:,6],(Nvox,1)).type_as(x),-TI.type_as(x))) + texp(tmat(tshape(1.0/x[:,6],(Nvox,1)).type_as(x),-TR.type_as(x)))).type_as(x)
            #s_tot = s_tot * texp(tmat(tshape(x[:,0]*x[:,1],(Nvox,1)).type_as(x),-bval.type_as(x)))
            s_tot = s_tot * texp(tmat(tshape(1.0/x[:,4],(Nvox,1)).type_as(x),-TD.type_as(x)))
            #s_tot = s_tot * texp(tmat(tshape(1.0/x[:,5],(Nvox,1)).type_as(x),-TD.type_as(x)))
            #s_tot = s_tot * tmat(tshape(x[:,7], (Nvox,1)), Tensor(np.ones((1,Nmeas))))
            #s_tot = s_tot * tmat(tshape(x[:,7], (Nvox,1)).type_as(x), ones_ten)
            # at the moment, we do not consider the correction factor
            """s_tot = s_tot * terf(tsqrt(tmat(tshape(x[:,2] - x[:,2]*x[:,3],(Nvox,1)).type_as(x),bval.type_as(x))))
            s_tot = s_tot / tsqrt(tmat(tshape(x[:,2] - x[:,2]*x[:,3],(Nvox,1)).type_as(x),bval.type_as(x)))"""
            
            #x = torch.abs(1.0*s_tot)
            x = 1.0*s_tot
            return x
    
    def forward(self, x: torch.Tensor):#, ind_MUDI: torch.Tensor) -> torch.Tensor:
        """Uses the trained decoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the decoder input.

        Returns:
            torch.Tensor: decoder output of size `output_size`.
        """
        encoded2 = self.getparams(x)
        decoded = self.getsignals(encoded2)
        return encoded2, decoded

class ConcreteAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ind_path: Path,
        #input_output_size: int = 1344,
        latent_size: int = 500,
        latent_size2: int = 7,
        encoder2_hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        #max_temp: float = 10.0,
        #min_temp: float = 0.1,
        #reg_lambda: float = 0.0,
        #reg_threshold: float = 1.0,
        profiler=None,
    ) -> None:
        """Trains a concrete autoencoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346).

        Args:
            input_output_size (int): size of the input and output layer. latent_size (int): size of the latent layer.
            decoder_hidden_layers (int, optional): number of hidden layers for the decoder. Defaults to 2.
            learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-3.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
            reg_lambda(float, optional): how much weight to apply to the regularization term. If the value is 0.0 then
            no regularization will be applied. Defaults to 0.0.
            reg_threshold (float, optional): regularization threshold. The encoder will be penalized when the sum of
            probabilities for a selection neuron exceed this threshold. Defaults to 1.0.
        """
        super(ConcreteAutoencoder, self).__init__()
        self.save_hyperparameters()

        self.qmrizebra = qmrizebra(
            #input_size1 = input_output_size,
            input_size2 = latent_size,
            output_size2 = latent_size2,
            n_hidden_layers2 = encoder2_hidden_layers,
        )
        """self.encoder = Encoder(
            input_output_size,
            latent_size,
            max_temp,
            min_temp,
            profiler=profiler,
            reg_threshold=reg_threshold,
        )
        
        self.decoder = Decoder(latent_size, input_output_size, encoder2_hidden_layers)"""

        self.learning_rate = learning_rate
        #self.ind_mudi = np.loadtxt('/cubric/data/sapap9/runs_indices_results/synt_wm_gm/seed42_subj15_val_maarten_noreg/K=500_epoch=2000_test_15_dec.txt')
        self.ind_mudi = np.loadtxt(ind_path)
        #self.reg_lambda = reg_lambda
        #self.flag_stage2 = False
        """self.pred_sig_save = torch.zeros(108378,1344,dtype=float,requires_grad=False)
        self.pred_sig_save[self.pred_sig_save == 0] = float('nan')
        self.param_save = torch.zeros(108378,8,dtype=float,requires_grad=False)
        self.param_save[self.param_save == 0] = float('nan')"""

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add model specific arguments to argparse.

        Args:
            parent_parser (ArgumentParser): parent argparse to add the new arguments to.

        Returns:
            ArgumentParser: parent argparse.
        """
        parser = parent_parser.add_argument_group("autoencoder.ConcreteAutoencoder")
        parser.add_argument(
            "--checkpoint",
            default=None,
            type=file_path,
            metavar="PATH",
            help="Checkpoint file path to restore from.",
        )
        parser.add_argument(
            "--hparams",
            default=None,
            type=file_path,
            metavar="PATH",
            help="hyper parameter file path to restore from.",
        )
        parser.add_argument(
            "--input_output_size",
            "-s",
            default=1344,
            type=int,
            metavar="N",
            help="size of the input and output layer",
        )
        parser.add_argument(
            "--latent_size",
            "-l",
            default=500,
            type=int,
            metavar="N",
            help="size of latent layer",
        )
        parser.add_argument(
            "--latent_size2",
            "-l2",
            default=7,
            type=int,
            metavar="N",
            help="size of latent layer 2",
        )
        parser.add_argument(
            "--encoder2_hidden_layers",
            default=2,
            type=int,
            metavar="N",
            help="number of hidden layers for the second encoder (default: 2)",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            metavar="N",
            help="learning rate for the optimizer (default: 1e-2)",
        )

        parser.add_argument(
            "--reg_lambda",
            default=0.0,
            type=float,
            metavar="N",
            help="how much weight to apply to the regularization term. If `0` then no regularization will be applied. (default: 0.0)",
        )

        parser.add_argument(
            "--reg_threshold",
            default=1.0,
            type=float,
            metavar="N",
            help="how many duplicates in the latent space are allowed before applying the penalty. (default: None)",
        )
        parser.add_argument(
            "--ind_path",
            default=None,
            type=file_path,
            metavar="PATH",
            help="File with the selected measurements",
        )

        return parent_parser

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uses the trained autoencoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as encoder input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (encoder output, decoder output)
        """
        #encoded = self.encoder(x)
        #decoded = self.decoder(encoded)
        #ind_mudi = self.encoder.get_indexes()
        #x = x[:,ind_mudi]
        decoded, decoded2 = self.qmrizebra(x[:,self.ind_mudi]) # Actually, this decoded should be "encoded", but it has the name "decoded" to be returned together with the predicted signal and respect the original notation
        #decoded, decoded2 = self.qmrizebra(x)
        #encoded2, decoded2 = self.qmrizebra(x)
        return decoded, decoded2

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval(batch, batch_idx, "train")

        """if self.reg_lambda > 0:
            reg_term = self.encoder.regularization()
            loss = loss + (self.reg_lambda * reg_term)

            self.log("regularization_term", reg_term, on_step=False)
            self.log("regularized_train_loss", loss, on_step=False)"""

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    """def on_train_epoch_start(self) -> None:
        temp = self.encoder.update_temp(self.current_epoch, self.trainer.max_epochs)
        self.log("temp", temp, on_step=False, prog_bar=True)

    def on_epoch_end(self) -> None:
        mean_max = self.encoder.calc_mean_max()
        self.log("mean_max", mean_max, on_step=False, prog_bar=True)
        if mean_max > 0.998:
            np.savetxt('/cubric/data/sapap9/runs_indices_results/pred_sig15_zebra_seed_xx.txt', self.pred_sig_save.cpu().detach().numpy())
            np.savetxt('/cubric/data/sapap9/runs_indices_results/param_sig15_zebra_seed_xx.txt', self.param_save.cpu().detach().numpy())"""
        #if self.flag_stage2 == False and mean_max > 0.9:
        #    self.flag_stage2 = True

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
            prefix (str): prefix for logging.

        Returns:
            torch.Tensor: calculated loss.
        """
        decoded, decoded2 = self.forward(batch)
        loss = F.mse_loss(decoded2, batch)
        # here we evaluate the error against the maps instead of against the original signal
        #loss = F.mse_loss(decoded2, decoded)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    

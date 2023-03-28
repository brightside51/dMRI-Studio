import os
from argparse import ArgumentParser, Namespace

import torch

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything

import sys
sys.path.insert(0,'/cubric/data/sapap9/v2b_runs_indices_maarten/src/autoencoder2')

from concrete_autoencoder_zebraorig_v3b2 import ConcreteAutoencoder
from dataset2 import MRIDataModule
from logger import logger, set_log_level

import numpy as np


def trainer(args: Namespace) -> None:
    """Take command line arguments to train a model.

    Args:
        args (Namespace): arguments from argparse
    """
    experiment_name = "concrete_autoencoder"

    set_log_level(args.verbose)
    is_verbose = args.verbose < 30

    logger.info("Start training with params: %s", str(args.__dict__))

    model = ConcreteAutoencoder(
        args.ind_path,
        args.latent_size,
        args.latent_size2,
        args.encoder2_hidden_layers,
        learning_rate=args.learning_rate,
    )

    if args.checkpoint is not None:
        logger.info("Loading from checkpoint")
        model = model.load_from_checkpoint(
            str(args.checkpoint), hparams_file=str(args.hparams)
        )

    dm = MRIDataModule(
        data_file=args.data_file,
        header_file=args.header_file,
        batch_size=args.batch_size,
        val_subj=args.val_subj,
        #subject_train = args.subject_train,
        #subject_val = args.subject_val,
        #seed_number = args.seed_number,
        in_memory=args.in_memory,
        #path_save=args.path_save,
        #path_save_param=args.path_save_param,
    )

    plugins = []
    if args.accelerator == "ddp":
        plugins = [
            DDPPlugin(find_unused_parameters=False, gradient_as_bucket_view=True)
        ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=float("inf"),
                stopping_threshold=0.00001,
                verbose=is_verbose,
            ),
            ModelCheckpoint(
                monitor="mean_max",
                mode="max",
                save_top_k=1,
                verbose=is_verbose,
            ),
        ],
        checkpoint_callback=True,
        logger=TensorBoardLogger("logs", name=experiment_name),
        plugins=plugins,
    )

    if "MLFLOW_ENDPOINT_URL" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT_URL"])

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    mlflow.log_params(vars(args))
    
    #seed_everything(42)
    seed_everything(args.seed_number)
    
    dm.setup("test")
    prediction = trainer.predict(model, dm.test_dataloader())
    
    if args.val_subj == 11:
        param_save = torch.zeros(108300,7,dtype=float,requires_grad=False) # subject 11
        pred_sig_save = torch.zeros(108300,1344,dtype=float,requires_grad=False) #subj11
    elif args.val_subj == 12:
        param_save = torch.zeros(114233,7,dtype=float,requires_grad=False) # subject 12
        pred_sig_save = torch.zeros(114233,1344,dtype=float,requires_grad=False) #subj12
    elif args.val_subj == 13:
        param_save = torch.zeros(109236,7,dtype=float,requires_grad=False) # subject 13
        pred_sig_save = torch.zeros(109236,1344,dtype=float,requires_grad=False) #subj13
    elif args.val_subj == 14:
        param_save = torch.zeros(136799,7,dtype=float,requires_grad=False) # subject 14
        pred_sig_save = torch.zeros(136799,1344,dtype=float,requires_grad=False) #subj14
    elif args.val_subj == 15:
        param_save = torch.zeros(108378,7,dtype=float,requires_grad=False) # subject 15
        pred_sig_save = torch.zeros(108378,1344,dtype=float,requires_grad=False) #subj15
    
    param_save[param_save == 0] = float('nan')
    # wm-gm
    #param_save = torch.zeros(69445,7,dtype=float,requires_grad=False) #subj15
    #param_save = torch.zeros(93423,7,dtype=float,requires_grad=False) #subj14
    #param_save = torch.zeros(74322,7,dtype=float,requires_grad=False) #subj13
    #param_save = torch.zeros(75779,7,dtype=float,requires_grad=False) #subj12
    #param_save = torch.zeros(72105,7,dtype=float,requires_grad=False) #subj11
    
    # wm-gm
    #pred_sig_save = torch.zeros(69445,1344,dtype=float,requires_grad=False) #subj15
    #pred_sig_save = torch.zeros(93423,1344,dtype=float,requires_grad=False) #subj14
    #pred_sig_save = torch.zeros(74322,1344,dtype=float,requires_grad=False) #subj13
    #pred_sig_save = torch.zeros(75779,1344,dtype=float,requires_grad=False) #subj12
    #pred_sig_save = torch.zeros(72105,1344,dtype=float,requires_grad=False) #subj11
    
    pred_sig_save[pred_sig_save == 0] = float('nan')
    batch_idx = 0
    #print(prediction)
    #textfile = open("/cubric/data/sapap9/runs_indices_results/pred_sig14_zebra_seed_142.txt", "w")
    for element in prediction:
        #textfile.write(signal.detach().cpu().numpy() + "\n")
        param_save[batch_idx*256:(batch_idx+1)*256,:] = element[0]
        pred_sig_save[batch_idx*256:(batch_idx+1)*256,:] = element[1]
        batch_idx = batch_idx+1
    np.savetxt(args.path_save_param, param_save.cpu().detach().numpy())
    np.savetxt(args.path_save, pred_sig_save.cpu().detach().numpy())


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Concrete Autoencoder trainer", usage="%(prog)s [options]"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 10, 20, 30, 40, 50],
        default=20,
        metavar="XX",
        help="verbosity level (default: 10)",
    )
    """parser.add_argument(
        "--val_subj",
        default=15,
        type=int,
        metavar="N",
        help="subject employed for validation (default: 15)",
    )"""
    parser.add_argument(
        "--seed_number",
        default=42,
        type=int,
        metavar="N",
       help="seed employed to initialise the job (default: 42)",
    )
    parser.add_argument(
        "--path_save",
        type=str,
        required=False,
        metavar="PATH",
        help="file name of the path to save the predicted signal",
    )
    parser.add_argument(
        "--path_save_param",
        type=str,
        required=False,
        metavar="PATH",
        help="file name of the path to save the predicted maps of parameters",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ConcreteAutoencoder.add_model_specific_args(parent_parser=parser)
    parser = MRIDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    trainer(args)

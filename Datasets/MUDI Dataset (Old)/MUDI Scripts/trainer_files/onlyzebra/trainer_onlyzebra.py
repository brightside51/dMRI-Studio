import os
from argparse import ArgumentParser, Namespace

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
import torch

def trainer(args: Namespace) -> None:
    """Take command line arguments to train a model.

    Args:
        args (Namespace): arguments from argparse
    """
    experiment_name = "concrete_autoencoder"

    set_log_level(args.verbose)
    is_verbose = args.verbose < 30

    logger.info("Start training with params: %s", str(args.__dict__))
    
    #pred_sig = torch.zeros(108378,1344,dtype=float,requires_grad=False)
    #pred_sig[pred_sig == 0] = float('nan')

    model = ConcreteAutoencoder(
        #args.input_output_size,
        args.ind_path,
        args.latent_size,
        args.latent_size2,
        args.encoder2_hidden_layers,
        learning_rate=args.learning_rate,
        #reg_lambda=args.reg_lambda,
        #reg_threshold=args.reg_threshold,
        #pred_sig_save=pred_sig,
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
                monitor="val_loss",
                mode="min",
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

    trainer.fit(model, dm)
    
    #print(np.sort(model.get_indices()))
    #np.savetxt('/home/sapap9/PythonCode/MUDI/runs/textfiles/K=500_epoch=2000_test_' + '15' + '_dec.txt', np.array(model.get_indices(), dtype=int), fmt='%d')
    
    #torch.save(model.get_params(), '/cubric/data/sapap9/runs_indices_results/seed142_subj15_zebra_params.pt')


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

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ConcreteAutoencoder.add_model_specific_args(parent_parser=parser)
    parser = MRIDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    trainer(args)

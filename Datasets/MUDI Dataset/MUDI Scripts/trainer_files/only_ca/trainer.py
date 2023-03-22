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

from concrete_autoencoder_orig2 import ConcreteAutoencoder
from dataset2 import MRIDataModule
from logger import logger, set_log_level

import numpy as np


#parser2 = ArgumentParser(
#        description="Some parameters to save files", usage="%(prog)s [options]"
#    )


#args2 = parser2.parse_args()

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
        args.input_output_size,
        args.latent_size,
        args.decoder_hidden_layers,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        reg_threshold=args.reg_threshold,
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
                monitor="mean_max",
                mode="max",
                patience=float("inf"),
                stopping_threshold=0.998,
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

    trainer.fit(model, dm)
    
    #print(np.sort(model.get_indices()))
    #np.savetxt('/home/sapap9/PythonCode/MUDI/runs/textfiles/K=500_epoch=2000_test_' + '15' + '_dec.txt', np.array(model.get_indices(), dtype=int), fmt='%d')
    np.savetxt('/home/sapap9/PythonCode/MUDI/runs/textfiles/K=500_epoch=2000_test_' + str(args.val_subj) + '_dec.txt', np.array(model.get_indices(), dtype=int), fmt='%d')

    """dm.setup("test")
    prediction = trainer.predict(model, dm.test_dataloader())
    pred_sig_save = torch.zeros(136799,1344,dtype=float,requires_grad=False)
    pred_sig_save[pred_sig_save == 0] = float('nan')
    batch_idx = 0
    #textfile = open("/cubric/data/sapap9/runs_indices_results/pred_sig14_zebra_seed_142.txt", "w")
    for element in prediction:
        #textfile.write(signal.detach().cpu().numpy() + "\n")
        pred_sig_save[batch_idx*256:(batch_idx+1)*256,:] = element[1]
        batch_idx = batch_idx+1
    #textfile.close()
    np.savetxt('/cubric/data/sapap9/runs_indices_results/pred_sig14_zebra_seed_xx.txt', pred_sig_save.cpu().detach().numpy())"""


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

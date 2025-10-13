import argparse
import sys
import os
import time
import pandas as pd
import wandb
import gc
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from datamodule import *
from model import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="../data/SERSIC/dataset_multires_30.npy", help='Images path')
    parser.add_argument('--metadata_path', type=str, default="../data/SERSIC/df_coords_fix.csv", help='Metadata path')
    parser.add_argument('--autolabeling_dataset_path',  type=str, default=None)
    parser.add_argument('--model_name', type=str, default="delight", help='delight or resnet')

    parser.add_argument('--lr', type=float, default=0.0014, help='Learning Rate Train')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay Train')

    parser.add_argument('--batch_size', type=int, default=40, help='Training Batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate grad_batches')
    parser.add_argument('--devices', type=int, default=1, help='N° GPU devices')

    parser.add_argument('--num_workers', type=int, default=4, help='Num of Workers of dataloaders')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument('--run_name', type=str, default="prueba", help='Name of the w&b run')

    parser.add_argument('--seed', type=int, default=0, help='Seed of the experiment')
 
    args = parser.parse_args()


    #-----------REPRODUCIBILIDAD-----------#
    L.seed_everything(args.seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    #-----------REPRODUCIBILIDAD-----------#

    #-----W&B-----#
    wandb.login()
    #-----W&B-----#

    #-----------CARPETA CONTENEDORA-----------#
    save_files = f"../resultados/{args.run_name}"
    os.makedirs(save_files, exist_ok=True)
    #-----------CARPETA CONTENEDORA-----------#

    #-----------CARGA DE DATOS-----------#
    print("Cargando Datos\n")

    inicio = time.time()

    images = np.load(args.images_path)                               
    df = pd.read_csv(args.metadata_path, dtype={'objID': 'Int64'})    

    sn_pos = df[["dx","dy"]].values.astype(np.float32)

    base_sersic_path = Path("..") / "data" / "SERSIC"
    oid_train = np.load(base_sersic_path / "id_train.npy", allow_pickle=True)
    oid_val = np.load(base_sersic_path / "id_validation.npy", allow_pickle=True)
    oid_test = np.load(base_sersic_path / "id_test.npy", allow_pickle=True)

    idx_train = df[df['oid'].isin(oid_train)].index.to_numpy()
    idx_val = df[df['oid'].isin(oid_val)].index.to_numpy()
    idx_test = df[df['oid'].isin(oid_test)].index.to_numpy()

    df_train = df[df['oid'].isin(oid_train)]

    X_train = images[idx_train]
    X_val = images[idx_val]
    X_test = images[idx_test]

    del images, df

    y_train = sn_pos[idx_train]
    y_val = sn_pos[idx_val]
    y_test = sn_pos[idx_test]

    if args.autolabeling_dataset_path:
        
        print("Using autolabeling dataset")
        data = np.load(base_sersic_path / args.autolabeling_dataset_path)
        X_train = data["imgs"]
        y_train = data["pos"]

        # mask_x = (np.abs(y_train[:,0]) < 255)
        # mask_y = (np.abs(y_train[:,1]) < 255)
        # mask = mask_x & mask_y
        
        # X_train = X_train[mask]
        # y_train = y_train[mask]

        del data

    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)

    print(f"Carga de datos finalizada en {time.time()-inicio} [s]\n")
    #-----------CARGA DE DATOS-----------#

    #-----------ENTRENAMIENTO-----------#
    dm = DelightDataModule(X_train=X_train, 
                            X_val=X_val, 
                            X_test=X_test, 
                            y_train=y_train, 
                            y_val=y_val, 
                            y_test=y_test,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            seed=args.seed)


    if args.model_name == "delight":
        config = {
            "nconv1": 52,
            "nconv2": 57,
            "nconv3": 41,
            "ndense": 685,
            "dropout": 0.06,
            "channels": 1,
            "levels": 5,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }

        model = Delight(config)
    
    elif args.model_name == 'resnet18':

        config = {
            "ndense": 685,
            "dropout": 0.06,
            "channels": 1,
            "levels": 5,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }

        model = Resnet18(config)

    wandb_logger = WandbLogger(project="PRISM", name =args.run_name, save_dir = save_files, entity="fforster-uchile")

    wandb_logger.log_hyperparams({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed
    })

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", 
        dirpath=save_files, 
        filename='delight_best_{epoch}', 
        save_top_k=1,
        save_last = False,  
        mode="min")

    lr_callback = LearningRateMonitor(logging_interval="epoch",
                                      log_momentum=True,
                                      log_weight_decay=True)
    
    progress_bar_callback = RichProgressBar()

    trainer = L.Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        deterministic=True,
        max_epochs=args.epoch,
        accelerator ="gpu",
        devices = args.devices,
        callbacks=[checkpoint_callback, lr_callback, progress_bar_callback])

    inicio = time.time()
    trainer.fit(model, dm)
    print(f"Entrenamiento finalizado en {time.time()-inicio} [s]\n")
    #-----------ENTRENAMIENTO-----------#

    #-----------PREDICCIONES-----------#
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    best_ckpt_path = checkpoint_callback.best_model_path

    trainer_test = L.Trainer(
        accelerator="gpu",   
        devices=1,
        logger=wandb_logger,
        deterministic=True,
        num_sanity_val_steps=0

    )

    trainer_test.test(
        model=model,         
        ckpt_path=best_ckpt_path,
        datamodule=dm
    )
    #trainer.test(ckpt_path="best", datamodule=dm)

    test_preds = model.test_predictions
    test_targets = model.test_targets 

    test_mean_preds = model.test_mean_preds
    test_original_targets = model.test_original_targets

    wandb.finish()

    np.savez(os.path.join(save_files, "test_results.npz"), 
             preds=test_preds.numpy(), 
             targets=test_targets.numpy(),
             mean_preds = test_mean_preds.numpy(),
             original_target = test_original_targets.numpy())
    #-----------PREDICCIONES-----------#



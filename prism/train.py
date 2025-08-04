import argparse
import sys
import os
import time
import pandas as pd
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from prism.datamodule import *
from model import *


def get_balance_mask(df, seed):

    n=12
    bins_arcsec = np.linspace(0,300*0.25,n)

    df['bin'] = pd.cut(df['rSerRadius'] * 3, bins=bins_arcsec, right=False)
    df['bin'] = df['bin'].astype(object)

    # Extraer límite izquierdo de cada bin
    df['bin_left'] = df['bin'].map(lambda x: x.left if pd.notnull(x) else np.nan)

    # Crear máscara inicial
    mask = pd.Series(False, index=df.index)

    # Bins < 40 → ordenar e interpolar de 10% a 50%
    bins_lt_40 = df[df['bin_left'] < 40]['bin'].dropna().unique()
    bins_lt_40 = sorted(bins_lt_40, key=lambda x: x.left)  # ordenarlos por el límite izquierdo

    n_bins = len(bins_lt_40)
    #fracs = np.linspace(0.05, 0.3, n_bins)  
    fracs = np.logspace(np.log10(0.01), np.log10(0.5), n_bins)

    for bin_i, frac in zip(bins_lt_40, fracs):
        df_bin = df[df['bin'] == bin_i]
        n_samples = int(len(df_bin) * frac)
        sampled_idx = df_bin.sample(n=n_samples, replace=False, random_state=seed).index
        mask.loc[sampled_idx] = True

    # Bins >= 40 → conservar todos
    mask.loc[df[df['bin_left'] >= 40].index] = True

    return mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="..\data\SERSIC\dataset_multires_30.npy", help='Images path')
    parser.add_argument('--metadata_path', type=str, default="..\data\SERSIC\df_coords_fix.csv", help='Metadata path')
    parser.add_argument('--augmented_dataset', action='store_true', help='Usa el dataset aumentado')
    parser.add_argument('--model_name', type=str, default="delight", help='delight or resnet')

    parser.add_argument('--lr', type=float, default=0.0014, help='Learning Rate Train')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay Train')

    parser.add_argument('--batch_size', type=int, default=40, help='Training Batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate grad_batches')

    parser.add_argument('--num_workers', type=int, default=4, help='Num of Workers of dataloaders')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument('--save_files', type=str, default="../resultados/prueba", help='File name of the results')
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
    os.makedirs(args.save_files, exist_ok=True)
    #-----------CARPETA CONTENEDORA-----------#

    #-----------CARGA DE DATOS-----------#
    print("Cargando Datos\n")

    inicio = time.time()

    images = np.load(args.images_path)                               
    df = pd.read_csv(args.metadata_path, dtype={'objID': 'Int64'})    

    sn_pos = df[["dx","dy"]].values.astype(np.float32)

    oid_train = np.load(f"..\data\SERSIC\id_train.npy",allow_pickle=True) 
    oid_val = np.load(f"..\data\SERSIC\id_validation.npy",allow_pickle=True) 
    oid_test = np.load(f"..\data\SERSIC\id_test.npy",allow_pickle=True) 

    idx_train = df[df['oid'].isin(oid_train)].index.to_numpy()
    idx_val = df[df['oid'].isin(oid_val)].index.to_numpy()
    idx_test = df[df['oid'].isin(oid_test)].index.to_numpy()

    df_train = df[df['oid'].isin(oid_train)]

    delta = 0.10  # 15%
    diferencia_relativa = np.abs(df_train["rSerRadius"] * 3 - df_train["hostsize"]) / df_train["hostsize"]
    mask_10 = diferencia_relativa <= delta
    idx = np.arange(len(df_train))[mask_10]

    X_train = images[idx_train][mask_10]
    X_val = images[idx_val]
    X_test = images[idx_test]

    print(X_train.shape)
    del images, df

    y_train = sn_pos[idx_train][mask_10]
    y_val = sn_pos[idx_val]
    y_test = sn_pos[idx_test]

    if args.augmented_dataset:
        
        idx_aug = np.concatenate([np.arange(n * 30, n * 30 + 30) for n in idx])

        print("Using augmented dataset")
        data = np.load("..\data\SERSIC\X_train_augmented_x30.npz")
        #data = np.load("..\data\SERSIC\X_train_pasquet_augmented_x10.npz")
        X_train = data["imgs"][idx_aug]
        y_train = data["pos"][idx_aug]

        mask_ceros = (X_train.sum((1,2))==0).any(1)
        print(f"Valores nulos: {mask_ceros.sum()}")

        X_train = X_train[~mask_ceros]
        y_train = y_train[~mask_ceros]


        # df_train = pd.read_csv("..\data\SERSIC\df_train.csv", dtype={'objID': 'Int64'})   
        # mask = get_balance_mask(df_train, args.seed)

        # idx_tiny = (df_train[mask]).index
        # idx_tiny = np.hstack([range(idx*30,idx*30+30) for idx in idx_tiny])

        # mask_balance = np.isin(np.arange(len(X_train)), idx_tiny)
        # print(f"Balance: {mask_balance.sum()/len(mask_balance)}")

        # X_train = X_train[mask_balance]
        # y_train = y_train[mask_balance]

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
    
    else:

        config = {
            "ndense": 685,
            "dropout": 0.06,
            "channels": 1,
            "levels": 5,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }

        model = Resnet18(config)

    wandb_logger = WandbLogger(project="PRISM", name =args.run_name, save_dir = args.save_files, entity="fforster-uchile")

    wandb_logger.experiment.config.update({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed
    })

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", 
        dirpath=args.save_files, 
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
        devices = "auto",
        callbacks=[checkpoint_callback, lr_callback, progress_bar_callback])

    inicio = time.time()
    trainer.fit(model, dm)
    print(f"Entrenamiento finalizado en {time.time()-inicio} [s]\n")
    #-----------ENTRENAMIENTO-----------#

    #-----------PREDICCIONES-----------#
    trainer.test(ckpt_path="best", datamodule=dm)

    test_preds = model.test_predictions
    test_targets = model.test_targets 

    test_mean_preds = model.test_mean_preds
    test_original_targets = model.test_original_targets

    wandb.finish()

    np.savez(os.path.join(args.save_files, "test_results.npz"), 
             preds=test_preds.numpy(), 
             targets=test_targets.numpy(),
             mean_preds = test_mean_preds.numpy(),
             original_target = test_original_targets.numpy())
    #-----------PREDICCIONES-----------#

    #-----------RESULTADOS-----------#
    np.save(os.path.join(args.save_files, "curvas.npy"), model.curves)
    #-----------RESULTADOS-----------#


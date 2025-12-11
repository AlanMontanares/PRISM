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
from lightning.pytorch.strategies import DDPStrategy
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from datamodule import *
from model import *
from utils.plot_functions import obtain_predicted_ra_dec


def obtain_inputs(dataset_path, task, channels):

    data = np.load(dataset_path)

    X = data["imgs"]

    all_channels = ["g", "r", "i", "z", "y"]
    selected_channels = list(channels)
    idx_channels = [all_channels.index(c) for c in selected_channels]

    if X.shape[2] > 1: # multi-channel
        X = torch.from_numpy(X[:, :, idx_channels, :, :])
    else:
        X = torch.from_numpy(X)

    if "g" in selected_channels:
        print("usando arcoseno hiperbolico")
        g_idx = selected_channels.index("g")
        X[:, :, g_idx, :, :] = torch.asinh(X[:, :, g_idx, :, :])

    mask = (X.sum((3,4))==0).any((1,2)) # Casos con bandas == 0

    if task == "galaxy_hunter":
        pos = torch.from_numpy(data["pos"])
        return X[~mask], pos[~mask]

    elif task == "redshift_prediction":
        z = torch.from_numpy(data["z"])
        range_z = np.linspace(0, 0.4, 181)[:-1]
        z_train_class = torch.tensor(np.digitize(data["z"],range_z)-1)

        return X[~mask], z_train_class[~mask], z[~mask]

    elif task == "multitask":
        pos = torch.from_numpy(data["pos"])

        range_z = np.linspace(0, 0.4, 181)[:-1]
        z_train_class = torch.tensor(np.digitize(data["z"],range_z)-1)

        return X[~mask], pos[~mask], z_train_class[~mask]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str, default="train_delight.npz")
    parser.add_argument('--backbone', type=str, default="delight", help='delight, resnet18, resnet32, resnet50')
    parser.add_argument('--task', type=str, default='galaxy_hunter', help='galaxy_hunter, redshift_prediction or multitask')
    parser.add_argument('--center_on_galaxy', type=str, default=None, help="center on galaxy or sn")
    parser.add_argument('--channels', type=str, default='r', help='Image filters to use')

    parser.add_argument('--use_gradnorm', type=str, default=None, help='Use Gradnorm or not')


    parser.add_argument('--lr', type=float, default=0.0014, help='Learning Rate Train')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay Train')
    parser.add_argument('--alpha', type=float, default=1.0, help='GradNorm alpha')

    parser.add_argument('--use_sampler', type=str, default=None)
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

    #-----------CONFIGURACION DE MODELO-----------#
    config = {
        "backbone": args.backbone,
        "nconv1": 52,
        "nconv2": 57,
        "nconv3": 41,
        "ndense": 685,
        "ndense_cls": 1096,
        "out_cls": 180,
        "dropout": 0.06,
        "channels": len(args.channels),
        "levels": 5,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha
    }

    #-----------CONFIGURACION DE MODELO-----------#

    #-----------CARGA DE DATOS-----------#
    print("Cargando Datos\n")

    inicio = time.time()

    base_sersic_path = Path("..") / "data" / "SERSIC"

    pos_train, pos_val, pos_test = None, None, None
    z_train_class, z_val_class, z_test_class = None, None, None
    X_val, X_test, X_val_delight, X_test_delight, X_val_z, X_test_z = None, None, None, None, None, None

    if args.task == "galaxy_hunter":

        X_train, pos_train = obtain_inputs(dataset_path= base_sersic_path / args.train_dataset_name, task=args.task, channels=args.channels)
        X_val, pos_val = obtain_inputs(dataset_path= base_sersic_path / "val_delight.npz", task=args.task, channels=args.channels)
        X_test, pos_test = obtain_inputs(dataset_path= base_sersic_path / "test_delight_fixed.npz", task=args.task, channels=args.channels)

        model = Delight(config)


    elif args.task == "redshift_prediction":
        
        if args.center_on_galaxy:
            print("Center on Galaxy")
            X_train, z_train_class, _ = obtain_inputs(dataset_path= base_sersic_path / args.train_dataset_name, task=args.task, channels=args.channels)
            X_val, z_val_class, _ = obtain_inputs(dataset_path= base_sersic_path / "X_val_pasquet.npz", task=args.task, channels=args.channels)
            X_test, z_test_class, z_test = obtain_inputs(dataset_path= base_sersic_path / "X_test_pasquet.npz", task=args.task, channels=args.channels)

        else:
            X_train, z_train_class, _ = obtain_inputs(dataset_path= base_sersic_path / args.train_dataset_name, task=args.task, channels=args.channels)
            X_val, z_val_class, _ = obtain_inputs(dataset_path= base_sersic_path / "X_val_autolabeling_pasquet.npz", task=args.task, channels=args.channels)
            X_test, z_test_class, z_test = obtain_inputs(dataset_path= base_sersic_path / "X_test_autolabeling_pasquet.npz", task=args.task, channels=args.channels)

        model = Delight_z(config)

    elif args.task == "multitask":

        X_train, pos_train, z_train_class = obtain_inputs(dataset_path= base_sersic_path / args.train_dataset_name, task=args.task, channels=args.channels)

        X_val_delight, pos_val = obtain_inputs(dataset_path= base_sersic_path / "val_delight.npz", task="galaxy_hunter", channels=args.channels)
        X_test_delight, pos_test = obtain_inputs(dataset_path= base_sersic_path / "test_delight_fixed.npz", task="galaxy_hunter", channels=args.channels)

        X_val_z, z_val_class, _ = obtain_inputs(dataset_path= base_sersic_path / "X_val_autolabeling_pasquet.npz", task="redshift_prediction", channels=args.channels)
        X_test_z, z_test_class, z_test = obtain_inputs(dataset_path= base_sersic_path / "X_test_autolabeling_pasquet.npz", task="redshift_prediction", channels=args.channels)

        if args.use_gradnorm:
            model = Delight_multitask(config)
        else:
            print("sin usar gradnorm")
            model = Delight_multitask_no_GN(config)

    print(f"Carga de datos finalizada en {time.time()-inicio} [s]\n")
    #-----------CARGA DE DATOS-----------#

    #-----------ENTRENAMIENTO-----------#
    dm = PRISMDataModule(X_train=X_train, 
                        X_val=X_val, 
                        X_test=X_test,
                        X_val_pos = X_val_delight,
                        X_val_z = X_val_z,
                        X_test_pos = X_test_delight,
                        X_test_z = X_test_z,
                        pos_train=pos_train, 
                        pos_val=pos_val, 
                        pos_test=pos_test,
                        z_train = z_train_class,
                        z_val=z_val_class,
                        z_test=z_test_class,
                        task=args.task,
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers, 
                        seed=args.seed,
                        use_sampler=args.use_sampler)



    wandb_logger = WandbLogger(project="PRISM", name =args.run_name, save_dir = save_files, entity="fforster-uchile")

    wandb_logger.log_hyperparams({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed
    })

 
    lr_callback = LearningRateMonitor(logging_interval="epoch",
                                      log_momentum=True,
                                      log_weight_decay=True)
    
    progress_bar_callback = RichProgressBar()


    if args.task == "multitask":
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss" if args.task != "redshift_prediction" else "val/z_loss", 
            dirpath=save_files, 
            filename='best_loss_{epoch}', 
            save_top_k=1,
            save_last = False,  
            mode="min")

        checkpoint_callback_z = ModelCheckpoint(
            monitor="val/z_loss", 
            dirpath=save_files, 
            filename='best_zloss_{epoch}', 
            save_top_k=1,
            save_last = False,  
            mode="min")
        
        callbacks = [checkpoint_callback, checkpoint_callback_z ,lr_callback, progress_bar_callback]
    else:

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss" if args.task != "redshift_prediction" else "val/z_loss", 
            dirpath=save_files, 
            filename='delight_best_{epoch}', 
            save_top_k=1,
            save_last = False,  
            mode="min")

        callbacks = [checkpoint_callback, lr_callback, progress_bar_callback]
    

    trainer = L.Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        deterministic=True,
        max_epochs=args.epoch,
        accelerator ="gpu",
        strategy=DDPStrategy(find_unused_parameters=False) if args.task == "multitask" else "auto",
        devices = args.devices,
        callbacks=callbacks)

    inicio = time.time()
    trainer.fit(model, dm)
    print(f"Entrenamiento finalizado en {time.time()-inicio} [s]\n")
    #-----------ENTRENAMIENTO-----------#

    #-----------PREDICCIONES-----------#
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer_preds = L.Trainer(
        accelerator="gpu",   
        devices=1,
        logger=wandb_logger,
        deterministic=True,
        num_sanity_val_steps=0

    )

    #-----------TEST DATASET-----------#

    if args.task == "galaxy_hunter":

        best_ckpt_path = checkpoint_callback.best_model_path
        trainer_preds.test(model=model, ckpt_path=best_ckpt_path, datamodule=dm)

        test_mean_preds = model.test_mean_preds
        test_original_targets = model.test_original_targets
    
        df_test = pd.read_csv("../data/SERSIC/df_test_delight_fixed.csv")
        coords_preds = obtain_predicted_ra_dec(df_test, test_mean_preds)

        df_preds = pd.DataFrame(coords_preds, columns=['ra_pred','dec_pred'])
        df_preds.to_csv(os.path.join(save_files, "test_predictions.csv"), index=False)

        np.savez(os.path.join(save_files, "test_results.npz"), 
                mean_preds = test_mean_preds.numpy(),
                original_target = test_original_targets.numpy())

    elif args.task == "redshift_prediction":

        best_ckpt_path = checkpoint_callback.best_model_path
        trainer_preds.test(model=model, ckpt_path=best_ckpt_path, datamodule=dm)

        test_zphot = model.test_zphot
        np.savez(os.path.join(save_files, "test_results_redshift.npz"), 
                zphot = test_zphot.numpy(),
                zspect = z_test.numpy())

    elif args.task == "multitask":

        #====== Galaxy pos results ======#

        best_ckpt_path = checkpoint_callback.best_model_path
        trainer_preds.test(model=model, ckpt_path=best_ckpt_path, datamodule=dm)

        print(f"PRIMER CKPT : {model.test_zphot[:5]}")
        test_mean_preds = model.test_mean_preds_pos 
        test_original_targets = model.test_original_targets_pos
    
        df_test = pd.read_csv("../data/SERSIC/df_test_delight_fixed.csv")

        coords_preds = obtain_predicted_ra_dec(df_test, test_mean_preds)

        df_preds = pd.DataFrame(coords_preds, columns=['ra_pred','dec_pred'])
        df_preds.to_csv(os.path.join(save_files, "test_predictions.csv"), index=False)

        np.savez(os.path.join(save_files, "test_results.npz"), 
                mean_preds = test_mean_preds.numpy(),
                original_target = test_original_targets.numpy())


        #====== Redshift results ======#

        best_ckpt_path = checkpoint_callback_z.best_model_path
        trainer_preds.test(model=model, ckpt_path=best_ckpt_path, datamodule=dm)
        print(f"SEGUNDO CKPT : {model.test_zphot[:5]}")
        test_zphot = model.test_zphot
        np.savez(os.path.join(save_files, "test_results_redshift.npz"), 
                zphot = test_zphot.numpy(),
                zspect = z_test.numpy())
    #-----------PREDICCIONES-----------#



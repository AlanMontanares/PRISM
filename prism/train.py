import argparse
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from prism.datamodule import *
from model import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('--images_path', type=str, default="..\data\SERSIC\h2f_ps1_multires_delight_delight_method.npy", help='Images path')
    parser.add_argument('--metadata_path', type=str, default="..\data\SERSIC\delight_sersic.csv", help='Metadata path')
    parser.add_argument('--oids_origin', type=str, default="DELIGHT", help='Origin of the oids (DELIGHT/SERSIC)')

    parser.add_argument('--train_dataset_type', type=str, default="delight_classic", help='Classic Augmentation, Delight Augmentation or Auto-Labeling')

    parser.add_argument('--lr', type=float, default=0.0014, help='Learning Rate Train')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay Train')

    parser.add_argument('--batch_size', type=int, default=40, help='Training Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Num of Workers of dataloaders')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument('--save_files', type=str, default="../resultados/prueba", help='File name of the results')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the experiment')
 
    args = parser.parse_args()


#-----------REPRODUCIBILIDAD-----------#
    L.seed_everything(args.seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
#-----------REPRODUCIBILIDAD-----------#

#-----------CARPETA CONTENEDORA-----------#
    os.makedirs(args.save_files, exist_ok=True)
#-----------CARPETA CONTENEDORA-----------#

#-----------CARGA DE DATOS-----------#
    print("Cargando Datos\n")

    inicio = time.time()

    images = np.load("..\data\SERSIC\h2f_ps1_multires_delight_simple_method.npy")
    df = pd.read_csv(args.metadata_path, dtype={'objID': 'Int64'})

    sn_pos = df[["dx","dy"]].values.astype(np.float32)
    sersic_radius = df["rSerRadius"].values.astype(np.float32)
    sersic_ab = df["rSerAb"].values.astype(np.float32)
    sersic_phi = df["rSerPhi"].values.astype(np.float32)

    oid_train = np.load(f"..\data\{args.oids_origin}\id_train.npy",allow_pickle=True) # Escogemos las oids para entrenar, ya sea del split de Delight 
    oid_val = np.load(f"..\data\{args.oids_origin}\id_validation.npy",allow_pickle=True) # como del usado para auto-labeling (sersic)

    idx_train = df[df['oid'].isin(oid_train)].index.to_numpy()
    idx_val = df[df['oid'].isin(oid_val)].index.to_numpy()
    idx_test = np.setdiff1d(df.index, np.union1d(idx_train, idx_val))

    X_train = images[idx_train]
    X_val = images[idx_val]
    X_test = images[idx_test]

    del images

    y_train = sn_pos[idx_train]
    y_val = sn_pos[idx_val]
    y_test = sn_pos[idx_test]

    if args.train_dataset_type == "delight_autolabeling":
    
        mask_radius_10 = (df["rSerRadius"].values[idx_train] < 9.8)

        X_train = np.load("..\data\SERSIC\X_train_autolabeling.npy")[mask_radius_10]

        y_train = None
        sersic_radius = df["rSerRadius"].values.astype(np.float32)
        sersic_ab = df["rSerAb"].values.astype(np.float32)
        sersic_phi = df["rSerPhi"].values.astype(np.float32)

        train_sersic_radius = sersic_radius[idx_train][mask_radius_10]
        train_sersic_ab = sersic_ab[idx_train][mask_radius_10]
        train_sersic_phi = sersic_phi[idx_train][mask_radius_10]

        del sersic_radius, sersic_ab, sersic_phi

    else:
        train_sersic_radius = None
        train_sersic_ab = None
        train_sersic_phi = None  

    del df, sn_pos

    print(f"Carga de datos finalizada en {time.time()-inicio} [s]\n")
#-----------CARGA DE DATOS-----------#

#-----------ENTRENAMIENTO-----------#
    dm = DelightDataModule(X_train=X_train, 
                            X_val=X_val, 
                            X_test=X_test, 
                            y_train=y_train, 
                            y_val=y_val, 
                            y_test=y_test,
                            radius_train = train_sersic_radius,
                            ab_train = train_sersic_ab,
                            phi_train = train_sersic_phi, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            seed=args.seed, 
                            train_dataset_type=args.train_dataset_type)

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

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath=args.save_files, 
        filename='delight_best_{epoch}', 
        save_top_k=1,
        save_last = False,  
        mode="min")

    trainer = L.Trainer(
        num_sanity_val_steps=0,
        logger=False,
        deterministic=True,
        max_epochs=args.epoch,
        accelerator ="gpu",
        devices = "auto",
        callbacks=[checkpoint_callback])

    inicio = time.time()
    trainer.fit(model, dm)
    print(f"Entrenamiento finalizado en {time.time()-inicio} [s]\n")
#-----------ENTRENAMIENTO-----------#

#-----------PREDICCIONES-----------#
    test_preds = torch.cat(trainer.predict(model=model, datamodule =dm, ckpt_path="best"), dim=0)

    np.save(os.path.join(args.save_files, "test_predictions.npy"), test_preds.numpy())
#-----------PREDICCIONES-----------#

#-----------RESULTADOS-----------#
    np.save(os.path.join(args.save_files, "curvas.npy"), model.curves)
#-----------RESULTADOS-----------#
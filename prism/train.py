import argparse
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from prism.datamodule import *
from model import *



def obtain_clean_mask(df):

    Re_pix = df["rSerRadius"] / 0.25  # pasar a pixeles
    q = df["rSerAb"]                  # b/a
    theta_rad = np.deg2rad(df["rSerPhi"])  # en radianes
    r_max = 3  # radio elíptico que define la elipse (equivalente a r^2 = 9)

    # Coordenadas
    x = df["dx"]
    y = df["dy"]

    # Aplicar la misma rotación que en sersic_profile
    x_rot = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_rot = -x * np.sin(theta_rad) + y * np.cos(theta_rad)

    # Cálculo del radio elíptico al cuadrado
    ellipse_r = (x_rot / Re_pix)**2 + (y_rot / (Re_pix * q))**2

    # Condición: fuera de la elipse si radio elíptico > r^2
    mask_fuera_elipse = (ellipse_r > r_max**2)

    # Condicion: Chi-square > 50
    mask_chisq = (df["rSerChisq"] > 50)

    mask_final = (mask_fuera_elipse & mask_chisq)

    return ~mask_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="..\data\SERSIC\dataset_multires_30_simple_method.npy", help='Images path')
    parser.add_argument('--metadata_path', type=str, default="..\data\SERSIC\delight_sersic.csv", help='Metadata path')
    parser.add_argument('--augmented_dataset', action='store_true', help='Usa el dataset aumentado')
    parser.add_argument('--recenter', action='store_true', help='Centra las imagenes de autolabeling en las SN originales')
    parser.add_argument('--train_dataset_type', type=str, default="delight_classic", help='delight_classic or delight_autolabeling')

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

    images = np.load(args.images_path)                                # Imagenes multi-resolucion original
    df = pd.read_csv(args.metadata_path, dtype={'objID': 'Int64'})    # Dataframe con metadata de Sersic 

    sn_pos = df[["dx","dy"]].values.astype(np.float32)
    sersic_radius = df["rSerRadius"].values.astype(np.float32)
    sersic_ab = df["rSerAb"].values.astype(np.float32)
    sersic_phi = df["rSerPhi"].values.astype(np.float32)

    oid_train = np.load(f"..\data\SERSIC\id_train.npy",allow_pickle=True) 
    oid_val = np.load(f"..\data\SERSIC\id_validation.npy",allow_pickle=True) 
    oid_test = np.load(f"..\data\SERSIC\id_test.npy",allow_pickle=True) 

    idx_train = df[df['oid'].isin(oid_train)].index.to_numpy()
    idx_val = df[df['oid'].isin(oid_val)].index.to_numpy()
    idx_test = df[df['oid'].isin(oid_test)].index.to_numpy()

    X_train = images[idx_train]
    X_val = images[idx_val]
    X_test = images[idx_test]

    del images

    y_train = sn_pos[idx_train]
    y_val = sn_pos[idx_val]
    y_test = sn_pos[idx_test]

    train_sersic_radius = None
    train_sersic_ab = None
    train_sersic_phi = None  


    # Limpiamos el conjunto de train 
    # df_train = df.iloc[idx_train]

    # mask_clean_df = obtain_clean_mask(df_train) # Eliminamos perfiles malos
    # mask_accepted_radius = (df_train["rSerRadius"].values < 9.8) # Algunas posiciones quedan fuera del tamaño de la imagen (se corregirá)
    # mask_train = (mask_clean_df & mask_accepted_radius)

    # X_train = X_train[mask_train]
    # y_train = y_train[mask_train]


    if args.augmented_dataset:

        data = np.load("..\data\SERSIC\X_train_augmented_x10.npz")
        X_train = data["imgs"]
        y_train = data["pos"]
        
        #mask_ceros =  (X_train.sum((1,2))==0).any(1)
        #X_train = X_train[~mask_ceros]
        #y_train = y_train[~mask_ceros]

    if args.train_dataset_type == "delight_autolabeling":
        
        X_train = np.load("..\data\SERSIC\X_train_autolabeling.npy")

        if not args.recenter:

            train_sersic_radius = sersic_radius[idx_train]
            train_sersic_ab = sersic_ab[idx_train]
            train_sersic_phi = sersic_phi[idx_train]

            y_train = None

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
    #test_targets = torch.stack([dm.test_dataset()[i][1] for i in range(len(dm.test_dataset()))])
    test_targets = torch.cat([batch[1] for batch in dm.predict_dataloader()], dim=0)

    np.savez(os.path.join(args.save_files, "test_results.npz"), preds=test_preds.numpy(), targets=test_targets.numpy())
#-----------PREDICCIONES-----------#

#-----------RESULTADOS-----------#
    np.save(os.path.join(args.save_files, "curvas.npy"), model.curves)
#-----------RESULTADOS-----------#
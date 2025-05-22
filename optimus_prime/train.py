from dataloader import *
from model import *

import argparse
import os
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/Users/legers/Desktop/Galaxias/tesis/data", help='Dataset path')
    parser.add_argument('--train_augmentation', type=str, default="delight", help='Classic Augmentation or Delight Augmentation')

    parser.add_argument('--lr', type=float, default=0.0014, help='Learning Rate Train')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay Train')

    parser.add_argument('--batch_size', type=int, default=40, help='Training Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num of Workers of dataloaders')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument('--save_files', type=str, default="resultados", help='File name of the results')
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

    X_train = np.load(f"{args.dataset_path}/X_train.npy")
    X_val = np.load(f"{args.dataset_path}/X_validation.npy")
    X_test = np.load(f"{args.dataset_path}/X_test.npy")

    y_train = np.load(f"{args.dataset_path}/y_train.npy")
    y_val = np.load(f"{args.dataset_path}/y_validation.npy")
    y_test = np.load(f"{args.dataset_path}/y_test.npy")

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
                            seed=args.seed, 
                            train_augmentation=args.train_augmentation)

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
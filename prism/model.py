import timm
import torch
import lightning as L
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.inverse_transformation import revert_all_transforms

class Delight(L.LightningModule):

    def __init__(self, config):
        super(Delight, self).__init__()

        if config["backbone"] == "delight":
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(config["channels"], config["nconv1"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv1"], config["nconv2"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv2"], config["nconv3"], 3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

            in_features_regression = 4 * 4 * config["nconv3"] * config["levels"]

        elif config["backbone"] == "resnet18":
            self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet32":
            self.backbone = timm.create_model('resnet32ts.ra2_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]


        self.regression = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.loss = torch.nn.MSELoss()
        self.log_val_loss = True 

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        
        self.test_predictions, self.test_targets = [], []
        self.val_predictions, self.val_targets = [], []

        self.test_mean_preds = None
        self.test_original_targets = None
        self.val_mean_preds = None
        self.val_original_targets = None

        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.backbone(x)
        x = x.reshape(*new_shape)
        x = self.regression(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        train_loss = self.loss(x_hat, y)

        self.log("train/loss", train_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        if self.log_val_loss:
            self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        self.val_predictions.append(x_hat.cpu())
        self.val_targets.append(y.cpu())

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)

        self.test_predictions.append(x_hat.cpu())
        self.test_targets.append(y.cpu())

        return

    def on_validation_epoch_end(self):

        preds = torch.cat(self.val_predictions, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        reverted_preds = revert_all_transforms(preds)
        mean_preds = reverted_preds.mean(1) - 14
        original_targets = targets[:, 0, :]

        self.val_mean_preds = mean_preds
        self.val_original_targets = original_targets

        # limpiar para no acumular en siguiente epoch
        self.val_predictions = []
        self.val_targets = []

    def on_test_epoch_end(self):
        
        preds = torch.cat(self.test_predictions, dim=0)  
        targets = torch.cat(self.test_targets, dim=0)    

        reverted_preds = revert_all_transforms(preds)    
        mean_preds = reverted_preds.mean(1) - 14        

        original_targets = targets[:, 0, :]      

        self.test_mean_preds = mean_preds
        self.test_original_targets = original_targets

        mse = self.loss(mean_preds, original_targets)

        self.log("test/mean_mse", mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optimizer]

class Delight_z(L.LightningModule):

    def __init__(self, config):
        super(Delight_z, self).__init__()

        if config["backbone"] == "delight":
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(config["channels"], config["nconv1"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv1"], config["nconv2"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv2"], config["nconv3"], 3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

            in_features_regression = 4 * 4 * config["nconv3"] * config["levels"]

        elif config["backbone"] == "resnet18":
            self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet32":
            self.backbone = timm.create_model('resnet32ts.ra2_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense_cls"],
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["ndense_cls"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["out_cls"]),
        )


        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        
        self.test_predictions, self.test_targets = [], []

        self.test_mean_preds = None
        self.test_original_targets = None

        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.backbone(x)
        x = x.reshape(*new_shape)
        x = self.regression(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)
        train_loss = self.loss(x_hat.reshape(-1,180), y.reshape(-1))

        self.log("train/z_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat.reshape(-1,180), y.reshape(-1))
        self.log("val/z_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)

        self.test_predictions.append(x_hat.cpu())
        self.test_targets.append(y.cpu())

        return

    def on_test_epoch_end(self):
        
        preds = torch.cat(self.test_predictions, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        self.test_mean_preds = preds
        self.test_original_targets = targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optimizer]  



class Delight_multitask(L.LightningModule):

    def __init__(self, config):
        super(Delight_multitask, self).__init__()

        if config["backbone"] == "delight":
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(config["channels"], config["nconv1"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv1"], config["nconv2"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv2"], config["nconv3"], 3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

            in_features_regression = 4 * 4 * config["nconv3"] * config["levels"]

        elif config["backbone"] == "resnet18":
            self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet32":
            self.backbone = timm.create_model('resnet32ts.ra2_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        self.pos_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.redshift_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense_cls"],
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["ndense_cls"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["out_cls"]),
        )

        self.pos_loss = torch.nn.MSELoss()
        self.redshift_loss = torch.nn.CrossEntropyLoss()

        # GradNorm
        self.automatic_optimization = False
        self.task_weights = torch.nn.Parameter(torch.ones(2))
        self.alpha = config['alpha']

        # pérdidas iniciales para normalización (se llenan en la 1era epoch)
        self.register_buffer("initial_losses", torch.zeros(2))
        self.first_epoch = True

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.test_predictions = []
        self.test_targets = []

        self.test_mean_preds = None
        self.test_original_targets = None

        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.backbone(x)
        x = x.reshape(*new_shape)
        return self.pos_regressor(x), self.redshift_regressor(x)

    def shared_step(self, batch):
        x, pos, z = batch  # image, sn_pos, redshift
        pos_pred, z_pred = self.forward(x)
        reg_loss = self.pos_loss(pos_pred, pos)
        cls_loss = self.redshift_loss(z_pred.squeeze(2), z)
        losses = torch.stack([reg_loss, cls_loss])

        return reg_loss, cls_loss, losses


    def training_step(self, batch, batch_idx):
        
        # opt_w, opt_model = self.optimizers()

        # reg_loss, cls_loss, losses = self.shared_step(batch)

        # # Inicializar pérdidas base para GradNorm
        # if self.first_epoch and batch_idx == 0:
        #     self.initial_losses = losses.detach()
        #     self.first_epoch = False
        

        # weighted_losses = self.task_weights * losses

        # total_loss = weighted_losses.sum()

        # opt_model.zero_grad()
        # self.manual_backward(total_loss, retain_graph=True)

        # # ========= GradNorm ==========
        # last_shared_params = list(self.backbone[-3].parameters())

        # Gw = []
        # for i, li in enumerate(losses):
        #     g = torch.autograd.grad(
        #         self.task_weights[i] * li,
        #         last_shared_params, # ultima capa compartida del backbone
        #         retain_graph=True,
        #         create_graph=True
        #     )
        #     dl = g[0]
        #     Gw.append(dl.norm(2)) 

        # Gw = torch.stack(Gw)
        # Gw_avg = Gw.mean().detach()

        # # relative loss
        # loss_ratios = losses.detach() / self.initial_losses
        # avg_ratio = loss_ratios.mean()
        # r = loss_ratios / avg_ratio

        # target = (Gw_avg * r ** self.alpha).detach()
        # gradnorm_loss = (Gw - target).abs().sum()

        # opt_w.zero_grad()
        # self.manual_backward(gradnorm_loss)

        # opt_model.step()
        # opt_w.step()

        # # Renormalizar pesos
        # with torch.no_grad():
        #     self.task_weights.data *= (len(self.task_weights) /
        #                                self.task_weights.data.sum())

        opt_w, opt_model = self.optimizers()

        reg_loss, cls_loss, losses = self.shared_step(batch)

        # 1. Inicializar pérdidas base (Correcto)
        if self.first_epoch and batch_idx == 0:
            self.initial_losses = losses.detach()
            self.first_epoch = False
        

        # 2. ========= Cálculo de GradNorm ==========
        last_shared_params = list(self.backbone[-3].parameters())

        # Cálculo de Gw (Norma del gradiente) - Este ya está bien con retain_graph=True
        Gw = []
        for i in range(len(losses)):
            g = torch.autograd.grad(
                (self.task_weights[i] * losses[i]),
                last_shared_params,
                retain_graph=True, 
                create_graph=True
            )
            Gw.append(g[0].norm(2)) 

        Gw = torch.stack(Gw)
        Gw_avg = Gw.mean().detach()

        # Cálculo de la pérdida GradNorm (L_GradNorm)
        loss_ratios = losses.detach() / self.initial_losses
        avg_ratio = loss_ratios.mean()
        r = loss_ratios / avg_ratio

        target = (Gw_avg * r ** self.alpha).detach()
        gradnorm_loss = (Gw - target).abs().sum()


        # 3. Optimización de Pesos de Tarea (w_i) con GradNorm
        opt_w.zero_grad()
        # 🔥 CAMBIO CLAVE: Retenemos el grafo aquí para que la retropropagación de total_loss funcione
        self.manual_backward(gradnorm_loss, retain_graph=True) 
        opt_w.step()

        # Renormalizar pesos (Correcto)
        with torch.no_grad():
            self.task_weights.data *= (len(self.task_weights) /
                                       self.task_weights.data.sum())


        # 4. Optimización de Parámetros del Modelo (theta) con Pérdida Total
        weighted_losses = self.task_weights.detach() * losses
        total_loss = weighted_losses.sum()
        
        opt_model.zero_grad()
        # Esta es la última retropropagación, así que no necesita retain_graph=True (por defecto es False)
        self.manual_backward(total_loss) 
        opt_model.step()

        # logs
        for i in range(len(self.task_weights)):
            self.log(f"train/task_weight_{i}", self.task_weights.data[i], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/gradnorm_loss", gradnorm_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # ======================
        # Dataloader 0: 'pos' (Regression)
        # ======================
        if dataloader_idx == 0:
            x, pos = batch
            pos_pred, _ = self.forward(x) 
            reg_loss = self.pos_loss(pos_pred, pos)
            
            self.log("val/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return reg_loss

        # ======================
        # Dataloader 1: 'z' (Redshift/Clasificación)
        # ======================
        elif dataloader_idx == 1:
            x, z = batch
            _, z_pred = self.forward(x) 
            cls_loss = self.redshift_loss(z_pred.squeeze(2), z)

            self.log("val/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return cls_loss
    

    def test_step(self, batch, batch_idx):
        x, pos, _ = batch  # image, sn_pos, redshift
        pos_pred, _ = self.forward(x)

        self.test_predictions.append(pos_pred.cpu())
        self.test_targets.append(pos.cpu())

        return

    def on_test_epoch_end(self):
        
        preds = torch.cat(self.test_predictions, dim=0)  
        targets = torch.cat(self.test_targets, dim=0)    

        reverted_preds = revert_all_transforms(preds)    
        mean_preds = reverted_preds.mean(1) - 14        

        original_targets = targets[:, 0, :]      

        self.test_predictions =  preds
        self.test_targets = targets

        self.test_mean_preds = mean_preds
        self.test_original_targets = original_targets

        mse = self.loss(mean_preds, original_targets)

        self.log("test/mse", mse, prog_bar=True)

    def configure_optimizers(self):

        # optimizer 0: actualiza SOLO task_weights
        optimizer_w = torch.optim.Adam(
            [{"params": self.task_weights}],
            lr=self.lr
        )

        # optimizer 1: actualiza backbone + heads
        optimizer_model = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.pos_regressor.parameters()},
                {"params": self.redshift_regressor.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return [optimizer_w, optimizer_model]


class Delight_multitask_no_GN(L.LightningModule):

    def __init__(self, config):
        super(Delight_multitask_no_GN, self).__init__()

        if config["backbone"] == "delight":
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(config["channels"], config["nconv1"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv1"], config["nconv2"], 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(config["nconv2"], config["nconv3"], 3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

            in_features_regression = 4 * 4 * config["nconv3"] * config["levels"]

        elif config["backbone"] == "resnet18":
            self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet32":
            self.backbone = timm.create_model('resnet32ts.ra2_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 512 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        self.pos_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.redshift_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features_regression,
                out_features=config["ndense_cls"],
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["ndense_cls"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["out_cls"]),
        )

        self.pos_loss = torch.nn.MSELoss()
        self.redshift_loss = torch.nn.CrossEntropyLoss()

        # Ahora dejamos la optimización automática en Lightning
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.test_predictions = []
        self.test_targets = []
        self.test_mean_preds = None
        self.test_original_targets = None

        self.config = config
        self.save_hyperparameters(config)


    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.backbone(x)
        x = x.reshape(*new_shape)
        return self.pos_regressor(x), self.redshift_regressor(x)


    def shared_step(self, batch):
        x, pos, z = batch
        pos_pred, z_pred = self.forward(x)
        reg_loss = self.pos_loss(pos_pred, pos)
        cls_loss = self.redshift_loss(z_pred.squeeze(2), z)
        return reg_loss, cls_loss


    def training_step(self, batch, batch_idx):

        reg_loss, cls_loss = self.shared_step(batch)
        total_loss = reg_loss + cls_loss

        self.log("train/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:  # POS
            x, pos = batch
            pos_pred, _ = self.forward(x)
            reg_loss = self.pos_loss(pos_pred, pos)
            self.log("val/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return reg_loss

        elif dataloader_idx == 1:  # Z
            x, z = batch
            _, z_pred = self.forward(x)
            cls_loss = self.redshift_loss(z_pred.squeeze(2), z)
            self.log("val/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return cls_loss


    def test_step(self, batch, batch_idx):
        x, pos, _ = batch
        pos_pred, _ = self.forward(x)

        self.test_predictions.append(pos_pred.cpu())
        self.test_targets.append(pos.cpu())


    def on_test_epoch_end(self):

        preds = torch.cat(self.test_predictions, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        reverted_preds = revert_all_transforms(preds)
        mean_preds = reverted_preds.mean(1) - 14

        original_targets = targets[:, 0, :]

        self.test_predictions = preds
        self.test_targets = targets
        self.test_mean_preds = mean_preds
        self.test_original_targets = original_targets

        mse = self.loss(mean_preds, original_targets)
        self.log("test/mse", mse, prog_bar=True)


    def configure_optimizers(self):

        optimizer_model = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.pos_regressor.parameters()},
                {"params": self.redshift_regressor.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer_model
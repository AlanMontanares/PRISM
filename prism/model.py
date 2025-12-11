import timm
import torch
import lightning as L
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.inverse_transformation import revert_all_transforms
from utils.pasquet_model import *

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
            in_features_regression = 1536 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        elif config["backbone"] == "pasquet":
            self.backbone = Pasquet_backbone(in_channels=config["channels"])
            in_features_regression = 3132 * config["levels"]


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
        train_loss = self.loss(x_hat, y)

        self.log("train/loss", train_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
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
            in_features_regression = 1536 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        elif config["backbone"] == "pasquet":
            self.backbone = Pasquet_backbone(in_channels=config["channels"])
            in_features_regression = 3132 * config["levels"]

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
        
        self.test_predictions_z = []
        self.test_zphot = None

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
        x, _ = batch
        x_hat = self.forward(x)

        self.test_predictions_z.append(x_hat.cpu())

        return

    def on_test_epoch_end(self):
        
        preds_z = torch.cat(self.test_predictions_z, dim=0)

        soft = torch.nn.Softmax(dim=1)

        probs = soft(preds_z.permute(0, 2, 1))
        mid_point_z = (torch.linspace(0, 0.4, 181)[:-1] + torch.linspace(0, 0.4, 181)[1:]) / 2
        mid_point_z = mid_point_z.view(1, 180, 1)
        zphot  = (probs*mid_point_z).sum(1).mean(1)

        self.test_zphot = zphot

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
            in_features_regression = 1536 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        elif config["backbone"] == "pasquet":
            self.backbone = Pasquet_backbone(in_channels=config["channels"])
            in_features_regression = 3132 * config["levels"]


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
        self.task_weights = torch.nn.Parameter(torch.ones(2)) # parámetros libres
        self.alpha = config['alpha']

        # pérdidas iniciales para normalización (se llenan en la 1era epoch)
        self.register_buffer("initial_losses", torch.zeros(2))
        self.first_epoch = True

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.test_predictions_pos, self.test_targets_pos = [], []
        self.test_mean_preds_pos, self.test_original_targets_pos = None, None

        self.test_predictions_z = []
        self.test_zphot = None

        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
        torch.tensor(original_shape[:-3])
        ).item() # Batch*Transforms*Levels

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
        cls_loss = self.redshift_loss(z_pred.reshape(-1,180), z.reshape(-1))
        losses = torch.stack([reg_loss, cls_loss])

        return reg_loss, cls_loss, losses

    def training_step(self, batch, batch_idx):

        opt_w, opt_model = self.optimizers()

        reg_loss, cls_loss, losses = self.shared_step(batch)

        # ===== (1) Inicializar pérdidas base =====
        if self.first_epoch:
            self.initial_losses = losses.detach()
            self.first_epoch = False

        # ===== (2) Weighted loss forward =====
        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        # ===== (3) Backward normal (solo gradientes de modelo W) =====
        opt_model.zero_grad()
        self.manual_backward(total_loss, retain_graph=True) # NO step aquí

        # ===== (4) Compute GradNorm components =====
        last_shared = list(self.backbone[-3].parameters())

        G_list = []
        for i, Li in enumerate(losses):
            g = torch.autograd.grad(
            self.task_weights[i] * Li,
            last_shared,
            retain_graph=True,
            create_graph=True
            )[0]
            G_list.append(g.norm(2))

        G = torch.stack(G_list)
        G_avg = G.mean().detach()

        loss_ratios = (losses.detach() / self.initial_losses)
        relative = loss_ratios / loss_ratios.mean()

        target = (G_avg * (relative ** self.alpha)).detach()

        gradnorm_loss = (G - target).abs().sum()

        # ===== (5) Backward solo de w_i =====
        opt_w.zero_grad()
        self.manual_backward(gradnorm_loss)

        # ===== (6) Step de w_i =====
        opt_w.step()

        # ===== (7) Step de parámetros del modelo W =====
        opt_model.step()

        # ===== (8) Renormalización final =====
        with torch.no_grad():
            w = self.task_weights.clamp(min=1e-6)
            w = w * (len(w) / w.sum())
            self.task_weights.data = w

        # ===== Logging =====
        for i in range(len(self.task_weights)):
            self.log(f"train/task_weight_{i}", self.task_weights[i], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train/gradnorm_loss", gradnorm_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train/reg_loss", reg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train/cls_loss", cls_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # ======================
        # Dataloader 0: 'pos' (Regression)
        # ======================
        if dataloader_idx == 0:
            x, pos = batch
            pos_pred, _ = self.forward(x) 
            reg_loss = self.pos_loss(pos_pred, pos)

            self.log("val/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return {"reg_loss": reg_loss}

        # ======================
        # Dataloader 1: 'z' (Redshift/Clasificación)
        # ======================
        elif dataloader_idx == 1:
            x, z = batch
            _, z_pred = self.forward(x) 
            cls_loss = self.redshift_loss(z_pred.reshape(-1,180), z.reshape(-1))

            self.log("val/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return {"cls_loss": cls_loss}

    def on_validation_epoch_end(self):
        outs = self.trainer.callback_metrics  # ya está sincronizado y reducido

        total_loss = outs["val/loss"] + outs["val/z_loss"]
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        # ======================
        # Dataloader 0: 'pos' (Regression)
        # ======================
        if dataloader_idx == 0:
            x, pos = batch
            pos_pred, _ = self.forward(x) 

            self.test_predictions_pos.append(pos_pred.cpu())
            self.test_targets_pos.append(pos.cpu())

            return
        # ======================
        # Dataloader 1: 'z' (Redshift/Clasificación)
        # ======================
        elif dataloader_idx == 1:
            x, _ = batch
            _, z_pred = self.forward(x) 

            self.test_predictions_z.append(z_pred.cpu())

            return 

    def on_test_epoch_end(self):
        

        # Galaxy pos
        preds_pos = torch.cat(self.test_predictions_pos, dim=0)  
        targets_pos = torch.cat(self.test_targets_pos, dim=0)    

        reverted_preds = revert_all_transforms(preds_pos)    
        mean_preds = reverted_preds.mean(1) - 14        

        original_targets = targets_pos[:, 0, :]      

        self.test_mean_preds_pos = mean_preds
        self.test_original_targets_pos = original_targets


        # Redshift
        preds_z = torch.cat(self.test_predictions_z, dim=0)

        soft = torch.nn.Softmax(dim=1)

        probs = soft(preds_z.permute(0, 2, 1))
        mid_point_z = (torch.linspace(0, 0.4, 181)[:-1] + torch.linspace(0, 0.4, 181)[1:]) / 2
        mid_point_z = mid_point_z.view(1, 180, 1)
        zphot  = (probs*mid_point_z).sum(1).mean(1)

        self.test_zphot = zphot


    def configure_optimizers(self):

        # optimizer 0: actualiza SOLO task_weights
        optimizer_w = torch.optim.Adam(
            [{"params": self.task_weights}],
            lr=0.025
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
            in_features_regression = 1536 * config["levels"]

        elif config["backbone"] == "resnet50":
            self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0, in_chans=config["channels"])
            in_features_regression = 2048 * config["levels"]

        elif config["backbone"] == "pasquet":
            self.backbone = Pasquet_backbone(in_channels=config["channels"])
            in_features_regression = 3132 * config["levels"]
            
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

        self.test_predictions_pos, self.test_targets_pos = [], []
        self.test_mean_preds_pos, self.test_original_targets_pos = None, None

        self.test_predictions_z = []
        self.test_zphot = None

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
        x, pos, z = batch  # image, sn_pos, redshift
        pos_pred, z_pred = self.forward(x)
        reg_loss = self.pos_loss(pos_pred, pos)
        cls_loss = self.redshift_loss(z_pred.reshape(-1,180), z.reshape(-1))
        
        return reg_loss, cls_loss

    def training_step(self, batch, batch_idx):

        reg_loss, cls_loss = self.shared_step(batch)
        total_loss = reg_loss + cls_loss

        self.log("train/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # ======================
        # Dataloader 0: 'pos' (Regression)
        # ======================
        if dataloader_idx == 0:
            x, pos = batch
            pos_pred, _ = self.forward(x) 
            reg_loss = self.pos_loss(pos_pred, pos)

            self.log("val/loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return {"reg_loss": reg_loss}

        # ======================
        # Dataloader 1: 'z' (Redshift/Clasificación)
        # ======================
        elif dataloader_idx == 1:
            x, z = batch
            _, z_pred = self.forward(x) 
            cls_loss = self.redshift_loss(z_pred.reshape(-1,180), z.reshape(-1))

            self.log("val/z_loss", cls_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return {"cls_loss": cls_loss}

    def on_validation_epoch_end(self):
        outs = self.trainer.callback_metrics  # ya está sincronizado y reducido

        total_loss = outs["val/loss"] + outs["val/z_loss"]
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        # ======================
        # Dataloader 0: 'pos' (Regression)
        # ======================
        if dataloader_idx == 0:
            x, pos = batch
            pos_pred, _ = self.forward(x) 

            self.test_predictions_pos.append(pos_pred.cpu())
            self.test_targets_pos.append(pos.cpu())

            return
        # ======================
        # Dataloader 1: 'z' (Redshift/Clasificación)
        # ======================
        elif dataloader_idx == 1:
            x, _ = batch
            _, z_pred = self.forward(x) 

            self.test_predictions_z.append(z_pred.cpu())

            return 

    def on_test_epoch_end(self):
        

        # Galaxy pos
        preds_pos = torch.cat(self.test_predictions_pos, dim=0)  
        targets_pos = torch.cat(self.test_targets_pos, dim=0)    

        reverted_preds = revert_all_transforms(preds_pos)    
        mean_preds = reverted_preds.mean(1) - 14        

        original_targets = targets_pos[:, 0, :]      

        self.test_mean_preds_pos = mean_preds
        self.test_original_targets_pos = original_targets


        # Redshift
        preds_z = torch.cat(self.test_predictions_z, dim=0)

        soft = torch.nn.Softmax(dim=1)

        probs = soft(preds_z.permute(0, 2, 1))
        mid_point_z = (torch.linspace(0, 0.4, 181)[:-1] + torch.linspace(0, 0.4, 181)[1:]) / 2
        mid_point_z = mid_point_z.view(1, 180, 1)
        zphot  = (probs*mid_point_z).sum(1).mean(1)

        self.test_zphot = zphot


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
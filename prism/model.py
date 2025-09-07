import torch
import lightning as L
import sys
import os
import timm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.inverse_transformation import revert_all_transforms

class Delight(L.LightningModule):

    def __init__(self, config):
        super(Delight, self).__init__()

        self.bottleneck = torch.nn.Sequential(
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

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=4 * 4 * config["nconv3"] * config["levels"],
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.loss = torch.nn.MSELoss()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        
        self.test_predictions = []
        self.test_targets = []

        self.test_mean_preds = None
        self.test_original_targets = None


        self.config = config
        self.save_hyperparameters(config)

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight) #Glorot Uniform
    #             torch.nn.init.constant_(m.bias, 0.1)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.bottleneck(x)
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

        self.test_predictions =  preds
        self.test_targets = targets

        self.test_mean_preds = mean_preds
        self.test_original_targets = original_targets

        mse = self.loss(mean_preds, original_targets)

        self.log("test/mse", mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0, last_epoch=-1)
        return [optimizer]  # , [scheduler]


class Resnet18(L.LightningModule):

    def __init__(self, config):
        super(Resnet18, self).__init__()

        self.bottleneck = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=1)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512 * config["levels"],
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )
        self.loss = torch.nn.MSELoss()
        
        self.classification = config['classification']
        self.regression = config['regression']

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.test_predictions = []
        self.test_targets = []

        self.test_mean_preds = None
        self.test_original_targets = None

        self.config = config
        self.save_hyperparameters(config)

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight) #Glorot Uniform
    #             torch.nn.init.constant_(m.bias, 0.1)

    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.bottleneck(x)
        x = x.reshape(*new_shape)
        x = self.regression(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        train_loss = self.loss(x_hat, y)

        self.log("train/loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

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

        self.test_predictions =  preds
        self.test_targets = targets

        self.test_mean_preds = mean_preds
        self.test_original_targets = original_targets

        mse = self.loss(mean_preds, original_targets)

        self.log("test/mse", mse, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return [optimizer]#, [scheduler]


class Delight_multitask(L.LightningModule):

    def __init__(self, config):
        super(Delight_multitask, self).__init__()

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

        self.pos_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=4 * 4 * config["nconv3"] * config["levels"],
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.redshift_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=4 * 4 * config["nconv3"] * config["levels"],
                out_features=config["ndense_cls"],
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=config["ndense_cls"]*8, out_features=config["ndense_cls"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_cls"], out_features=config["out_cls"]),
        )

        self.pos_loss = torch.nn.MSELoss()
        self.redshift_loss = torch.nn.MSELoss()
        self.grad_loss = torch.nn.L1Loss()

        # GradNorm
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

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight) #Glorot Uniform
    #             torch.nn.init.constant_(m.bias, 0.1)

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

        return self.pos_regressor(x), self.redshift_regressor(x)#self.classificator(x.reshape(x.size(0), -1))

    def shared_step(self, batch, batch_idx):
        x, pos, z = batch  # image, sn_pos, redshift
        pos_pred, z_pred = self.forward(x)

        reg_loss = self.pos_loss(pos_pred, pos)
        cls_loss = self.redshift_loss(z_pred, z)
        losses = torch.stack([reg_loss, cls_loss])

        return reg_loss, cls_loss, losses


    def training_step(self, batch, batch_idx):
        reg_loss, cls_loss, losses = self.shared_step(batch, batch_idx)

        # Guardar pérdidas iniciales
        if self.first_epoch and batch_idx == 0:
            self.initial_losses = losses.detach()

        # Combinar pérdidas con los pesos entrenables
        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        # ======================
        # Paso GradNorm
        # ======================
        shared_params = list(self.backbone.parameters())

        norms = []
        for i, li in enumerate(losses):
            g = torch.autograd.grad(
                self.task_weights[i] * li,
                shared_params,
                retain_graph=True,
                create_graph=True
            )
            norms.append(torch.cat([gi.view(-1) for gi in g]).norm(2))
        norms = torch.stack(norms)

        # tasa de pérdida relativa
        loss_ratios = losses.detach() / self.initial_losses
        avg_loss_ratio = loss_ratios.mean()
        target = (loss_ratios / avg_loss_ratio) ** self.alpha * norms.mean()

        gradnorm_loss = self.grad_loss(norms, target.detach())
        total_loss = total_loss + gradnorm_loss

        # logs
        self.log("train/reg_loss", reg_loss, prog_bar=True, on_epoch=True)
        self.log("train/cls_loss", cls_loss, prog_bar=True, on_epoch=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_epoch=True)

        for i, w in enumerate(self.task_weights.detach().cpu()):
            self.log(f"train/task_weight_{i}", w, prog_bar=True, on_epoch=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        reg_loss, cls_loss, losses = self.shared_step(batch, batch_idx)

        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        self.log("val/reg_loss", reg_loss, prog_bar=True, on_epoch=True)
        self.log("val/cls_loss", cls_loss, prog_bar=True, on_epoch=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True)

        return total_loss

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
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.pos_regressor.parameters()},
                {"params": self.redshift_regressor.parameters()},
                {"params": [self.task_weights], "weight_decay": 0.0},  # <- GradNorm weights
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [optimizer]



class Resnet18_multitask(L.LightningModule):

    def __init__(self, config):
        super(Resnet18_multitask, self).__init__()

        self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0, in_chans=1)

        self.pos_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512 * config["levels"],
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.redshift_regressor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512 * config["levels"],
                out_features=config["ndense_z"],
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=config["ndense_z"]*8, out_features=config["ndense_z"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config["ndense_z"], out_features=config["out_z"]),
        )

        self.pos_loss = torch.nn.MSELoss()
        self.redshift_loss = torch.nn.MSELoss()
        self.grad_loss = torch.nn.L1Loss()

        # GradNorm
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

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight) #Glorot Uniform
    #             torch.nn.init.constant_(m.bias, 0.1)

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

        return self.pos_regressor(x), self.redshift_regressor(x)#self.classificator(x.reshape(x.size(0), -1))

    def shared_step(self, batch, batch_idx):
        x, pos, z = batch  # image, sn_pos, redshift
        pos_pred, z_pred = self.forward(x)

        reg_loss = self.pos_loss(pos_pred, pos)
        cls_loss = self.redshift_loss(z_pred, z)
        losses = torch.stack([reg_loss, cls_loss])

        return reg_loss, cls_loss, losses


    def training_step(self, batch, batch_idx):
        reg_loss, cls_loss, losses = self.shared_step(batch, batch_idx)

        # Guardar pérdidas iniciales
        if self.first_epoch and batch_idx == 0:
            self.initial_losses = losses.detach()

        # Combinar pérdidas con los pesos entrenables
        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        # ======================
        # Paso GradNorm
        # ======================
        shared_params = list(self.backbone.parameters())

        norms = []
        for i, li in enumerate(losses):
            g = torch.autograd.grad(
                self.task_weights[i] * li,
                shared_params,
                retain_graph=True,
                create_graph=True
            )
            norms.append(torch.cat([gi.view(-1) for gi in g]).norm(2))
        norms = torch.stack(norms)

        # tasa de pérdida relativa
        loss_ratios = losses.detach() / self.initial_losses
        avg_loss_ratio = loss_ratios.mean()
        target = (loss_ratios / avg_loss_ratio) ** self.alpha * norms.mean()

        gradnorm_loss = self.grad_loss(norms, target.detach())
        total_loss = total_loss + gradnorm_loss

        # logs
        self.log("train/reg_loss", reg_loss, prog_bar=True, on_epoch=True)
        self.log("train/cls_loss", cls_loss, prog_bar=True, on_epoch=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_epoch=True)

        for i, w in enumerate(self.task_weights.detach().cpu()):
            self.log(f"train/task_weight_{i}", w, prog_bar=True, on_epoch=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        reg_loss, cls_loss, losses = self.shared_step(batch, batch_idx)

        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        self.log("val/reg_loss", reg_loss, prog_bar=True, on_epoch=True)
        self.log("val/cls_loss", cls_loss, prog_bar=True, on_epoch=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True)

        return total_loss

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
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.pos_regressor.parameters()},
                {"params": self.redshift_regressor.parameters()},
                {"params": [self.task_weights], "weight_decay": 0.0},  # <- GradNorm weights
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [optimizer]
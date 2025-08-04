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

        self.training_predictions = []
        self.training_targets = []

        self.val_predictions = []
        self.val_targets = []

        self.test_predictions = []
        self.test_targets = []

        self.test_mean_preds = None
        self.test_original_targets = None

        self.curves = {
            "train_loss": [],
            "val_loss": [],
        }

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

        self.training_predictions.append(x_hat.detach().cpu())
        self.training_targets.append(y.cpu())

        self.log("train/loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        self.val_predictions.append(x_hat.cpu())
        self.val_targets.append(y.cpu())

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)

        self.test_predictions.append(x_hat.cpu())
        self.test_targets.append(y.cpu())

        return

    def on_train_epoch_end(self):

        predictions = torch.cat(self.training_predictions, dim=0)
        targets = torch.cat(self.training_targets, dim=0)

        self.curves["train_loss"].append(self.loss(predictions, targets).item())

        self.training_predictions.clear()
        self.training_targets.clear()

    def on_validation_epoch_end(self):

        predictions = torch.cat(self.val_predictions, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        self.curves["val_loss"].append(self.loss(predictions, targets).item())

        self.val_predictions.clear()
        self.val_targets.clear()

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

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512 * config["levels"],
                out_features=config["ndense"],
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config["dropout"]),
            torch.nn.Linear(in_features=config["ndense"], out_features=2),
        )

        self.loss = torch.nn.MSELoss()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.training_predictions = []
        self.training_targets = []

        self.val_predictions = []
        self.val_targets = []

        self.test_predictions = []
        self.test_targets = []

        self.test_mean_preds = None
        self.test_original_targets = None

        self.curves = {
            "train_loss": [],
            "val_loss": [],
        }

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

        self.training_predictions.append(x_hat.detach().cpu())
        self.training_targets.append(y.cpu())

        self.log("train/loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        self.val_predictions.append(x_hat.cpu())
        self.val_targets.append(y.cpu())

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)

        self.test_predictions.append(x_hat.cpu())
        self.test_targets.append(y.cpu())

        return

    def on_train_epoch_end(self):

        predictions = torch.cat(self.training_predictions, dim=0)
        targets = torch.cat(self.training_targets, dim=0)

        self.curves["train_loss"].append(self.loss(predictions, targets).item())

        self.training_predictions.clear()
        self.training_targets.clear()

    def on_validation_epoch_end(self):

        predictions = torch.cat(self.val_predictions, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        self.curves["val_loss"].append(self.loss(predictions, targets).item())

        self.val_predictions.clear()
        self.val_targets.clear()

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return [optimizer], [scheduler]

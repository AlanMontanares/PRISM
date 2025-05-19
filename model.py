import torch
import lightning as L


class Delight(L.LightningModule):

    def __init__(self, config):
        super(Delight, self).__init__()

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(config.channels, config.nconv1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(config.nconv1, config.nconv2, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(config.nconv2, config.nconv3, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=4 * 4 * config.nconv3 * config.levels,
                out_features=config.ndense,
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config.dropout),
            torch.nn.Linear(in_features=config.ndense, out_features=2),
        )

        self.loss = torch.nn.MSELoss()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.training_predictions = []
        self.training_classes = []

        self.val_predictions = []
        self.val_classes = []

        self.curves = {
            "train_loss": [],
            "val_loss": [],
        }

        self.save_files = config["save_files"]

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
        self.training_classes.append(y.cpu())

        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x_hat = self.forward(x)
        val_loss = self.loss(x_hat, y)

        self.val_predictions.append(x_hat.cpu())
        self.val_classes.append(y.cpu())

        self.log("val_loss", val_loss, prog_bar=True)

        return val_loss

    def on_train_epoch_end(self):

        predictions = torch.cat(self.training_predictions, dim=0)
        classes = torch.cat(self.training_classes, dim=0)

        self.curves["train_loss"].append(self.loss(predictions, classes))

        self.training_predictions.clear()
        self.training_classes.clear()

    def on_validation_epoch_end(self):

        predictions = torch.cat(self.val_predictions, dim=0)
        classes = torch.cat(self.val_classes, dim=0)

        self.curves["val_loss"].append(self.loss(predictions, classes))

        self.val_predictions.clear()
        self.val_classes.clear()

    def predict_step(self, batch):
        x, ebv, _ = batch
        return self(x, ebv)

    def configure_optimizers(self):
        # lr_lambda = lambda epoch: 0.95 ** epoch

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0, last_epoch=-1)
        return [optimizer]  # , [scheduler]

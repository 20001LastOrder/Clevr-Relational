import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


class AttrNetClassificationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.output_dims = self.hparams.output_dims
        self.net = _Net(self.output_dims)

    def forward(self, im_features, obj_features):
        predictions = self.net(im_features, obj_features)
        return predictions

    def get_metrics(self, batch):
        im_features, obj_features, labels, _, _ = batch
        predictions = self.forward(im_features, obj_features)
        loss = sum([self.criterion(prediction, label) for prediction, label in zip(predictions, labels)])
        accuracy = [self.accuracy(torch.argmax(prediction, dim=1), label) for prediction, label in
                    zip(predictions, labels)]
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/train', loss)
        for i, acc in enumerate(accuracy):
            self.log(f'acc_{i}/train', acc)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/val', loss)
        for i, acc in enumerate(accuracy):
            self.log(f'acc_{i}/val', acc)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        for i, acc in enumerate(accuracy):
            self.log(f'acc_{i}/test', acc)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [scheduler]


class _Net(nn.Module):

    def __init__(self, output_dims):
        super(_Net, self).__init__()

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512)
        )

        self.output_layers = nn.ModuleList()
        for output_dim in output_dims:
            self.output_layers.append(nn.Linear(512, output_dim))

    def forward(self, im_features, obj_features):
        x = self.avg_pool(im_features)
        x = self.fc(torch.hstack([x, obj_features]))
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(x))
        return outputs


def get_model(opt):
    model = AttrNetClassificationModule(opt)
    return model

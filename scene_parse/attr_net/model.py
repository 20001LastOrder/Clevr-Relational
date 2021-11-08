import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


class AttrNetClassificationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        if self.hparams.concat_img:
            self.input_channels = 6
        else:
            self.input_channels = 3

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        # TODO: Parameterize
        self.output_dims = [8, 3, 2, 2]
        self.net = _Net(self.output_dims, self.input_channels)

    def forward(self, images):
        predictions = self.net(images)
        return predictions

    def get_metrics(self, batch):
        images, labels, _, _ = batch
        predictions = self.forward(images)
        loss = sum([self.criterion(prediction, label) for prediction, label in zip(predictions, labels)])
        accuracy = [self.accuracy(torch.argmax(prediction, dim=1), label) for prediction, label in zip(predictions, labels)]
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

    def __init__(self, output_dims, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        self.feature_extractor = nn.Sequential(*layers)
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
        self.output_layers = nn.ModuleList()
        for output_dim in output_dims:
            self.output_layers.append(nn.Linear(512, output_dim))

        # layers = list(resnet.children())
        
        # # remove the last layer
        # layers.pop()
        # # remove the first layer as we take a 6-channel input
        # layers.pop(0)
        # layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        # self.main = nn.Sequential(*layers)
        # self.final_layer = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(x))
        return outputs


def get_model(opt):
    model = AttrNetClassificationModule(opt)
    return model

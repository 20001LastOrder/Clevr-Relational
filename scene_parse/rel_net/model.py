import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


PYTORCH_VER = torch.__version__

class RelNetClassificationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

  
        self.input_channels = 1

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        # TODO: Parameterize
        self.output_dim = 2
        self.net = _Net(self.output_dim, self.input_channels)

    def forward(self, s):
        predictions = self.net(s)
        return predictions

    def get_metrics(self, batch):
        sources, labels, _, _, _ = batch
        predictions = self.forward(sources)
        loss =  self.criterion(predictions, labels)
        accuracy = self.accuracy(torch.argmax(predictions, dim=1), labels)
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log(f'acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log(f'acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log(f'acc/test', accuracy)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [scheduler]


class _Net(nn.Module):

    def __init__(self, output_dim, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.feature_extractor = nn.Sequential(*layers)
        
        self.output_layers = nn.ModuleList()
        self.output_layer = nn.Linear(512, output_dim)

        # layers = list(resnet.children())
        
        # # remove the last layer
        # layers.pop()
        # # remove the first layer as we take a 6-channel input
        # layers.pop(0)
        # layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        # self.main = nn.Sequential(*layers)
        # self.final_layer = nn.Linear(512, output_dim)

    def forward(self, s):
        s = self.feature_extractor(s)
        s = s.view(s.size(0), -1)

        return self.output_layer(s)


def get_model(opt):
    model = RelNetModule(opt)
    return model


class RelNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        self.criterion = torch.nn.BCELoss()
        self.accuracy = Accuracy()

        # TODO: parameterize other parameters
        self.net = _RelNet(self.hparams.num_rels)

    def forward(self, source, rel, target):
        predictions = self.net(source, rel, target)
        return predictions

    def get_metrics(self, batch):
        sources, rel, targets, labels = batch
        predictions = self.forward(sources, rel, targets).squeeze()
        loss = self.criterion(predictions, labels.float())
        accuracy = self.accuracy(torch.round(predictions), labels)
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log(f'acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log(f'acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.get_metrics(batch)
        self.log(f'acc/test', accuracy)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [scheduler]


class _RelNet(nn.Module):
    def __init__(self, num_rels, input_channels=4, num_features=512, hidden_size=128):
        super(_RelNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.num_features = num_features
        self.feature_extractor = nn.Sequential(*layers)
        # self.feature_fc = nn.Linear(512, num_features, bias=True)

        self.rel_embedding = nn.Embedding(num_rels, num_features)

        self.rnn = nn.RNN(num_features, hidden_size, batch_first=False)

        self.output = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def _feature_extractor(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, source, rel, target):
        source = self._feature_extractor(source).unsqueeze(0)
        rel = self.rel_embedding(rel).unsqueeze(0)
        target = self._feature_extractor(target).unsqueeze(0)

        combined = torch.cat([source, rel, target], dim=0)
        combined, _ = self.rnn(combined)
        output = self.output(combined[2])
        return output

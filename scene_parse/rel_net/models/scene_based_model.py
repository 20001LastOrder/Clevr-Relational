import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from scene_parse.rel_net.constraints import build_adjacency_matrix
from scene_parse.rel_net.constraints.constraint_loss import get_dag_constraint, get_anti_symmetry_constraint


class SceneBasedRelNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        self.criterion = torch.nn.BCELoss()

        constraints = [get_dag_constraint(epsilon=0.05), get_anti_symmetry_constraint(epsilon=-0.5)]
        self.logic_criteria = lambda a: (constraints[0].cal_loss(a) + constraints[1].cal_loss(a)) / torch.numel(a)

        self.net = _RelNet(self.hparams.num_rels, self.hparams.dropout_p)

    def forward(self, data, sources, targets, labels):
        predictions = self.net(data, sources, targets, labels)
        return predictions

    def get_metrics(self, batch):
        data, sources, targets, labels, _ = batch
        data, sources, targets = data.squeeze(dim=0), sources.squeeze(dim=0), targets.squeeze(dim=0)
        labels = labels.squeeze(dim=0)

        predictions = self.forward(data, sources, targets, labels)
        loss = self.criterion(predictions, labels.float())

        adj = build_adjacency_matrix(predictions, int(math.sqrt(sources.shape[0])))
        constraint_loss = self.logic_criteria(adj)

        predictions = torch.round(predictions)

        # get axccuracy for each type of the relationship
        accuracies = (predictions == labels).sum(dim=0) / predictions.shape[0]

        return loss, constraint_loss, accuracies

    def training_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log('loss/train', loss)
        self.log('constraint_loss/train', constraint_loss)
        self.log(f'acc_overall/train', accuracies.mean())
        self.log('lr', self.scheduler.get_last_lr()[0])
        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/train', acc)

        return 0.8 * loss + 0.2 * constraint_loss if self.hparams.include_constraint_loss else loss

    def validation_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log('loss/val', loss)
        self.log('constraint_loss/val', constraint_loss)
        self.log(f'acc_overall/val', accuracies.mean())

        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/val', acc)
            self.log(f'scene_acc{i}/val', 1 if np.isclose(1, acc.item()) else 0)

    def test_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log(f'acc_overall/test', accuracies.mean())
        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/test', acc)

        return accuracies

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        self.scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [self.scheduler]


class _RelNet(nn.Module):
    def __init__(self, num_rels, dropout_p, input_channels=4, num_features=512):
        super(_RelNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.num_features = num_features
        self.feature_extractor = nn.Sequential(*layers)

        # self.rnn = nn.Sequential(nn.Dropout(dropout_p), nn.RNN(num_features, 128, batch_first=False))
        # self.output = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(128, num_rels),
        #                             nn.Sigmoid())
        self.output = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(2 * num_features, 256), nn.Linear(256, num_rels),
                                    nn.Sigmoid())

    def _feature_extractor(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, objs, sources, targets, labels):
        features = self._feature_extractor(objs)
        relations = torch.cat([features[sources], features[targets]], dim=1)
        # source = features[sources].unsqueeze(0)
        # target = features[targets].unsqueeze(0)
        # combined = torch.cat([source, target], dim=0)
        # combined, _ = self.rnn(combined)
        # output = self.output(combined[-1])
        # return output

        return self.output(relations)

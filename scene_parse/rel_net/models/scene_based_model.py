import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from scene_parse.rel_net.constraints import build_adjacency_matrix, adj_probability
from scene_parse.rel_net.constraints.constraint_loss import get_deduct_constraint, get_anti_symmetry_constraint,\
    get_transitivity_constraint


class SceneBasedRelNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)

        self.criterion = torch.nn.BCELoss()

        constraints = [get_deduct_constraint([(0, 1), (2, 3)]), get_anti_symmetry_constraint(epsilon=-0.5),
                       get_transitivity_constraint()]
        self.logic_criteria = lambda a: (constraints[0](a) + constraints[1](a) + constraints[2](a)) / torch.numel(a)

        num_rels = self.hparams.num_rels if self.hparams.used_rels is None else len(self.hparams.used_rels)
        self.net = _RelNet(num_rels, self.hparams.dropout_p, self.hparams.use_sigmoid)

    def forward(self, data, sources, targets):
        predictions = self.net(data, sources, targets)
        return predictions

    def get_metrics(self, batch):
        data, sources, targets, labels, _, (nums_obj, nums_edges), _ = batch
        # data, sources, targets = data.squeeze(dim=0), sources.squeeze(dim=0), targets.squeeze(dim=0)
        # labels = labels.squeeze(dim=0)
        n = data.shape[0]
        accuracies = torch.zeros((n, labels.shape[2]))  # accuracies for each label type
        losses = torch.zeros(n)
        constraint_losses = torch.zeros(n)

        # TODO: Find a better way to handle multi scenes in one epoch
        for i in range(n):
            num_obj = nums_obj[i].item()
            num_edges = nums_edges[i].item()
            data_i = data[i][:num_obj]
            sources_i = sources[i][:num_edges].long()
            targets_i = targets[i][:num_edges].long()
            labels_i = labels[i][:num_edges].long()

            predictions = self.forward(data_i, sources_i, targets_i)

            adj_size = int(math.sqrt(predictions.shape[0]))
            adj = build_adjacency_matrix(predictions, adj_size)

            if not self.hparams.use_sigmoid:
                adj = adj_probability(adj)
            adj_gt = build_adjacency_matrix(labels_i, adj_size)
            if self.hparams.include_constraint_loss:
                constraint_losses[i] = self.logic_criteria(adj)

            losses[i] = self.criterion(predictions, labels_i.float())

            adj = torch.round(adj)
            accuracies[i] = (adj == adj_gt).sum(dim=[1, 2]) / predictions.shape[0]

        return losses.mean(), constraint_losses.mean(), accuracies.mean(dim=0)

    def training_step(self, batch, batch_nb):
        loss, constraint_loss, accuracies = self.get_metrics(batch)
        self.log('loss/train', loss)
        if self.hparams.include_constraint_loss:
            self.log('constraint_loss/train', constraint_loss)
        self.log(f'acc_overall/train', accuracies.mean())
        self.log('lr', self.scheduler.get_last_lr()[0])
        for i, acc in enumerate(accuracies):
            self.log(f'acc_{i}/train', acc)

        # labelled = batch[-1].item()

        # return loss if labelled else constraint_loss
        return 0.4 * loss + 0.6 * constraint_loss if self.hparams.include_constraint_loss else loss

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
    def __init__(self, num_rels, dropout_p, use_sigmoid, input_channels=4, num_features=512):
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
        output_layer = [nn.Dropout(dropout_p), nn.Linear(2 * num_features, 256), nn.Linear(256, num_rels)]
        if use_sigmoid:
            output_layer.append(nn.Sigmoid())
        self.output = nn.Sequential(*output_layer)

    def _feature_extractor(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, objs, sources, targets):
        features = self._feature_extractor(objs)
        relations = torch.cat([features[sources], features[targets]], dim=1)

        return self.output(relations)


# class _RelNet(nn.Module):
#     def __init__(self, num_rels, dropout_p, use_sigmoid, input_channels=4, num_features=512):
#         super(_RelNet, self).__init__()
#
#         resnet = models.resnet34(pretrained=True)
#         layers = list(resnet.children())
#         layers.pop()
#         layers.pop(0)
#         layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
#         self.num_features = num_features
#         self.feature_extractor = nn.Sequential(*layers)
#
#         self.rnn = nn.Sequential(nn.Dropout(dropout_p), nn.RNN(num_features, 128, batch_first=False))
#         output_layer = [nn.Dropout(dropout_p), nn.Linear(128, num_rels)]
#         # output_layer = [nn.Dropout(dropout_p), nn.Linear(2 * num_features, 256), nn.Linear(256, num_rels)]
#         if use_sigmoid:
#             output_layer.append(nn.Sigmoid())
#         self.output = nn.Sequential(*output_layer)
#
#     def _feature_extractor(self, x):
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, objs, sources, targets):
#         # features = self._feature_extractor(objs)
#         # relations = torch.cat([features[sources], features[targets]], dim=1)
#
#         features = self._feature_extractor(objs).unsqueeze(0)
#         relations = torch.cat([features[:, sources], features[:, targets]], dim=0)
#         combined, _ = self.rnn(relations)
#         return self.output(combined[-1])

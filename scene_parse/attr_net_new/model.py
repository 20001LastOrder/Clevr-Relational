import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


PYTORCH_VER = torch.__version__

class AttrNetClassificationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

#         if self.hparams.concat_img:
#             self.input_channels = 6
#         else:
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
    
    
class AttributeClassificationNetwork():

    def __init__(self, opt, output_dim):    
#         if opt.concat_img:
#             self.input_channels = 6
#         else:
        self.input_channels = 3

        if opt.load_checkpoint_path:
            print('| loading checkpoint from %s' % opt.load_checkpoint_path)
            checkpoint = torch.load(opt.load_checkpoint_path)
            if self.input_channels != checkpoint['input_channels']:
                raise ValueError('Incorrect input channels for loaded model')
            self.output_dim = checkpoint['output_dim']
            self.net = _Net(self.output_dim, self.input_channels)
            self.net.load_state_dict(checkpoint['model_state'])
        else:
            print('| creating new model')
            self.output_dim = output_dim
            self.net = _Net(self.output_dim, self.input_channels)

        self.criterion = nn.CrossEntropyLoss()#nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)

        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.net.cuda(opt.gpu_ids[0])

        self.input, self.label = None, None
                
    def set_input(self, x, y=None):
        self.input = self._to_var(x)
        if y is not None:
            self.label = self._to_var(y)

    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        self.optimizer.step()

    def forward(self):
        self.pred = self.net(self.input)
        if self.label is not None:
            self.loss = self.criterion(self.pred, self.label)
            
    def get_loss(self):
        return self.loss.data.item()

    def get_pred(self):
        return self.pred.data.cpu().numpy()

    def eval_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def save_checkpoint(self, save_path):
        checkpoint = {
            'input_channels': self.input_channels,
            'output_dim': self.output_dim,
            'model_state': self.net.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])

    def _to_var(self, x):
        if self.use_cuda:
            x = x.cuda()
        return x


class _Net(nn.Module):

    def __init__(self, output_dims, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        layers.pop()
        self.feature_extractor = nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
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
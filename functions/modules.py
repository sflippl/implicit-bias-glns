
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .gln import GLN
from .relu_net import FRN

class Module(LightningModule):
    def __init__(self, model, hparams=None, **kwargs):
        super().__init__()
        hparams = self.get_hparams(hparams, **kwargs)
        self.save_hyperparameters(hparams)
        self.model = model

    def get_hparams(self, hparams=None, **kwargs):
        if hparams is None:
            parser = argparse.ArgumentParser()
            parser = self.add_argparse_args(parser)
            hparams = parser.parse_args([])
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        for key, value in kwargs.items():
            hparams.__setattr__(key, value)
        return hparams

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        x, y = batch
        outp = self(x)[:,0]
        y_hat = torch.sigmoid(outp)
        loss = F.binary_cross_entropy(y_hat, y.float())
        acc = (outp.sign()==(2*y-1)).float().mean()
        return {'loss': loss, 'acc': acc}
    
    def training_step(self, batch, batch_idx):
        dct = self.step(batch, batch_idx)
        self.log_dict({f'train_{k}': v for k, v in dct.items()})
        return dct['loss']

    def validation_step(self, batch, batch_idx):
        dct = self.step(batch, batch_idx)
        self.log_dict({f'val_{k}': v for k, v in dct.items()})

    def parameters(self):
        return self.model.parameters()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=0.)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.decay_steps, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--momentum', type=float, default=0.)
        parser.add_argument('--decay_steps', type=float, nargs='+', default=[100])
        parser.add_argument('--gamma', type=float, default=0.1)
        return parser

class GLNModule(Module):
    def __init__(self, hparams=None, **kwargs):
        hparams = self.get_hparams(hparams, **kwargs)
        self.save_hyperparameters(hparams)
        model = GLN(
            self.hparams.inp_dim, self.hparams.latent_dims, 1, self.hparams.shatter_dims, bias=(not self.hparams.no_bias), seed=hparams.context_seed
        )
        super().__init__(model, hparams=hparams)

    @staticmethod
    def add_argparse_args(parser):
        parser = Module.add_argparse_args(parser)
        parser.add_argument('--inp_dim', type=int, default=785)
        parser.add_argument('--latent_dims', type=int, nargs='+', default=[])
        parser.add_argument('--shatter_dims', type=int, nargs='+', default=[3])
        parser.add_argument('--no_bias', action='store_true')
        parser.add_argument('--context_seed', type=int, default=1)
        return parser
    
    def quantile_init_beta(self, x):
        self.model.quantile_init_beta(x)
    
    def has_median_init(self):
        return True

class ReLUModule(Module):
    def __init__(self, hparams=None, **kwargs):
        hparams = self.get_hparams(hparams, **kwargs)
        self.save_hyperparameters(hparams)
        model = FRN(
            self.hparams.inp_dim, self.hparams.hidden_dims, 1, init_seed=self.hparams.init_seed, bias=(not self.hparams.no_bias)
        )
        super().__init__(model, hparams=hparams)

    @staticmethod
    def add_argparse_args(parser):
        parser = Module.add_argparse_args(parser)
        parser.add_argument('--inp_dim', type=int, default=785)
        parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
        parser.add_argument('--init_seed', type=int, default=1)
        parser.add_argument('--no_bias', action='store_true')
        return parser
    
    def has_median_init(self):
        return False

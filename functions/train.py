import argparse
import os

import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import MNIST
from torchvision import transforms

from .mnist import get_mnist, get_mnist_dataloader, split_mnist

def get_parser(model_class):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='data', type=str)
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--version', type=str, default='default')
    parser.add_argument('--checkpoint_path', default='data', type=str)
    parser.add_argument('--train_size', default=48000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_seed', default=1, type=int)
    parser.add_argument('--binary_labels', nargs=2, default=[[0,1,2,3,4],[5,6,7,8,9]], type=eval)
    parser.add_argument('--fit_seed', default=1, type=int)
    parser.add_argument('--median_init', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = model_class.add_argparse_args(parser)
    return parser

def train(args, model_class):
    mnist = get_mnist(args.binary_labels)
    train_loader, val_loader = get_mnist_dataloader(mnist, args.data_seed, args.train_size, batch_size=args.batch_size)
    model = model_class(args)
    if args.median_init and model.has_median_init():
        train_set, val_set = split_mnist(mnist, args.data_seed, args.train_size)
        x = torch.stack([x for x, y in mnist])
        model.quantile_init_beta(x)
    args.logger = CSVLogger(
        args.save_dir, name=args.experiment_name, version=args.version
    )
    args.callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.experiment_name, args.version),
            save_last=True,
            save_top_k=0
        )
    ]
    print(len(train_loader))
    trainer = pl.Trainer.from_argparse_args(args)
    pl.seed_everything(args.fit_seed)
    trainer.fit(model, train_loader, val_loader)

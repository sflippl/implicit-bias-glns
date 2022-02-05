import sys
import os
import argparse
from pathlib import Path
sys.path.append('')

import torch
import pandas as pd

from functions.train_convex import train_convex, get_loee
from functions.mnist import get_mnist, split_mnist
from functions.modules import GLNModule

@torch.no_grad()
def get_data(dataset, gln, shatter_dims, latent_dims):
    x = torch.stack([_x for _x, _y in dataset])
    y = 2*torch.stack([_y for _x, _y in dataset])-1
    if shatter_dims[0] > 0:
        context = gln.model.funs[0].context(x)
    else:
        context = torch.ones((x.shape[0],latent_dims[0],1))
    return {'x': x.numpy(), 'y': y.numpy(), 'context': context.numpy()}

def main(args):
    mnist = get_mnist(args.binary_labels)
    train_set, val_set = split_mnist(mnist, args.data_seed, args.train_size)
    latent_dims = [args.latent_dims]
    shatter_dims = [args.shatter_dims, 0]
    gln = GLNModule(latent_dims=latent_dims, shatter_dims=shatter_dims, context_seed=args.context_seed)
    if args.median_init:
        train_set, val_set = split_mnist(mnist, args.data_seed, args.train_size)
        x = torch.stack([x for x, y in mnist])
        gln.quantile_init_beta(x)
    train_data = get_data(train_set, gln, shatter_dims, latent_dims)
    val_data = get_data(val_set, gln, shatter_dims, latent_dims)
    if args.comparison_model is not None:
        comparison_model = GLNModule.load_from_checkpoint(args.comparison_model)
        assert comparison_model.hparams.train_size == args.train_size
        #assert comparison_model.hparams.latent_dims[0] == args.latent_dims
        assert comparison_model.hparams.context_seed == args.context_seed
    else:
        comparison_model = None
    accs, args_to_loee = train_convex(args.constraints, train_data, val_data, max_iters=args.max_iters, comparison_model=comparison_model)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if len(accs) == 2:
        acc_type = ['train', 'val']
    else:
        acc_type = ['train', 'val', 'comparison']
    df = pd.DataFrame({
        'latent_dims': [args.latent_dims]*len(accs),
        'shatter_dims': [args.shatter_dims]*len(accs),
        'train_size': [args.train_size]*len(accs),
        'context_seed': [args.context_seed]*len(accs),
        'data_seed': [args.data_seed]*len(accs),
        'type': acc_type,
        'acc': accs
    })
    df.to_csv(os.path.join(args.save_dir, 'accs.csv'))
    print(f'Training Accuracy: {accs[0]}')
    print(f'Validation Accuracy: {accs[1]}')
    if args.constraints == 'none':
        loee_df = get_loee(*args_to_loee, compute_radius=not args.no_radius)
        loee_df.to_csv(os.path.join(args.save_dir, 'bounds.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_labels', type=eval, default=[[0,1,2,3,4],[5,6,7,8,9]])
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--latent_dims', type=int, default=1)
    parser.add_argument('--shatter_dims', type=int, default=1)
    parser.add_argument('--context_seed', type=int, default=1)
    parser.add_argument('--median_init', action='store_true')
    parser.add_argument('--constraints', choices=['none', 'architecture', 'implicit_bias'], default='none')
    parser.add_argument('--max_iters', type=int, default=200)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--no_radius', action='store_true')
    parser.add_argument('--comparison_model', default=None, type=str)
    args = parser.parse_args()
    main(args)

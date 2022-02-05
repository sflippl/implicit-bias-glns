import sys
import os
import argparse
from pathlib import Path
sys.path.append('')

import torch
import torch.nn as nn
import pandas as pd

from functions.train_convex_relu import train_convex_relu
from functions.mnist import get_mnist, split_mnist
from functions.modules import ReLUModule

@torch.no_grad()
def get_data(dataset, contexts, objective):
    x = torch.stack([_x for _x, _y in dataset])
    y = 2*torch.stack([_y for _x, _y in dataset])-1
    if objective == 'gates_only':
        _, gates = contexts.model(x, return_gates=True)
        gates = gates[0]
        return {'x': gates.numpy(), 'y': y.numpy(), 'comp_x': x.numpy()}
    elif objective == 'hidden_layer':
        hidden_layer = (contexts.model.gates[0](x)>=0.).float()*contexts.model.weights[0](x)
        return {'x': hidden_layer.numpy(), 'y': y.numpy(), 'comp_x': x.numpy()}
    elif objective == 'random_contexts':
        context = contexts(x)
        return {'x': x.numpy(), 'y': y.numpy(), 'context': context.numpy(), 'comp_x': x.numpy()}
    elif objective == 'learned_contexts':
        _, gates = contexts.model(x, return_gates=True)
        gates = gates[0]
        return {'x': x.numpy(), 'y': y.numpy(), 'context': gates.numpy(), 'comp_x': x.numpy()}
    else:
        raise NotImplementedError()

class RandomContext(nn.Module):
    def __init__(self, inp_dim, outp_dim, seed=1):
        super().__init__()
        self.linear = nn.Linear(inp_dim, outp_dim)
        torch.manual_seed(seed)
        nn.init.orthogonal_(self.linear.weight)
        self.beta = torch.zeros((outp_dim,))

    def quantile_init_beta(self, x):
        outp = self.linear(x)
        self.beta, _ = torch.median(outp, dim=0)

    def forward(self, x):
        outp = (self.linear(x) >= self.beta.unsqueeze(0)).float()
        outp = torch.cat([outp, torch.ones(outp.shape[0], 1)], dim=1)
        return outp

def main(args):
    mnist = get_mnist(args.binary_labels)
    train_set, val_set = split_mnist(mnist, args.data_seed, args.train_size)
    model = ReLUModule.load_from_checkpoint(args.model_path)
    if args.objective == 'random_contexts':
        contexts = RandomContext(785, model.hparams.hidden_dims[0], seed=args.context_seed)
        train_set, val_set = split_mnist(mnist, args.data_seed, args.train_size)
        x = torch.stack([x for x, y in mnist])
        contexts.quantile_init_beta(x)
    else:
        contexts = model
    train_data = get_data(train_set, contexts, args.objective)
    val_data = get_data(val_set, contexts, args.objective)
    accs = train_convex_relu(args.objective, train_data, val_data, max_iters=args.max_iters, comparison_model=model)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'model': [args.model_path]*len(accs),
        'train_size': [args.train_size]*len(accs),
        'context_seed': [args.context_seed]*len(accs),
        'data_seed': [args.data_seed]*len(accs),
        'objective': [args.objective]*len(accs),
        'type': ['train', 'val', 'comparison'],
        'acc': accs
    })
    df.to_csv(os.path.join(args.save_dir, 'accs.csv'))
    print(f'Training Accuracy: {accs[0]}')
    print(f'Validation Accuracy: {accs[1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_labels', type=eval, default=[[0,1,2,3,4],[5,6,7,8,9]])
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--context_seed', type=int, default=1)
    parser.add_argument('--objective', choices=['gates_only', 'hidden_layer', 'learned_contexts', 'random_contexts'], default='learned_contexts')
    parser.add_argument('--max_iters', type=int, default=200)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)

import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

argparse_array = ArgparseArray(
    max_epochs=(lambda array_id, train_size, **kwargs: int(200*2000/train_size)),
    shatter_dims=[[i] for i in range(16)],
    experiment_name='shallow_gln',
    version=(lambda array_id, shatter_dims, context_seed, momentum, train_size, median_init, **kwargs: f'shd_{shatter_dims[0]}-cs_{context_seed}-mom_{momentum}-ts_{train_size}-mi_{median_init}'),
    lr=0.04,
    log_every_n_steps=(lambda array_id, train_size, **kwargs: int(train_size/128)+1),
    context_seed=[1,2,3],
    momentum=[0., 0.9],
    train_size=[500, 1000, 2000],
    batch_size=128,
    gamma=0.25,
    median_init=[True, False],
    decay_steps=(lambda array_id, train_size, **kwargs: [int(100*2000/train_size)])
)

def main(args):
    argparse_array.call_script('scripts/train_gln.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)

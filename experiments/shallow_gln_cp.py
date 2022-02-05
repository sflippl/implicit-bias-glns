import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

argparse_array = ArgparseArray(
    shatter_dims=list(range(9)),
    save_dir=lambda array_id, shatter_dims, context_seed, train_size, median_init, **kwargs: f'data/shallow_gln_cp/shd_{shatter_dims}-cs_{context_seed}-ts_{train_size}-mi_{median_init}',
    train_size=[500, 1000, 2000],
    context_seed=[1,2,3],
    median_init=[True, False],
    comparison_model=(lambda array_id, train_size, context_seed, shatter_dims, median_init, **kwargs: f'data/shallow_gln/shd_{shatter_dims}-cs_{context_seed}-mom_0.0-ts_{train_size}-mi_{median_init}/last.ckpt')
)

def main(args):
    argparse_array.call_script('scripts/train_convex.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)

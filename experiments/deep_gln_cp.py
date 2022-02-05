import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

argparse_array = ArgparseArray(
    shatter_dims=[1, 2],
    latent_dims=[10, 20, 50, 100],
    save_dir=(lambda array_id, shatter_dims, context_seed, train_size, median_init, latent_dims, constraints, **kwargs: f'data/deep_gln_cp/shd_{shatter_dims}-ld_{latent_dims}-cs_{context_seed}-ts_{train_size}-mi_{median_init}-constraints_{constraints}'),
    comparison_model=(lambda array_id, shatter_dims, context_seed, train_size, median_init, latent_dims, **kwargs: f'data/deep_gln/shd_{shatter_dims}-ld_{latent_dims}-cs_{context_seed}-mom_0.0-ts_{train_size}-mi_{median_init}/last.ckpt'),
    context_seed=[1, 2, 3],
    train_size=[500, 1000, 2000],
    median_init=[True, False],
    constraints=['architecture', 'implicit_bias'],
    no_radius=True
)

def main(args):
    argparse_array.call_script('scripts/train_convex.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)

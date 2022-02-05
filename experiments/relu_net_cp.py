import argparse
import sys
import os

sys.path.append('')

from functions.array_training import ArgparseArray

model_path = [f'data/relu_net/{folder}/last.ckpt' for folder in os.listdir('data/relu_net') if int(folder.split('-')[0].split('_')[1]) <= 100] # 72

argparse_array = ArgparseArray(
    model_path=model_path,
    train_size=(lambda array_id, model_path, **kwargs: model_path.split('/')[2].split('-')[3].split('_')[1]),
    data_seed=1,
    objective=['gates_only', 'hidden_layer', 'learned_contexts', 'random_contexts'],
    save_dir=(lambda array_id, model_path, objective, **kwargs: os.path.join(*model_path.split('/')[:3], objective))
)

def main(args):
    argparse_array.call_script('scripts/train_convex_relu.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)

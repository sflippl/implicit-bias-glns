import sys
sys.path.append('')

from functions.train import get_parser, train
from functions.modules import GLNModule

if __name__=='__main__':
    parser = get_parser(GLNModule)
    args = parser.parse_args()
    train(args, GLNModule)

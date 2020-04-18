import numpy as np
from model import *
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dataloader import *
import argparse
import logging
from train import train

if __name__ == '__main__':
    """
        Example usage train:
        python3 main.py -data_dir data -hidden_dim 256  -embed_dim 100 -batch_size 32\
            -epochs 50 -lr 0.001 -clip_value 0.0 -w_decay 0.0 -max_decoding_length 300 \
            -is_stochastic 1 -train 1 -models_dir models -logs_dir logs -model_path best.pth \
            -num_workers 64

        Example usage inference:
        python3 main.py -data_dir data -hidden_dim 256  -embed_dim 100 -batch_size 32\
            -epochs 50 -lr 0.001 -clip_value 0.0 -w_decay 0.0 -max_decoding_length 300 \
            -is_stochastic 1 -train 0 -models_dir models -logs_dir logs -model_path best.pth \
            -num_workers 64
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", dest="data_dir", type=str, default="")
    parser.add_argument("-hidden_dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("-embed_dim", dest="embed_dim", type=int, default=40)
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("-epochs", dest="epochs", type=int, default=50)
    parser.add_argument("-lr", dest="lr", type=float, default=0.001)
    parser.add_argument("-clip_value", dest="clip_value", type=float, default=0)
    parser.add_argument("-w_decay", dest="w_decay", type=float, default=0.00001)
    parser.add_argument("-max_decoding_length", dest="max_decoding_length", type=int, default=300)
    parser.add_argument("-is_stochastic", dest="is_stochastic", type=int, default=1)
    parser.add_argument("-train", dest="train", type=int, default=1)
    parser.add_argument("-models_dir", dest="model_dir", type=str, default='models')
    parser.add_argument("-logs_dir", dest="logs_dir", type=str, default='logs')
    parser.add_argument("-model_path", dest="model_path", type=str, default='best.pth')
    parser.add_argument("-num_workers", dest="num_workers", type=int, default=64)
    args = parser.parse_args()

    for s in [args.model_dir, args.logs_dir]:
        create_folder(args.model_dir) 
    cuda = torch.cuda.is_available()
    create_logging(args.logs_dir, filemode = 'w')   
    logging.info('logging started for model = {}'.format(args.model_file_name))

    if args.train == 1:
        print('Training started .......')
        model = train(args, logging, cuda)
    elif args.train == 0:
        print('Inference started .......')
        # TODO Inference pipeline and prediction saving





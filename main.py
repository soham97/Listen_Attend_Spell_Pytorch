import numpy as np
from model import *
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dataloader import *
import argparse
import logging
from train import train, continue_train, create_submission

if __name__ == '__main__':
    """
        Example usage for training:
        python3 main.py -data_dir data -hidden_dim 256  -embed_dim 40 -batch_size 32\
            -epochs 100 -lr 0.001 -clip_value 0.0 -w_decay 0.0 -max_decoding_length 300 \
            -is_stochastic 1 -train 1 -models_dir models -logs_dir logs -model_path best.pth \
            -num_workers 64 -tf 0.3 -locked_dropout 0.3

        Example usage to generate submission:
        python3 main.py -data_dir data -hidden_dim 256  -embed_dim 40 -batch_size 32\
            -epochs 100 -lr 0.001 -clip_value 0.0 -w_decay 0.0 -max_decoding_length 300 \
            -is_stochastic 1 -train 0 -models_dir models -logs_dir logs -model_path best.pth \
            -num_workers 64 -tf 0.3 -locked_dropout 0.3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", type=str, default="")
    parser.add_argument("-hidden_dim", "--hidden_dim", type=int, default=256)
    parser.add_argument("-embed_dim", "--embed_dim", type=int, default=40)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=32)
    parser.add_argument("-epochs", "--epochs", type=int, default=50)
    parser.add_argument("-lr", "--lr", type=float, default=0.001)
    parser.add_argument("-clip_value", "--clip_value", type=float, default=0)
    parser.add_argument("-w_decay", "--w_decay", type=float, default=0.00001)
    parser.add_argument("-max_decoding_length", "--max_decoding_length", type=int, default=300)
    parser.add_argument("-is_stochastic", "--is_stochastic", type=int, default=1)
    parser.add_argument("-train", "--train", type=int, default=1)
    parser.add_argument("-models_dir", "--model_dir", type=str, default='models')
    parser.add_argument("-logs_dir", "--logs_dir", type=str, default='logs')
    parser.add_argument("-model_path", "--model_path", type=str, default='best.pth')
    parser.add_argument("-num_workers", "--num_workers", type=int, default=64)
    parser.add_argument("-tf", "--tf", type=float, default=0.1)
    parser.add_argument("-locked_dropout", "--locked_dropout", type=float, default=0.3)
    parser.add_argument("-pretrained_model_path", "-pretrained_model_path", type=str, default='best.pth')
    args = parser.parse_args()

    for s in [args.model_dir, args.logs_dir]:
        create_folder(args.model_dir) 
    cuda = torch.cuda.is_available()

    if args.train == 0:
        print('Creating submission for test data .......')
        create_submission(args, cuda)
    elif args.train == 1:
        model = train(args, cuda)
    elif args.train == 2:
        print('Loading pretrained model and training ......')
        continue_train(args, cuda)





import numpy as np
from model import LAS
from dataloader import WSJ_DataLoader
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from utils import *
from train_helper import train_batch, eval_batch
import logging

def train(args, cuda):
    create_logging(args.logs_dir, filemode = 'w')   
    logging.info('logging started for model = {}'.format(args.model_path))
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len, cuda)
    if cuda:
        model = model.cuda()

    model_path = os.path.join(args.model_dir, args.model_path)
    criterian = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)
    print('Data loading compelete .......')

    print('Training started .......')
    best_val_dist = np.inf
    for epoch in range(args.epochs):
        model.train()
        train_dist, train_loss = train_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len)
        model.eval()
        val_dist, val_loss = eval_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len)
        
        if val_dist < best_val_dist:
            best_val_dist = val_dist
            save_model(epoch, model, optimizer, scheduler, model_path)
            print('Model saved!')
        
        if epoch%14 == 0:
            save_model(epoch, model, optimizer, scheduler, os.path.join(args.model_dir, f'epoch_{str(epoch)}.pth'))

        logging.info('epoch: {}, train_loss: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, train_dist, val_loss, val_dist))
    
    return model

def continue_train(args, cuda):
    create_logging(args.logs_dir, filemode = 'w')   
    logging.info('logging started for model = {}'.format(args.model_path))
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len, cuda)
    load_model_path = os.path.join(args.model_dir, args.pretrained_model_path)
    checkpoint = load_model(load_model_path,cuda)
    model.load_state_dict(checkpoint['model'])
    if cuda:
        model = model.cuda()
    model_path = os.path.join(args.model_dir, args.model_path)
    criterian = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)
    print('Data loading compelete .......')

    print('Training started .......')
    best_val_dist = np.inf
    for epoch in range(args.epochs):
        model.train()
        train_dist, train_loss = train_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len)
        model.eval()
        val_dist, val_loss = eval_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len)
        
        if val_dist < best_val_dist:
            best_val_dist = val_dist
            save_model(epoch, model, optimizer, scheduler, model_path)
            print('Model saved!')
        
        if epoch%14 == 0:
            save_model(epoch, model, optimizer, scheduler, os.path.join(args.model_dir, f'epoch_{str(epoch)}.pth'))

        logging.info('epoch: {}, train_loss: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, train_dist, val_loss, val_dist))
    
    return model

def create_submission(args, cuda):
    model_path = os.path.join(args.model_dir, args.model_path)
    DataLoaderContainer = WSJ_DataLoader(args, cuda)
    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len

    # load model
    model = LAS(args, vocab_len, max_input_len, cuda)
    checkpoint = load_model(model_path, cuda)
    model.load_state_dict(checkpoint['model'])
    if cuda:
        model = model.cuda()

    model.eval()
    create_folder('results')
    j = 0
    file = open('results/submission.csv', 'w')
    file.write("Id,Predicted\n")
    for (x, x_len, _, _, _) in tqdm(DataLoaderContainer.test_dataloader):
        if cuda:
            x = x.cuda()
        output = model(x, x_len)
        char_pred = [DataLoaderContainer.index_to_char[idx] for idx in output]
        char_pred = ''.join(char_pred)
        file.write("{},{}\n".format(j, char_pred))
        j += 1
    file.close()
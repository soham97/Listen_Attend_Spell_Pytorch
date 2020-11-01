import numpy as np
from model import LAS
import os
import torch.nn as nn
import torch
from Levenshtein import distance as levenshtein_distance
import logging
from utils import *

def get_distance(DataLoaderContainer, y_pred, y):
    y_greedy = torch.max(y_pred, dim=1)[1]
    y_pred_char = ''.join([DataLoaderContainer.index_to_char[idx] for idx in y_greedy.detach().cpu()])
    y_true_char = ''.join([DataLoaderContainer.index_to_char[idx] for idx in y.detach().cpu()])
    return levenshtein_distance(y_pred_char, y_true_char)

def get_dataloader(DataLoaderContainer, name):
    if name == 'test':
        return DataLoaderContainer.test_dataloader
    elif name == 'val':
        return DataLoaderContainer.val_dataloader
    elif name == 'train':
        return DataLoaderContainer.train_dataloader

def train_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len):
    train_loss_samples = []
    train_dist = []
    for batch,(x, x_len, y, y_len, y_mask) in enumerate(DataLoaderContainer.train_dataloader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
            y_mask = y_mask.cuda()
        optimizer.zero_grad()
        y_pred = model(x, x_len, y, y_len, tf = args.tf)
        # compute loss now: can also used masked_select here,
        # but instead going with nonzero(), just a random choice
        y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
        y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
            dim = 0, index = y_mask)
        y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)

        loss = criterian(y_pred, y) # no batch_size so using sum, then / by bs
        loss = loss/args.batch_size
        loss.backward()
        if batch % 100 == 0:
            print(f'Batch: {str(batch)}, loss: {str(loss.cpu().item())}')
        if args.clip_value > 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
        optimizer.step()
        train_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
        train_loss_samples.append(loss.data.cpu().numpy())
    
    return np.mean(train_dist), np.mean(train_loss_samples)

def eval_batch(model, DataLoaderContainer, optimizer, criterian, scheduler, args, cuda, vocab_len):
    val_loss_samples = []
    val_dist = []
    for batch, (x, x_len, y, y_len, y_mask) in enumerate(DataLoaderContainer.val_dataloader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
            y_mask = y_mask.cuda()

        y_pred = model(x, x_len, y, y_len, tf = args.tf)
        y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
        y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
            dim = 0, index = y_mask)
        y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)
        loss = criterian(y_pred, y)
        loss = loss/args.batch_size
        val_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
        val_loss_samples.append(loss.data.cpu().numpy())
    
    return np.mean(val_dist), np.mean(val_loss_samples)







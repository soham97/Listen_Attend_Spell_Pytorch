import numpy as np
from model import LAS
from dataloader import WSJ_DataLoader
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from utils import *
from Levenshtein import distance as levenshtein_distance
import logging

def get_distance(DataLoaderContainer, y_pred, y):
    y_greedy = torch.max(y_pred, dim=1)[1]
    y_pred_char = ''.join([DataLoaderContainer.index_to_char[idx] for idx in y_greedy.detach().cpu()])
    y_true_char = ''.join([DataLoaderContainer.index_to_char[idx] for idx in y.detach().cpu()])
    return levenshtein_distance(y_pred_char, y_true_char)

def get_tf(args, epoch, model_path):
    if epoch >= 0 and epoch < 15:
        new_tf =  args.tf #this is 0.3
    if epoch >= 15 and epoch < 30: 
         new_tf =  args.tf + 0.1 #this is 0.4
    # if epoch >= 30 and epoch < 45:
    #     new_tf =  args.tf + 0.2 #this is 0.5
    if epoch >= 30:
        new_tf = 0.4
    return new_tf

def train(args, cuda):
    create_logging(args.logs_dir, filemode = 'w')   
    logging.info('logging started for model = {}'.format(args.model_path))
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len, cuda)
    if cuda:
        model = model.cuda()
    model_path = os.path.join(args.model_dir, args.model_path.split('.')[0])
    criterian = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)
    print('Data loading compelete .......')

    print('Training started .......')
    best_val_loss_03 = np.inf
    best_val_loss_04 = np.inf
    tf = args.tf
    for epoch in range(args.epochs):
        train_loss_samples = []
        val_loss_samples = []
        train_dist = []
        val_dist = []
        model.train()
        tf = get_tf(args, epoch, model_path) # get tf value by epoch
        for (x, x_len, y, y_len, y_mask) in DataLoaderContainer.train_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            optimizer.zero_grad()
            y_pred = model(x, x_len, y, y_len, tf = tf)
            # compute loss now: can also used masked_select here,
            # but instead going with nonzero(), just a random choice
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)

            loss = criterian(y_pred, y) # no batch_size so using sum, then / by bs
            loss = loss/args.batch_size
            loss.backward()
            if args.clip_value > 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            train_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
            train_loss_samples.append(loss.data.cpu().numpy())
        
        model.eval()
        for (x, x_len, y, y_len, y_mask) in DataLoaderContainer.val_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            y_pred = model(x, x_len, y, y_len, tf = tf)
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)
            loss = criterian(y_pred, y)
            loss = loss/args.batch_size
            val_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
            val_loss_samples.append(loss.data.cpu().numpy())
        
        train_loss = np.mean(train_loss_samples)
        val_loss = np.mean(val_loss_samples)
        train_dist = np.mean(train_dist)
        val_dist = np.mean(val_dist)
        # scheduler.step(val_dist)

        if tf == 0.3:
            if val_loss < best_val_loss_03:
                best_val_loss_03 = val_loss
                save_model(epoch, model, optimizer, scheduler, model_path + '_' +str(tf)+'_.pth')
        elif tf ==0.4:
            if val_loss < best_val_loss_04:
                best_val_loss_04 = val_loss
                save_model(epoch, model, optimizer, scheduler, model_path + '_' +str(tf)+'_.pth')
        
        if epoch%14 == 0:
            save_model(epoch, model, optimizer, scheduler, os.path.join(args.model_dir, f'epoch_{str(epoch)}.pth'))

        # logging.info('epoch: {}, train_loss: {:.3f}, train_perplexity: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_perplexity: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, np.exp(train_loss), train_dist, val_loss, np.exp(val_loss), val_dist))
        logging.info('epoch: {}, train_loss: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, train_dist, val_loss, val_dist))
    
    return model

def continue_train(args, cuda):
    create_logging(args.logs_dir, filemode = 'w')   
    logging.info('logging started for model = {}'.format(args.model_path))
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len, cuda)
    load_model_path = os.path.join(args.model_dir, 'best.pth')
    checkpoint = load_model(load_model_path,cuda)
    model.load_state_dict(checkpoint['model'])
    if cuda:
        model = model.cuda()
    model_path = os.path.join(args.model_dir, args.model_path)
    criterian = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)
    print('Data loading compelete .......')

    print('Training started .......')
    best_val_loss = np.inf
    tf = args.tf
    for epoch in range(args.epochs):
        train_loss_samples = []
        val_loss_samples = []
        train_dist = []
        val_dist = []
        model.train()
        tf = get_tf(args, epoch) # get tf value by epoch
        for batch, (x, x_len, y, y_len, y_mask) in enumerate(DataLoaderContainer.train_dataloader):
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            optimizer.zero_grad()
            y_pred = model(x, x_len, y, y_len, tf = tf)
            # compute loss now: can also used masked_select here,
            # but instead going with nonzero(), just a random choice
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)

            loss = criterian(y_pred, y) # no batch_size so using sum, then / by bs
            loss = loss/args.batch_size

            if batch % 100 == 0:
                print(f'Batch: {str(batch)}, loss: {str(loss)}')
            # changes not getting registered
            loss.backward()
            if args.clip_value > 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            train_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
            train_loss_samples.append(loss.data.cpu().numpy())
        
        model.eval()
        for (x, x_len, y, y_len, y_mask) in DataLoaderContainer.val_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            y_pred = model(x, x_len, y, y_len, tf = tf)
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)
            loss = criterian(y_pred, y)
            loss = loss/args.batch_size
            val_dist.append(get_distance(DataLoaderContainer, y_pred, y)/args.batch_size)
            val_loss_samples.append(loss.data.cpu().numpy())
        
        train_loss = np.mean(train_loss_samples)
        val_loss = np.mean(val_loss_samples)
        train_dist = np.mean(train_dist)
        val_dist = np.mean(val_dist)
        # scheduler.step(val_dist)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(epoch, model, optimizer, scheduler, model_path)
        
        if epoch%14 == 0:
            save_model(epoch, model, optimizer, scheduler, os.path.join(args.model_dir, f'epoch_{str(epoch)}.pth'))

        # logging.info('epoch: {}, train_loss: {:.3f}, train_perplexity: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_perplexity: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, np.exp(train_loss), train_dist, val_loss, np.exp(val_loss), val_dist))
        logging.info('epoch: {}, train_loss: {:.3f}, train_dist: {:.3f}, val_loss: {:.3f}, val_dist: {:.3f}'.format(epoch, train_loss, train_dist, val_loss, val_dist))
    
    return model


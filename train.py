import numpy as np
from model import LAS
from dataloader import WSJ_DataLoader
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from utils import *

def train(args, logging, cuda):
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len, cuda)
    if cuda:
        model = model.cuda()
    model_path = os.path.join(args.model_dir, args.model_path)
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, verbose = True)
    print('Data loading compelete .......')

    print('Training started .......')
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss_samples = []
        val_loss_samples = []
        model.train()
        for (x, x_len, y, y_len, y_mask) in tqdm(DataLoaderContainer.val_dataloader):
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            optimizer.zero_grad()
            y_pred = model(x, x_len, y, y_len)
            # compute loss now: can also used masked_select here,
            # but instead going with nonzero(), just a random choice
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)

            loss = criterian(y_pred, y)
            loss.backward()
            optimizer.step()

            if args.clip_value > 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_value)
            train_loss_samples.append(loss.data.cpu().numpy())
        
        model.eval()
        for (x, x_len, y, y_len, y_mask) in tqdm(DataLoaderContainer.val_dataloader):
            if cuda:
                x = x.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            y_pred = model(x, x_len, y, y_len)
            y_mask = y_mask[:, 1:].contiguous().view(-1).nonzero().squeeze()
            y_pred = torch.index_select(y_pred.contiguous().view(-1, vocab_len),\
                dim = 0, index = y_mask)
            y = torch.index_select(y[:, 1:].contiguous().view(-1), dim=0, index=y_mask)
            loss = criterian(y_pred, y)
            val_loss_samples.append(loss.data.cpu().numpy())
        
        train_loss = np.mean(train_loss_samples)
        val_loss = np.mean(val_loss_samples)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(epoch, model, optimizer, scheduler, model_path)

        logging.info('epoch: {}, train_loss: {:.3f}, train_perplexity: {:.3f}, \
                    val_loss: {:.3f}, val_perplexity: {:.3f}'.format(epoch, train_loss, np.exp(train_loss), val_loss, np.exp(val_loss)))
    
    
    return model


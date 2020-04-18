import numpy as np
from model import LAS
from dataloader import WSJ_DataLoader
from tqdm import tqdm

def train(args, logging, cuda):
    DataLoaderContainer = WSJ_DataLoader(args, cuda)

    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len
    model = LAS(args, vocab_len, max_input_len)
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, verbose = True)

    if cuda:
        model = model.cuda()

    best_val_loss = np.inf
    for epoch in tqdm(range(args.epochs)):
        train_loss_samples = []
        val_loss_samples = []
        model.train()
        for (x, x_len, y, y_len, y_mask) in DataLoaderContainer.train_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            y_pred = model(x, x_len, y, y_len, y_mask)
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
        for (x, x_len, y, y_len, y_mask) in DataLoaderContainer.val_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x, x_len, y, y_len, y_mask)
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
            save_model(epoch, model, optimizer, scheduler, args.model_path)

        logging.info('epoch: {}, train_loss: {:.3f}, val_loss: {:.3f}'.format(epoch, train_loss, val_loss))
    
    
    return model


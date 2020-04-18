import numpy as np
from utils import *
from model import LAS
from dataloader import WSJ_DataLoader
import os
import torch.nn as nn
import torch
from tqdm import tqdm

def eval(args, logging, cuda):
    model_path = os.path.join(args.model_dir, args.model_path)
    DataLoaderContainer = WSJ_DataLoader(args, cuda)
    vocab_len = len(DataLoaderContainer.index_to_char)
    max_input_len = DataLoaderContainer.max_input_len

    # load model
    model = LAS(args, vocab_len, max_input_len, cuda)
    checkpoint = load_model(model_path, cuda)
    model.load_state_dict(checkpoint['model'])
    # start decoding
    for name in ['val', 'test', 'train']:
        decode(model, DataLoaderContainer, name, cuda)
        print('{} decoding complete!'.format(name))

def get_dataloader(DataLoaderContainer, name):
    if name == 'test':
        return DataLoaderContainer.test_dataloader
    elif name == 'val':
        return DataLoaderContainer.val_dataloader
    elif name == 'train':
        return DataLoaderContainer.train_dataloader

def decode(model, DataLoaderContainer, name, cuda):
        """
        :param my_net:
        :return: Writes the decoded result of test set in submission.csv
        """
        model.eval()
        dataloader = get_dataloader(DataLoaderContainer, name)
        create_folder('results')
        i = 0
        file = open('results/{}_submission.csv'.format(name), 'w')
        file.write("Id,Predicted\n")
        for (x, x_len, _, _, _) in tqdm(dataloader):
            if cuda:
                x = x.cuda()
            output, raw_preds = model(x, x_len)
            output = get_best_out(output, raw_preds)
            pred = [DataLoaderContainer.index_to_char[idx] for idx in output]
            pred = ''.join(pred)
            file.write("{},{}\n".format(i, pred))
            i += 1
        file.close()

def get_best_out(output, raw_preds):
        criterian = torch.nn.CrossEntropyLoss()
        best_loss = np.inf
        best_out = None
        for i, each in enumerate(output):
            loss = criterian(raw_preds[i], torch.from_numpy(np.array(each)).long()).data.cpu().numpy()
            if loss < best_loss:
                best_loss = loss
                best_out = each[:-1]
        return best_out
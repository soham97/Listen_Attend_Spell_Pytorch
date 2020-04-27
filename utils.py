import torch
import torch.nn as nn
import os
import logging
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self, dropout = 0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

def save_model(epoch, model, optimizer, scheduler, path_model = 'best.pth'):
    checkpoint = { 
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, path_model)

def load_model(path_model, cuda):
    if cuda:
        return torch.load(path_model)
    else:
        return torch.load(path_model, map_location=torch.device('cpu'))

def to_variable(tensor, requires_grad=False):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.normal_(module.bias.data)

    elif isinstance(module, nn.RNNBase) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.normal_(param.data)

def create_folder(fd):
    if not os.path.exists(fd):
        # creates problems when multiple scripts creating folders
        try:
            os.makedirs(fd)
        except:
            print('Folder already exits')
    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
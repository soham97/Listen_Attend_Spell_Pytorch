import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import random
from utils import *

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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
    
    def forward(self, query, keys, values, x_len, cuda):
        """
        :param query: (batch_size, hidden_size), decoder state of a single timestep
        :param keys: (batch_size, max_len, hidden_size)
        :param lengths: (batch_size,), lengths of source sequences
        :returns: attended context: (batch_size, hidden_size)
        """
        # using a linear layer to augment query
        query = self.linear(query)
        # energy (batch_size, 1, max_len) = (batch_size,1,hidden_size) x (batch_size, hidden_size, max_len)
        # in LAS: max_len = seq_len/8
        energy = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))
        # Create mask
        # padded_len: (1, max_len) --> expand --> (batch_size, max_len)
        # actual_len: is (batch_size, 1)
        # mask is (batch_size, max_len) is False when we have padded output
        padded_len = torch.arange(x_len[0]).unsqueeze(0).expand(len(x_len), -1)
        actual_len = x_len.unsqueeze(1).float()
        mask = padded_len < actual_len
        mask = mask.unsqueeze(1).type(torch.FloatTensor)
        mask = mask.cuda() if cuda else mask
        # mask: (batch_size, 1, max_len)
        attention = F.softmax(energy, dim=2)
        attention = attention * mask
        attention = attention / attention.sum(2).unsqueeze(2)
        context = torch.bmm(attention, values.permute(1, 0, 2)).squeeze(1)
        # context is (batch_size, hidden_size)
        return context

class CustomLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__(input_size, hidden_size)
        # This is done to make sure, the initial states are learnable params
        self.c0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.h0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, h, hx, cx):
            return super(CustomLSTMCell, self).forward(h, (hx, cx))

class Encoder(nn.Module):
    """
    pBILSTM from paper: https://arxiv.org/pdf/1508.01211.pdf
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        """
        This uses following args parameters:
        args.hidden_size
        args.embed_dim
        args.num_layers

        Donot need the sequential nature of nn.Sequential, hence
        using nn.Module
        """
        hidden_size = args.hidden_dim #32
        embedding_dim = args.embed_dim #40
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True),
            nn.LSTM(input_size=2*2*hidden_size, hidden_size=hidden_size, bidirectional=True),
            nn.LSTM(input_size=2*2*hidden_size, hidden_size=hidden_size, bidirectional=True),
            nn.LSTM(input_size=2*2*hidden_size, hidden_size=hidden_size, bidirectional=True)])
        
        self.keys = nn.Linear(in_features = 2*hidden_size, out_features = 2*hidden_size)
        self.values = nn.Linear(in_features = 2*hidden_size, out_features = 2*hidden_size)

    
    def forward(self, x, x_len):
        """
        x shape: (seq_len, batch_size, mel_bins = 40)
        x_len shape: (batch_size, )
        """
        h = x
        for i, lstm in enumerate(self.lstm_layers):
            if i > 0:
                # After first lstm layer, pBiLSTM
                seq_len = h.size(0)
                if seq_len % 2 == 0:
                    h = h.permute(1,0,2).contiguous()
                    # h = (h_f + h_b) / 2
                    # h = h.view(h.size(0), h.size(1) // 2, 2, h.size(2)).sum(2)/ 2
                    h = h.view(h.size(0), h.size(1) // 2, h.size(2)*2)
                    h = h.permute(1,0,2).contiguous()
                    x_len /= 2
                else:
                    print('Seq_len not divisible by 2')
                    exit()
            # First BiLSTM
            packed_h = pack_padded_sequence(h, x_len.cpu().numpy())
            h, _ = lstm(packed_h)
            h, _ = pad_packed_sequence(h)      # seq_len, bs, (2 * hidden_dim)
            # Summing forward and backward representation ie h = (h_f + h_b) / 2
            # h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2 
        
        # both keys and values are of shape (bs, seq_len/8, 256)
        keys = self.keys(h)
        values = self.values(h)
        return keys, values

class Decoder(nn.Module):
    def __init__(self, args, output_size, cuda):
        super(Decoder, self).__init__()
        self.hidden_size = 2*args.hidden_dim 
        self.embedding_dim = args.embed_dim
        self.is_stochastic = args.is_stochastic
        self.max_decoding_length = args.max_decoding_length
        self.cuda = cuda

        # Embedding layer
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=self.hidden_size)
        # 3 LSTM cells
        self.lstm_cells = nn.ModuleList([
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)
        ])
        # 3 Locked dropout, one for each LSTM cell
        self.locked_dropouts = nn.ModuleList([
            LockedDropout(dropout = args.locked_dropout),
            LockedDropout(dropout = args.locked_dropout),
            LockedDropout(dropout = args.locked_dropout)
        ])
        # Attention
        self.attention = Attention(self.hidden_size)
        # For character projection to vocab_size
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.activation =  nn.Hardtanh(inplace = True)
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
        # Tying weights of last layer and embedding layer for regularisation
        self.projection_layer2.weight = self.embed.weight

    def forward(self, keys, values, label, y_len, x_len, tf):
        # convert label to d dimensional vector
        hidden_states = []
        embed = self.embed(label) # bs, label_len, 256
        output = None

        # Initial context
        # here the h0 from lstmcell2 is is extended to (batch_size, embed_size = hidden_size)
        query = self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous()
        context = self.attention(query, keys, values, x_len, self.cuda)
        # list hidden_states contains [(h_x, c_x) at t = 0, (h_x, c_x) at t = 1, ......]
        for i in range(y_len.max() - 1):
            teacher_forcing = False if random.random() > tf else True
            if teacher_forcing and i != 0:
                _, index = torch.max(y_char_vocab, dim = 1)
                h = self.embed(index)
            else:
                h = embed[:, i, :] # (bs, 256) --> considering particular indexed embedding
            h = torch.cat((h, context), dim = 1)
            # Passing data through 3 LSTM cells
            for j,lstm in enumerate(self.lstm_cells):
                h = self.locked_dropouts[j](h.unsqueeze(1)).squeeze(1)
                if i == 0:
                    h_x_0, c_x_0 = lstm(h, lstm.h0.expand(embed.size(0), -1).contiguous(), \
                                        lstm.c0.expand(embed.size(0), -1).contiguous())
                    hidden_states.append((h_x_0, c_x_0))
                else:
                    h_x_0, c_x_0 = hidden_states[j]
                    hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]
            
            context = self.attention(h, keys, values, x_len, self.cuda)
            y_char = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 3 lstm cells. Passing it through the projection layers
            y_char_vocab = self.projection_layer2(self.activation(self.projection_layer1(y_char)))
            # Accumulating the output at each timestep
            # Checking first if output is not None 
            output = y_char_vocab.unsqueeze(1) if output is None else torch.cat((output, y_char_vocab.unsqueeze(1)), dim=1)

        return output

    def testdecode(self, keys, values, x_len, args):
        """
        returns Best decoded sequence test
        """
        # here the batch should be 1 for decoding
        hidden_states = []
        output = []
        # Initial context
        query = self.lstm_cells[2].h0  # bs, 256 --> This is the query
        context = self.attention(query, keys, values, x_len, self.cuda)
        
        h = self.embed(to_variable(torch.zeros(args.batch_size).long()))  # Start token provided for generating the sentence
        for i in range(self.max_decoding_length):
            h = torch.cat((h, context), dim=1)
            for j, lstm in enumerate(self.lstm_cells):
                if i == 0:
                    h_x_0, c_x_0 = lstm(h, lstm.h0, lstm.c0)  # bs * 512
                    hidden_states.append((h_x_0, c_x_0))
                else:
                    h_x_0, c_x_0 = hidden_states[j]
                    hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]

            context = self.attention(h, keys, values, x_len, self.cuda)
            y_char = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 3 lstm cells. Passing it through the projection layers
            y_char_vocab = self.projection_layer2(self.activation(self.projection_layer1(y_char)))

            y_char_max = torch.max(y_char_vocab, dim=1)[1]
            output.append(y_char_max.data.cpu().numpy()[0])

            if y_char_max.data.cpu().numpy() == 1: # <eos> encountered
                break

            # Primer for next character generation
            h = self.embed(y_char_max)
        return output

class LAS(nn.Module):
    def __init__(self, args, output_size, max_seq_len, cuda):
        super(LAS, self).__init__()
        self.hidden_size = args.hidden_dim
        self.embedding_dim = args.embed_dim
        self.max_seq_len = max_seq_len
        self.output_size = output_size
        self.encoder = Encoder(args)
        self.decoder = Decoder(args, output_size, cuda)
        self.args = args
        self.apply(init_weights)
        print('Layers initialised')

    def forward(self, input, x_len, label=None, y_len=None, tf = None):
        input = input.permute(1, 0, 2)
        keys, values = self.encoder(input, x_len)
        if label is None:
            # During decoding of test data
            return self.decoder.testdecode(keys, values, x_len, self.args)
        else:
            # During training
            return self.decoder(keys, values, label, y_len, x_len, tf)

if __name__ == "__main__":
    print('Testing starts here: ')
    encoder = Encoder([])
    # Here the input is (64, 8, 10) = (seq_len, bs, dim) for testing
    # The lens are are tensor of 8 values, each length is constant of value 64
    # In a way, the max value here in the batch is 64
    # For the pBILSTM to work: the max value needs to be divisible by 2 per iteration
    # OR in total the max value needs to be divisible by 8
    batch_size = 8
    embed_dim = 40
    seq_len_padded = 64
    cuda = False
    keys, values = encoder(torch.rand(seq_len_padded,batch_size,embed_dim), 64*torch.ones(batch_size))
    print('Encoder check complete')

    decoder = Decoder([],32)
    outputs = decoder(keys, values, torch.ones(batch_size,seq_len_padded).long(), \
        8*torch.ones(batch_size).int(), 8*torch.ones(batch_size), cuda)
    print('Decoder train check complete')

    outputs, raw_preds = decoder.decode(torch.rand(8,1,32), torch.rand(8,1,32))
    print('Decoder test check complete')

    # input here to the whole model is batch as first dim
    # However, the input to encoder is batch as second dim
    las = LAS([], 32, 64, cuda)
    las(torch.rand(batch_size, seq_len_padded,embed_dim), 64*torch.ones(batch_size), \
        torch.ones(batch_size,seq_len_padded).long(), \
        8*torch.ones(batch_size).int())
    print('Total model check complete')
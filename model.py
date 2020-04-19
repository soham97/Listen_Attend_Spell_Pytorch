import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import random
from utils import *

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
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)])
        
        self.keys = nn.Linear(in_features = hidden_size, out_features = hidden_size)
        self.values = nn.Linear(in_features = hidden_size, out_features = hidden_size)

    
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
                    h = h.view(h.size(0), h.size(1) // 2, 2, h.size(2)).sum(2)/ 2
                    h = h.permute(1,0,2).contiguous()
                    x_len /= 2
                else:
                    print('Seq_len not divisible by 2')
                    exit()
            # First BiLSTM
            packed_h = pack_padded_sequence(h, x_len.cpu().numpy())
            h, _ = lstm(packed_h)
            h, _ = pad_packed_sequence(h)      # seq_len * bs * (2 * hidden_dim)
            # Summing forward and backward representation ie h = (h_f + h_b) / 2
            h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2       
        
        # both keys and values are of shape (bs, seq_len/8, 256)
        keys = self.keys(h)
        values = self.values(h)
        return keys, values

class CustomLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__(input_size, hidden_size)
        # This is done to make sure, the initial states are learnable params
        self.c0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.h0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, h, hx, cx):
            return super(CustomLSTMCell, self).forward(h, (hx, cx))

class Decoder(nn.Module):
    def __init__(self, args, output_size, cuda):
        super(Decoder, self).__init__()
        self.hidden_size = args.hidden_dim #32
        self.embedding_dim = args.embed_dim #100
        self.is_stochastic = args.is_stochastic #True
        self.max_decoding_length = args.max_decoding_length #75
        self.cuda = cuda

        # Embedding layer
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=self.hidden_size)
        # LSTM cells
        self.lstm_cells = nn.ModuleList([
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            CustomLSTMCell(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)
        ])
        # Attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        
        # For character projection
        self.pl1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.activation = nn.LeakyReLU()
        self.pl2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
        self.logsm = nn.LogSoftmax(dim=1)
        # Tying weights of last layer and embedding layer
        # self.pl2.weight = self.embed.weight

    def forward(self, keys, values, label, label_len, input_len, tf):
        # convert label to d dimensional vector
        embed = self.embed(label) # bs, label_len, 256
        output = None
        hidden_states = []

        # Initial context
        # here the h0 from lstmcell2 is is extended to (batch_size, hidden_size)
        query = self.linear(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous())  # bs, 256, This is the query
        # so (bs,1,hs) x (bs, hs, seq_len/8)
        energy = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs, 1, seq_len/8

        # Create mask
        # here a is (1, max_len) --> expand --> (batch_size, max_len)
        # here b is (batch_size, 1)
        # mask is (batch_size, max_len) is False we have padded output
        a = torch.arange(input_len[0]).unsqueeze(0).expand(len(input_len), -1)
        b = input_len.unsqueeze(1).float()
        mask = a < b

        if self.cuda: # here it should be args.cuda
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor)).cuda()
        else:
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor))
        # mask is now: (batch_size, 1, max_len)
        
        # here attention is over all seq_len which is last dim
        # removing the padded using mask, where its False, all elements get set to 0
        attn = F.softmax(energy, dim=2)
        attn = attn * mask
        attn = attn / attn.sum(2).unsqueeze(2)
        context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)

        # list hidden_states contains [(h_x, c_x) at t = 0, (h_x, c_x) at t = 1, ......]
        for i in range(label_len.max() - 1):
            teacher_forcing = False if random.random() > tf else True
            if teacher_forcing and i != 0:
                value, index = torch.max(h, dim = 1)
                h = self.embed(index)
            else:
                h = embed[:, i, :] # (bs, 256) --> considering particular indexed embedding
            h = torch.cat((h, context), dim = 1)
            for j,lstm in enumerate(self.lstm_cells):
                if i == 0:
                    h_x_0, c_x_0 = lstm(h, lstm.h0.expand(embed.size(0), -1).contiguous(), \
                                        lstm.c0.expand(embed.size(0), -1).contiguous())
                    hidden_states.append((h_x_0, c_x_0))
                else:
                    h_x_0, c_x_0 = hidden_states[j]
                    hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]
            
            query = self.linear(h)    # bs , 2048, This is the query
            energy = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))   # bs, 1, seq_len/8
            attn = F.softmax(energy, dim=2)
            attn = attn * mask
            attn = attn / attn.sum(2).unsqueeze(2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)       # bs, 256
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 3 lstm cells. Passing it through the projection layers
            h = self.activation(self.pl1(h))
            h = self.logsm(self.pl2(h))
            # Accumulating the output at each timestep
            if output is None:
                output = h.unsqueeze(1)
            else:
                output = torch.cat((output, h.unsqueeze(1)), dim=1)

        return output    # bs , max_label_seq_len, 33

    def sample_gumbel(self, shape, eps=1e-10, out=None):
        """
        Sample from Gumbel(0, 1)
        Inspired from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        """
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        return - torch.log(eps - torch.log(U + eps))

    def decode(self, keys, values, args):
        """
        Input: keys, values
        return: Best decoded sequence
        """
        # here the batch should be 1 for decoding
        output = []
        raw_preds = []

        for _ in range(100):
            hidden_states = []
            raw_pred = None
            raw_out = []
            # Initial context
            query = self.linear(self.lstm_cells[2].h0)  # bs, 256 --> This is the query
            energy = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs, 1, seq_len/8
            attn = F.softmax(energy, dim=2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)
            
            h = self.embed(to_variable(torch.zeros(args.batch_size).long()))  # Start token provided for generating the sentence
            for i in range(self.max_decoding_length):
                h = torch.cat((h, context), dim=1)
                for j, lstm in enumerate(self.lstm_cells):
                    if i == 0:
                        h_x_0, c_x_0 = lstm(h, lstm.h0,
                                            lstm.c0)  # bs * 512
                        hidden_states.append((h_x_0, c_x_0))
                    else:
                        h_x_0, c_x_0 = hidden_states[j]
                        hidden_states[j] = lstm(h, h_x_0, c_x_0)
                    h = hidden_states[j][0]

                query = self.linear(h)  # bs * 2048, This is the query
                energy = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
                # attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
                attn = F.softmax(energy, dim=2)
                context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)  # bs * 256
                h = torch.cat((h, context), dim=1)

                # At this point, h is the embed from the 3 lstm cells. Passing it through the projection layers
                h = self.activation(self.pl1(h))
                h = self.pl2(h)
                lsm = self.logsm(h)
                if self.is_stochastic > 0:
                    gumbel = torch.autograd.Variable(self.sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                # TODO: Do beam search later

                h = torch.max(h, dim=1)[1]
                raw_out.append(h.data.cpu().numpy()[0])
                if raw_pred is None:
                    raw_pred = lsm
                else:
                    raw_pred = torch.cat((raw_pred, lsm), dim=0)

                if h.data.cpu().numpy() == 1: # <eos> encountered
                    break

                # Primer for next character generation
                h = self.embed(h)
            output.append(raw_out)
            raw_preds.append(raw_pred)
        return output, raw_preds

class LAS(nn.Module):
    def __init__(self, args, output_size, max_seq_len, cuda):
        super(LAS, self).__init__()
        self.hidden_size = args.hidden_dim #32
        self.embedding_dim = args.embed_dim #40
        self.max_seq_len = max_seq_len
        self.output_size = output_size
        self.encoder = Encoder(args)      #pBilstm
        self.decoder = Decoder(args, output_size, cuda)
        self.args = args
        self.apply(init_weights)
        print('Layers initialised')

    def forward(self, input, input_len, label=None, label_len=None, tf = None):
        input = input.permute(1, 0, 2)
        keys, values = self.encoder(input, input_len)
        if label is None:
            # During decoding of test data
            return self.decoder.decode(keys, values, self.args)
        else:
            # During training
            return self.decoder(keys, values, label, label_len, input_len, tf)

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
                


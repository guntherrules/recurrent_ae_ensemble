"""
Contains class for LSTM network with skip connections and class to build a sequence to sequence autoencoder based on
that LSTM. A routine for loading the random weights used to create skip connections needs to be added. So far the model
is only trainable but can not be properly loaded from saved weights.
"""

import torch
import torch.nn as nn
import numpy as np
from queue import Queue
import random

class RSCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, skip_size, is_decoder, device):
        """
        :param input_dim: LSTM input dim
        :param hidden_dim: LSTM hidden dim
        :param skip_size: size of skip connections
        :param is_decoder: encoder or decoder building mode
        :param device: device to build on
        """
        super(RSCN, self).__init__()
        self.rnn_module = nn.LSTMCell(input_dim, hidden_dim).to(device)
        self.tanh = nn.Tanh().to(device)
        self.hidden_dim = hidden_dim
        self.skip_size = skip_size
        self.is_decoder = is_decoder
        self.device = device
        # fully connected layer to cast usual LSTM output dimensions into input dimensions
        if is_decoder:
            self.linear = nn.Linear(hidden_dim, input_dim).to(device)

    def forward(self, _input, init_hidden, time_steps):
        """

        :param _input: [batchsize, timesteps, input_dim]
        :param init_hidden: hidden state tuple (h, c) with dimensions ([batchsize, hidden_dim], [batchsize, hidden_dim])
                            this is to initialize the first hidden state of the model when building a decoder
        :param time_steps:
        :return: RSCN outputs, last hidden state (h, c)
        """

        # if encoder, initialize first hidden state using zeros
        if self.is_decoder == False:
            cur_hidden = (torch.rand(_input.shape[0], self.hidden_dim, device=self.device),
                          torch.rand(_input.shape[0], self.hidden_dim, device=self.device))
        # if decoder, initialize first hidden state with last hidden state from encoder
        else:
            cur_hidden = (init_hidden[0].to(self.device), init_hidden[1].to(self.device))

        # states for skip connection initialized with dummies
        skip_states = Queue(maxsize=self.skip_size)
        for i in range(self.skip_size):
            skip_states.put(cur_hidden)

        output = []
        h = _input[:, 0, :].to(self.device)
        weights = []
        for time_step in range(time_steps):
            if self.is_decoder == False:
                cur_input = _input[:, time_step, :].to(self.device)
            # for decoder use the output from last iteration as new input
            else:
                cur_input = h

            cur_hidden = self.rnn_module(cur_input, cur_hidden)

            cur_skip_hidden = skip_states.get()
            # print(cur_skip_hidden)
            # cur_skip_hidden = self.rnn_module(cur_skip_hidden, cur_input)
            skip_states.put(cur_hidden)

            if time_step < self.skip_size:
                w1 = torch.tensor([1]).to(self.device)
                w2 = torch.tensor([0]).to(self.device)
            else:
                w1, w2 = self.random_weights()
            weights.append([w1, w2])
            cur_hidden = self.make_skip_connection(w1, w2, cur_hidden, cur_skip_hidden)
            h = cur_hidden[0]
            c = cur_hidden[1]

            # project output to input size for decoder
            if self.is_decoder:
                h = self.linear(h)

            output.append(h)

        return torch.stack(output, dim=1).to(self.device), cur_hidden

    def random_weights(self):
        w1 = np.random.randint(2, size=1)
        if w1 == 0:
            w2 = [1]
        else:
            w2 = np.random.randint(2, size=1)
        w1 = torch.tensor(w1).to(self.device)
        w2 = torch.tensor(w2).to(self.device)
        return (w1, w2)

    def make_skip_connection(self, w1, w2, hidden, hidden_skip):
        norm = 1 / (w1[-1] + w2[-1])
        h = (w1 * hidden[0] + w2 * self.tanh(hidden_skip[0])) * norm
        c = (w1 * hidden[1] + w2 * self.tanh(hidden_skip[1])) * norm
        return (h, c)

    def cpu(self):
        #overload standard cpu function for custom model class
        self.device = torch.device('cpu')
        self.rnn_module.cpu()
        self.tanh.cpu()
        if self.is_decoder:
            self.linear.cpu()


class rand_seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, skip_max, device):
        """
        :param input_dim: LSTM network input
        :param hidden_dim: LSTM hidden size
        :param skip_max: maximum length of skip connections
        :param device: device to load the model on
        """
        super(rand_seq2seq, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skip_max = skip_max
        self.skip_sizes = self.random_skip_lengths()
        self.encoder = RSCN(input_dim, hidden_dim, self.skip_sizes[0], is_decoder=False, device=device)
        self.decoder = RSCN(input_dim, hidden_dim, self.skip_sizes[1], is_decoder=True, device=device)
        self.device = device

    def forward(self, _input):
        encoded, last_hidden = self.encoder(_input, init_hidden=None, time_steps=_input.shape[1])
        init_dec = torch.rand(_input.shape[0], 1, _input.shape[2], device=self.device)
        decoded, last_hidden = self.decoder(init_dec, init_hidden=last_hidden, time_steps=_input.shape[1])

        return torch.flip(decoded, [1]).to(self.device)

    def random_skip_lengths(self):
        skip_size1 = random.randint(1, self.skip_max)
        skip_size2 = random.randint(1, self.skip_max)

        return (skip_size1, skip_size2)

    def cpu(self):
        self.device = torch.device('cpu')
        self.encoder.cpu()
        self.decoder.cpu()
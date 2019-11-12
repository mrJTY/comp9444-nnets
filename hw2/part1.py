#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""

import torch


class rnn(torch.nn.Module):

    def __init__(self):
        super(rnn, self).__init__()

        self.ih = torch.nn.Linear(64, 128)
        self.hh = torch.nn.Linear(128, 128)

        self.W_ih = torch.randn(128)
        self.W_hh = torch.randn(128)

    def rnnCell(self, input, hidden):
        """
        TODO: Using only the above defined linear layers and a tanh
              activation, create an Elman RNN cell.  You do not need
              to include an output layer.  The network should take
              some input (inputDim = 64) and the current hidden state
              (hiddenDim = 128), and return the new hidden state.
        """
        wih_x = self.ih(input) * hidden
        whh_hidden = self.hh(wih_x) * hidden
        new_hidden_state = torch.tanh(wih_x + whh_hidden)
        return new_hidden_state

    def forward(self, input):
        hidden = torch.zeros(128)
        """
        TODO: Using self.rnnCell, create a model that takes as input
              a sequence of size [seqLength, batchSize, inputDim]
              and passes each input through the rnn sequentially,
              updating the (initally zero) hidden state.
              Return the final hidden state after the
              last input in the sequence has been processed.
        """
        new_hidden_state = self.rnnCell(input, hidden)
        return new_hidden_state


class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        TODO: Define self.net using a single PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        input_size = 64
        hidden_size = 128
        self.net = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        _, hidden = self.net(input)
        return hidden


def lstm(input, hiddenSize):
    """
    TODO: Let variable lstm be an instance of torch.nn.LSTM.
          Variable input is of size [batchSize, seqLength, inputDim]
    """
    # batch_first: input and output tensors are provided as (batch, seq, feature)
    batch_size = input.shape[0]
    seq_length = input.shape[1]
    input_size = input.shape[2]
    cell = torch.nn.LSTM(input_size=input_size, hidden_size=hiddenSize, batch_first=True)
    return cell(input)


def conv(input, weight):
    """
    TODO: Return the convolution of input and weight tensors,
          where input contains sequential data.
          The convolution should be along the sequence axis.
          input is of size [batchSize, inputDim, seqLength]
    """
    batch_size = input.shape[0]
    input_dim = input.shape[1]
    seq_len = input.shape[2]
    x = input.reshape([batch_size, seq_len, input_dim])
    w = weight.reshape([batch_size, seq_len, input_dim])
    return torch.conv1d(input=x, weight=w)

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

    def rnnCell(self, input, hidden):
        """
        TODO: Using only the above defined linear layers and a tanh
              activation, create an Elman RNN cell.  You do not need
              to include an output layer.  The network should take
              some input (inputDim = 64) and the current hidden state
              (hiddenDim = 128), and return the new hidden state.
        """
        x = self.ih(input)
        hidden = self.hh(hidden)
        new_hidden_state = torch.tanh(x + hidden)
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
        seq_len = input.size(0)
        # Pass through the seq
        for i in range(0, seq_len):
            hidden = self.rnnCell(input[i], hidden)
        return hidden


class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        TODO: Define self.net using a single PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        self.net = torch.nn.RNN(input_size=64, hidden_size=128, num_layers=1)

    def forward(self, input):
        seq_len = input.size(0)
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

    # Permuted to achieve the required size
    # [batch_size, in_channels, sequence_length] (see docs).
    # https://discuss.pytorch.org/t/how-to-convolve-along-a-single-axis/56353

    return torch.conv1d(input=input, weight=weight)

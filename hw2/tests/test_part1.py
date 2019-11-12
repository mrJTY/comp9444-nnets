import torch

import hw2.part1 as p1


# def test_elmann_cell():
#     rnn = p1.rnn()
#     input = torch.tensor()
#     out = rnn.forward()
#
#     N_INPUT = 4
#     N_NEURONS = 1
#
#     X0_batch = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 0],
#                              [6, 7, 8, 0], [9, 0, 1, 0]],
#                             dtype=torch.float)  # t=0 => 4 X 4
#
#     X1_batch = torch.tensor([[9, 8, 7, 0], [0, 0, 0, 0],
#                              [6, 5, 4, 0], [3, 2, 1, 0]],
#                             dtype=torch.float)  # t=1 => 4 X 4
#

def test_lstm():
    batch_size = 4
    seq = 2
    input_dim = 3
    hidden_size = 3
    input = torch.randn((batch_size, seq, input_dim))
    out, hidden = p1.lstm(input, hidden_size)

def test_rnn_cell():

    seq_len = 1
    batch_size = 1
    input_dim = 64
    input_vector = torch.randn(seq_len, batch_size, input_dim)

    rnn = p1.rnn()
    new_hidden = rnn.forward(input_vector)

    assert new_hidden.shape == torch.Size([1, 1, 128])


def test_rnn_simplified():
    input_vector = torch.randn(1, 1, 64)
    rnn = p1.rnnSimplified()
    new_hidden = rnn.forward(input_vector)

    assert new_hidden.shape == torch.Size([1, 1, 128])

def test_conv():
    batch_size = 1
    input_dim = 64
    seq_len = 1
    input_vector = torch.randn(batch_size, input_dim, seq_len,)
    weight = torch.randn(batch_size, input_dim, seq_len)
    result = p1.conv(input_vector, weight)
    print(result)

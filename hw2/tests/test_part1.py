import hw2 as p1
import torch

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




import torch
from torch import nn
import pytest
import torch.nn.functional as F

@pytest.mark.skip()
def test_conv():
    a = torch.randn(64, 251, 50)
    a = a.permute(0, 2, 1)
    # m = nn.Conv1d(100, 100, 1)
    m = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=1)
    out = m(a)
    print(out.size())
    # print(m)

def test_max_pool_over_time():
    # Suppose i have 64 batch with 50 features
    # and 21 seq
    inputs = torch.randn(64, 50, 21)

    # Let's get the max time
    conv = torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=4)
    out_conv = conv(inputs)
    print(out_conv.shape)

    size_of_seq = int(inputs.size(2) / 2)
    out_max_pool = F.max_pool1d(out_conv, kernel_size=size_of_seq)
    print(out_max_pool.shape)

    print(out_max_pool.reshape(64, 50).shape)



import torch
from src.part1 import *

def test_simple_addition():
    x = torch.ones(2, 2, requires_grad=True)
    y = torch.ones(2, 2, requires_grad=True)
    summation = simple_addition(x, y)
    expected_summation = x + y
    assert summation.equal(expected_summation)

def test_reshape():
    x = torch.tensor([[1,2]])
    expected = torch.tensor([[1], [2]])
    reshaped = simple_reshape(x, (2, 1))
    assert expected.equal(reshaped)

def test_flatten():
    x = torch.tensor([[1,2], [3,4]])
    flattened = simple_flat(x)
    assert flattened.equal(torch.tensor([1, 2, 3, 4]))

def test_transpose():
    x = torch.tensor([[1,2]])
    expected = torch.tensor([[1], [2]])
    reshaped = simple_transpose(x)
    assert expected.equal(reshaped)

def test_permute():
    # x = torch.tensor([1,2,3])
    x = torch.randn(2, 3, 5)
    size = x.size()
    permuted = simple_permute(x, [2, 0, 1])

    assert permuted.size() == torch.Size([5, 2, 3])


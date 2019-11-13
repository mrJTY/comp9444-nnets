#!/usr/bin/env python3
"""
part2.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.

YOU MAY MODIFY THE LINE net = NetworkLstm().to(device)
"""

import numpy as np

import torch
import torch.nn as tnn
import torch.optim as topti

from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class NetworkLstm(tnn.Module):
    """
    Implement an LSTM-based network that accepts batched 50-d
    vectorized inputs, with the following structure:
    LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkLstm, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.input_dim = 50
        self.hidden_dim = 100
        self.num_layers = 3
        self.batch_size = 64
        self.lstm_layer = torch.nn.LSTM(input_size=50, hidden_size=self.hidden_dim, batch_first=True,
                                        num_layers=self.num_layers)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=1)

    def zero_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        )

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """

        # Reference: https://www.youtube.com/watch?v=ogZi5oIo4fI

        batch_size = input.size(0)
        out, hidden = self.lstm_layer(input, self.zero_hidden(batch_size))

        fc2_output = self.fc2(out)
        relu_output = torch.relu(fc2_output)

        fc3_output = self.fc3(relu_output)
        softmax_output = torch.softmax(fc3_output, dim=1)

        # Get only the last output
        out = torch.zeros(batch_size)
        for i in range(0, batch_size):
            out[i] = softmax_output[i][-1][0]

        # output must just be a single dimension 64 tensor(batch) x 1
        assert out.shape == torch.Size([batch_size])
        return out


# Class for creating the neural network.
class NetworkCnn(tnn.Module):
    """
    Implement a Convolutional Neural Network.
    All conv layers should be of the form:
    conv1d(channels=50, kernel size=8, padding=5)

    Conv -> ReLu -> maxpool(size=4) -> Conv -> ReLu -> maxpool(size=4) ->
    Conv -> ReLu -> maxpool over time (global pooling) -> Linear(1)

    The max pool over time operation refers to taking the
    maximum val from the entire output channel. See Kim et. al. 2014:
    https://www.aclweb.org/anthology/D14-1181/
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkCnn, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.conv1 = torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        self.conv2 = torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        self.conv3 = torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        self.fc4 = torch.nn.Linear(in_features=50, out_features=1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """

        # Permute the seq length and dimension
        # x must be [batch_size, features, seq_length]
        x = input.permute(0, 2, 1)
        batch_size = x.size(0)
        input_size = x.size(1)
        kernel_size_of_seq = int(x.size(2) / 2)

        assert x.size(1) == 50

        # FIXME : RuntimeError: Expected 3-dimensional input for 3-dimensional weight 50 50 8,
        # but got 2-dimensional input of size [177, 50] instead
        out_conv1 = self.conv1(x)
        out_relu1 = torch.nn.functional.relu(out_conv1)
        out_pool1 = torch.nn.functional.max_pool1d(out_relu1, kernel_size=4)

        out_conv2 = self.conv2(out_pool1)
        out_relu2 = torch.nn.functional.relu(out_conv2)
        out_pool2 = torch.nn.functional.max_pool1d(out_relu2, kernel_size=4)

        # Max pool picks the maximum convovled feauture over the sequence
        out_conv3 = self.conv3(out_pool2)
        out_relu3 = torch.nn.functional.relu(out_conv3)

        done = False
        while not done:
            try:
                out_max_pool_over_time = torch.nn.functional.max_pool1d(out_relu3, kernel_size=kernel_size_of_seq)
                done = True
            except RuntimeError:
                kernel_size_of_seq = kernel_size_of_seq - 1

        out_max_pool_over_time = out_max_pool_over_time.reshape(batch_size, input_size)

        # Last fc
        out_fc4 = self.fc4(out_max_pool_over_time)
        out_fc4.reshape(batch_size)

        # Softmax output
        out_softmax = torch.softmax(out_fc4, dim=1)
        out = out_softmax.reshape(batch_size)

        return out



def lossFunc():
    """
    TODO:
    Return a loss function appropriate for the above networks that
    will add a sigmoid to the output and calculate the binary
    cross-entropy.
    """
    return torch.nn.BCELoss()


def measures(outputs, labels):
    """
    TODO:
    Return (in the following order): the number of true positive
    classifications, true negatives, false positives and false
    negatives from the given batch outputs and provided labels.

    outputs and labels are torch tensors.
    """
    # TODO: is round here ok?
    x = np.round(outputs.data.numpy())
    y = labels.data.numpy()

    tp = sum(np.where((x == 1) & (y == 1), 1, 0))
    tn = sum(np.where((x == 0) & (y == 0), 1, 0))
    fp = sum(np.where((x == 1) & (y == 0), 1, 0))
    fn = sum(np.where((x == 0) & (y == 1), 1, 0))
    return tp, tn, fp, fn


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    from imdb_dataloader import IMDB
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    # Create an instance of the network in memory (potentially GPU memory). Can change to NetworkCnn during development.
    # TODO: Switch back to NetworkLstm
    # net = NetworkLstm().to(device)
    net = NetworkCnn().to(device)

    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    # TODO: Change back to 10
    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()

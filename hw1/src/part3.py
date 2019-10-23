#!/usr/bin/env python3
"""
part3.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Linear(nn.Module):
    """
    DO NOT MODIFY
    Linear (10) -> ReLU -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # make sure inputs are flattened

        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)  # preserve batch dim

        return x


class FeedForward(nn.Module):
    """
    TODO: Implement the following network structure
    Linear (256) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> ReLU-> LogSoftmax
    """

    def __init__(self):
        super().__init__()

        input_shape = 28*28
        output_shape = 10
        self.fc1 = nn.Linear(in_features=input_shape, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # 64x784
        flattened = x.view(x.shape[0], -1)

        relu1_output = F.relu(self.fc1(flattened))
        relu2_output = F.relu(self.fc2(relu1_output))
        relu3_output = F.relu(self.fc3(relu2_output))

        softmax_output = F.log_softmax(relu3_output, dim=1)
        return softmax_output


class CNN(nn.Module):
    """
    TODO: Implement CNN Network structure

    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu -> Linear (10) -> LogSoftmax


    Hint: You will need to reshape outputs from the last conv layer prior to feeding them into
    the linear layers.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 64 # x.shape[0] # 64
        self.input_shape = 28*28 # image 28x28
        self.output_shape = 10 # num outputs

        # weight of size 10 1 5 5, expected input[64, 1, 28, 28] to have 1 channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=50, kernel_size=5, stride=1)

        # Reshape the output cnn to match. 800 = 50channels x 4x4 convulution output
        self.fc3 = nn.Linear(in_features=800, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=self.output_shape)

    def forward(self, x):
        # 64x784
        n_rows = x.shape[0]
        flattened = x.view(n_rows, -1) # or use x.shape[0]

        relu_output1 = F.relu(self.conv1(x))
        maxpool_output2 = F.max_pool2d(relu_output1, 2, 2)

        relu_output2 = F.relu(self.conv2(maxpool_output2))
        # 64batch x 50 channels x (4x4 per convovled image)
        maxpool_output2 = F.max_pool2d(relu_output2, 2, 2)

        # Reshape the CNN to match linear: 64x800
        cnn_output_reshaped = maxpool_output2.view(n_rows, 50*4*4)

        relu_output3 = F.relu(self.fc3(cnn_output_reshaped))

        softmax_output = F.log_softmax(relu_output3, dim=1)

        return softmax_output



class NNModel:
    def __init__(self, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        DO NOT MODIFY
        """

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

        # Download and load the test data
        testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        self.model = network

        """
        TODO: Set appropriate loss function such that learning is equivalent to minimizing the
        cross entropy loss. Note that we are outputting log-softmax values from our networks,
        not raw softmax values, so just using torch.nn.CrossEntropyLoss is incorrect.

        Hint: All networks output log-softmax values (i.e. log probabilities or.. likelihoods.).

        https://medium.com/@zhang_yang/understanding-cross-entropy-implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
        """
        self.lossfn = F.nll_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

    def _process_one_row(self, X):
        row = X[0].reshape(28,28)
        # First 8 rows
        for i in X[1:9]:
            img = i.reshape(28,28)
            # Stack horizontally...
            row = np.hstack((row, i.reshape(28,28)))
        return row

    def view_batch(self):
        """
        TODO: Display first batch of images from trainloader in 8x8 grid

        Do not make calls to plt.imshow() here

        Return:
           1) A float32 numpy array (of dim [28*8, 28*8]), containing a tiling of the batch images,
           place the first 8 images on the first row, the second 8 on the second row, and so on

           2) An int 8x8 numpy array of labels corresponding to this tiling
        """

        pixels_x = 28
        pixels_y = 28

        # Get one batch of data (first 64)
        it = iter(self.trainloader)
        X, y = it.next()


        segment_length = 8
        rnge = range(64)
        cuts = [rnge[x:x+segment_length] for x in range(0,len(rnge),segment_length)]
        print(cuts)

        rows = [self._process_one_row(X[i]) for i in cuts]

        images = np.vstack((rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]))

        y_out = y.reshape(8,8)
        return images, y_out


    def train_step(self):
        """
        Used for submission tests and may be usefull for debugging
        DO NOT MODIFY
        """
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self):
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                log_ps = self.model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples


def plot_result(results, names):
    """
    Take a 2D list/array, where row is accuracy at each epoch of training for given model, and
    names of each model, and display training curves
    """
    for i, r in enumerate(results):
        plt.plot(range(len(r)), r, label=names[i])
    plt.legend()
    plt.title("KMNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./part_2_plot.png")


def main():
    models = [Linear(), FeedForward(), CNN()]  # Change during development
    epochs = 10
    results = []

    # Can comment the below out during development
    images, labels = NNModel(Linear(), 0.003).view_batch()
    print(labels)
    plt.imshow(images, cmap="Greys")
    plt.show()

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        m = NNModel(model, 0.003)

        accuracies = [0]
        for e in range(epochs):
            m.train_epoch()
            accuracy = m.eval()
            print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
            accuracies.append(accuracy)
        results.append(accuracies)

    plot_result(results, [m.__class__.__name__ for m in models])


if __name__ == "__main__":
    main()

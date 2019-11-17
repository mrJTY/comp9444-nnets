import torch
import torch.nn as tnn
import torch.optim as topti
from imdb_dataloader import IMDB
from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input_dim = 50
        self.hidden_dim = 100
        self.num_layers = 4
        self.batch_size = 64
        self.lstm_layer = torch.nn.LSTM(input_size=50, hidden_size=self.hidden_dim, batch_first=True,
                                        num_layers=self.num_layers)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        batch_size = input.size(0)
        lstm_out, hidden = self.lstm_layer(input, self.zero_hidden(batch_size))

        # Take the final state of the lstm
        lstm_out_final = torch.zeros(batch_size, self.hidden_dim)
        for i in range(0, batch_size):
            lstm_out_final[i] = lstm_out[i][-1]

        # Pass to a fully connected network
        fc2_output = self.fc2(lstm_out_final)
        relu2_output = torch.relu(fc2_output)

        # Apply a dropout at the final fully connected layer
        fc3_output = self.fc3(relu2_output)
        drop_out = torch.nn.Dropout(p=0.2)
        out_drop = drop_out(fc3_output)

        out = out_drop.reshape(batch_size)
        return out

    def zero_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        )


class PreProcessing():

    def pre(x):
        """Called after tokenization"""

        # Source: https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                           'yourself',
                           'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                           'itself',
                           'they', 'them',
                           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                           'those',
                           'am', 'is',
                           'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                           'did',
                           'doing', 'a',
                           'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                           'for',
                           'with', 'about',
                           'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                           'from',
                           'up', 'down', 'in', 'out',
                           'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                           'where',
                           'why', 'how',
                           'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                           'not',
                           'only', 'own',
                           'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

        # x is a list of words
        # Do some transformations to remove stop words and punctuations
        no_stop_words = [i for i in x if i not in stop_words]
        no_punct = [''.join([i for i in word if i.isalpha()]) for word in no_stop_words]
        return no_punct

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization
        Fixed: Should only return vocab
        """
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return torch.nn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

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

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()

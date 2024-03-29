#!/usr/bin/env python3
"""
part2.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pdb

class LinearModel:
    def __init__(self, num_inputs, learning_rate):
        """
        Model is very similar to the Perceptron shown in Lectures 1c, slide 12, except that:
        (1) the bias is indexed by w(n+1) rather than w(0), and
        (2) the activation function is a (continuous) sigmoid rather than a (discrete) step function.

        x1 ----> * w1 ----\
        x2 ----> * w2 -----\
        x3 ----> * w3 ------\
        ...
                             \
        xn ----> * wn -------+--> s --> activation ---> z
        1  ----> * w(n+1) --/
        """
        self.num_inputs = num_inputs
        self.lr = learning_rate
        self.weights = np.asarray([1.0, -1.0, 0.0])  # Initialize as straight line

    def activation(self, x):
        """
        TODO: Implement a sigmoid activation function that accepts a float and returns
        a float, but raises a Value error if a boolean, list or numpy array is passed in
        hint: consider np.exp()
        """
        if type(x) in [bool, np.array, list]:
            raise ValueError("Expecting a float as input for activation")

        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        TODO: Implement the forward pass (inference) of a the model.

        inputs is a numpy array. The bias term is the last element in self.weights.
        hint: call the activation function you have implemented above.
        """
        z = self.calc_z(inputs)
        a = self.activation(z)
        return a

    def calc_z(self, inputs):
        # z = wT x + b
        bias_weight = self.weights[-1]
        input_weights = np.array(self.weights[0:len(self.weights)-1])
        return np.dot(input_weights, inputs) + bias_weight


    @staticmethod
    def loss(prediction, label):
        """
        TODO: Return the cross entropy for the given prediction and label
        hint: consider using np.log()
        """

        # Cross entropy,
        # Return a low cost if they match
        # But return a high cost when the values don't match
        if label == 1:
            cost = -np.log(prediction)
        else:
            cost = -np.log(1 - prediction)
        return cost

    @staticmethod
    def error(prediction, label):
        """
        TODO: Return the difference between the label and the prediction

        For example, if label= 1 and the prediction was 0.8, return 0.2
                     if label= 0 and the preduction was 0.43 return -0.43
        """
        return label - prediction

    def backward(self, inputs, diff):
        """
        TODO: Adjust self.weights by gradient descent

        We take advantage of the simplification shown in Lecture 2b, slide 23,
        to compute the gradient directly from the differential or difference
        dE/ds = z - t (which is passed in as diff)

        The resulting weight update should look essentially the same as for the
        Perceptron Learning Rule (shown in Lectures 1c, slide 11) except that
        the error can take on any continuous value between -1 and +1,
        rather than being restricted to the integer values -1, 0 or +1.

        Note: Numpy arrays are passed by reference and can be modified in-place
        """

        z = self.calc_z(inputs)

        # Derivative of error with respect to changes in z
        # dE/dz is (z-t) which is the diff

        # Reference
        # Andrew Ng: https://www.youtube.com/watch?v=2BkqApHKwn0&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=20
        dE_dz = diff
        dz_ds = z * (1-z)
        dE_dw =  dE_dz * inputs
        dE_db = dE_dz

        bias_weight = self.weights[-1]
        input_weights = np.array(self.weights[0:len(self.weights)-1])

        bias_weight_update = self.lr * dE_db
        input_weights_update = self.lr * dE_dw

        # Update bias weight
        self.weights[-1] = bias_weigth = bias_weight + (bias_weight_update)
        # Update input weights
        self.weights[0:len(self.weights)-1] = input_weights + (input_weights_update)

    def plot(self, inputs, marker):
        """
        Plot the data and the decision boundary
        """
        xmin = inputs[:, 0].min() * 1.1
        xmax = inputs[:, 0].max() * 1.1
        ymin = inputs[:, 1].min() * 1.1
        ymax = inputs[:, 1].max() * 1.1

        x = np.arange(xmin * 1.3, xmax * 1.3, 0.1)
        plt.scatter(inputs[:25, 0], inputs[:25, 1], c="C0", edgecolors='w', s=100)
        plt.scatter(inputs[25:, 0], inputs[25:, 1], c="C1", edgecolors='w', s=100)

        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.plot(x, -(self.weights[0] * x + self.weights[2]) / self.weights[1], marker, alpha=0.6)
        plt.title("Data and decision boundary")
        plt.xlabel("x1")
        plt.ylabel("x2").set_rotation(0)


def main():
    inputs, labels = pkl.load(open("../data/binary_classification_data.pkl", "rb"))

    epochs = 400
    model = LinearModel(num_inputs=inputs.shape[1], learning_rate=0.01)

    for i in range(epochs):
        num_correct = 0
        for x, y in zip(inputs, labels):
            # Get prediction
            output = model.forward(x)

            # Calculate loss
            cost = model.loss(output, y)

            # Calculate difference or differential
            diff = model.error(output, y)

            # Update the weights
            model.backward(x, diff)

            # Record accuracy
            preds = output > 0.5  # 0.5 is midline of sigmoid
            num_correct += int(preds == y)

        print(f" Cost: {cost:8.6f} Accuracy: {num_correct / len(inputs) * 100}%")
        model.plot(inputs, "C2--")
    model.plot(inputs, "k")
    plt.show()


if __name__ == "__main__":
    main()

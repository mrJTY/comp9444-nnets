from src.part3 import *
import pytest
import numpy as np
import pdb

epochs = 10

@pytest.mark.skip(reason="Plot done")
def test_plot():
    models = [Linear(), FeedForward(), CNN()]  # Change during development
    results = []
    images, labels = NNModel(Linear(), 0.003).view_batch()
    print(labels)

    plt.imshow(images, cmap='Greys')
    plt.show()


@pytest.mark.skip(reason="slow")
def test_loss_fn():
    m = NNModel(Linear(), 0.003)

    for e in range(epochs):
        print(e)
        m.train_epoch()
        accuracy = m.eval()
        print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")

@pytest.mark.skip(reason="done")
def test_feed_forward():
    m = NNModel(FeedForward(), 0.003)

    current_acc = 0

    for e in range(epochs):
        print(e)
        m.train_epoch()
        accuracy = m.eval()
        print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")

        # Rough assertion, in theory it should improve
        assert accuracy >= current_acc
        current_acc = accuracy

def test_feed_forward():
    m = NNModel(CNN(), 0.003)

    current_acc = 0
    for e in range(epochs):
        print(f"Epoch {e}")
        m.train_epoch()
        accuracy = m.eval()
        print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")

        # Rough assertion, in theory it should improve
        assert accuracy >= current_acc
        current_acc = accuracy

# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
from random import uniform


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.layers = list()
        self.weights = list()
        self.bias = list()

        # Setting up all weights, biases
        if hidden_layer:
            # hidden_weights[i][j] <- weight from inputNode_j to hiddenNode_i
            hidden_weights = np.array(
                [[uniform(-0.5, 0.5) for j in range(input_dim)] for i in range(self.hidden_units)])
            self.weights.append(hidden_weights)

            hidden_bias = np.array([uniform(-0.5, 0.5) for j in range(self.hidden_units)])
            self.bias.append(hidden_bias)

            # output_weights[i][j] <- weight from hiddenNode_j to outputNode_i
            output_weights = np.array([[uniform(-0.5, 0.5) for j in range(self.hidden_units)] for i in range(1)])
            self.weights.append(output_weights)

            output_bias = np.array([uniform(-0.5, 0.5) for j in range(1)])
            self.bias.append(output_bias)

        else:
            # output_weights[i][j] <- weight from hiddenNode_j to outputNode_i
            output_weights = np.array([[uniform(-0.5, 0.5) for j in range(input_dim)] for i in range(1)])
            self.weights.append(output_weights)

            output_bias = np.array([uniform(-0.5, 0.5) for i in range(1)])
            self.bias.append(output_bias)

        # Setting up all layers
        input_activations = np.zeros(input_dim)
        self.layers.append({'output': input_activations})

        if hidden_layer:
            hidden_activations = np.zeros(self.hidden_units)
            self.layers.append({'output': hidden_activations})

        output_activations = np.zeros(1)
        self.layers.append({'output': output_activations})

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class

        for iteration in range(self.epochs):
            for x, y in zip(self.x_train, self.y_train):
                # Propagate the inputs forward to compute the outputs

                # Set initial input activation
                self.layers[0]['output'] = x

                # Propagate all activations
                for l in range(len(self.layers) - 1):  # Skip first layer
                    weights = self.weights[l]
                    activations = self.layers[l]['output']
                    propagated_activations = np.dot(weights, activations) + self.bias[l]
                    self.layers[l + 1]['output'] = self.sigmoid(propagated_activations)

                # Propagate deltas backward from output layer to input layer
                # Initial output delta-error calculation
                self.layers[-1]['delta'] = self.der_sigmoid(self.layers[-1]['output']) * \
                                           (y - self.layers[-1]['output'])

                for l in reversed(range(1, len(self.layers) - 1)):  # Start from the back
                    g = self.der_sigmoid(self.layers[l]['output'])
                    w_ij = np.transpose(self.weights[l])
                    delta = self.layers[l + 1]['delta']
                    self.layers[l]['delta'] = g * (np.dot(w_ij, delta))

                # Update every weight in network using deltas
                for l in range(len(self.layers) - 1):
                    self.weights[l] += self.lr * np.dot(np.transpose(np.array([self.layers[l + 1]['delta']])),
                                                        np.array([self.layers[l]['output']]))
                    self.bias[l] += self.lr * self.layers[l + 1]['delta']

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.

        # Set initial input activation
        self.layers[0]['output'] = x

        # Propagate all activations
        for l in range(len(self.layers) - 1):  # Skip first layer
            weights = self.weights[l]
            activations = self.layers[l]['output']
            propagated_activations = np.dot(weights, activations) + self.bias[l]
            self.layers[l + 1]['output'] = self.sigmoid(propagated_activations)

        return self.layers[-1]['output'][0]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, output: np.ndarray) -> np.ndarray:
        return output * (1 - output)


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()

import unittest
from Neural_Networks_and_Deep_Learning.planar_utils import load_planar_dataset, plot_decision_boundary,load_extra_datasets
import matplotlib.pyplot as plt
import pylab
import numpy as np
import sklearn.linear_model
from Neural_Networks_and_Deep_Learning.testCases import *
import Neural_Networks_and_Deep_Learning.Planar_Data_Classification_with_One_Hidden_Layer as tc

class Planar_Data_Classification_with_One_Hidden_Layer_Test(unittest.TestCase):

    def setUp(self):
        self.X, self.Y = load_planar_dataset()

    def test_visualize_the_data(self):

        Y = np.squeeze(self.Y)
        plt.scatter(self.X[0, :], self.X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        pylab.show()

    def test_show_shape(self):
        shape_X = self.X.shape
        shape_Y = self.Y.shape
        m = self.Y.shape[1]
        print('The shape of X is: ' + str(shape_X))
        print('The shape of Y is: ' + str(shape_Y))
        print('I have m = %d training examples!' % (m))

    def test_simple_LR(self):
        clf = sklearn.linear_model.LogisticRegressionCV()
        clf.fit(self.X.T, self.Y.T)

        LR_predictions = clf.predict(self.X.T)
        print('Accuracy of logistic regression: %d ' % float(
            (np.dot(self.Y, LR_predictions) + np.dot(1 - self.Y, 1 - LR_predictions)) / float(self.Y.size) * 100) +
              '% ' + "(percentage of correctly labelled datapoints)")

        plot_decision_boundary(lambda x: clf.predict(x), self.X, self.Y)
        plt.title("Logistic Regression")
        pylab.show()

    def test_layer_sizes(self):
        X_assess, Y_assess = layer_sizes_test_case()
        (n_x, n_y) = tc.layer_sizes(X_assess, Y_assess)
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the output layer is: n_y = " + str(n_y))

    def test_initialize_parameters(self):
        n_x, n_h, n_y = initialize_parameters_test_case()

        parameters = tc.initialize_parameters(n_x, n_h, n_y)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test_forward_propagation(self):
        X_assess, parameters = forward_propagation_test_case()

        A2, cache = tc.forward_propagation(X_assess, parameters)

        # Note: we use the mean here just to make sure that your output matches ours.
        print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    def test_compute_cost(self):
        A2, Y_assess, parameters = compute_cost_test_case()
        print("cost = " + str(tc.compute_cost(A2, Y_assess)))

    def test_backward_propagation(self):
        parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

        grads = tc.backward_propagation(parameters, cache, X_assess, Y_assess)
        print("dW1 = " + str(grads["dW1"]))
        print("db1 = " + str(grads["db1"]))
        print("dW2 = " + str(grads["dW2"]))
        print("db2 = " + str(grads["db2"]))


    def test_update_parameters(self):
        parameters, grads = update_parameters_test_case()
        parameters = tc.update_parameters(parameters, grads)

        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test_nn_model(self):
        X_assess, Y_assess = nn_model_test_case()

        parameters = tc.nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))


    def test_predict(self):
        parameters, X_assess = predict_test_case()
        predictions = tc.predict(parameters, X_assess)
        print("predictions mean = " + str(np.mean(predictions)))

    def test_show(self):
        parameters = tc.nn_model(self.X, self.Y, n_h=4, num_iterations=10000, print_cost=True)
        predictions = tc.predict(parameters, self.X)
        print('Accuracy: %d' % float(
            (np.dot(self.Y, predictions.T) + np.dot(1 - self.Y, 1 - predictions.T)) / float(self.Y.size) * 100) + '%')
        # Plot the decision boundary
        plot_decision_boundary(lambda x: tc.predict(parameters, x.T), self.X, self.Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))
        pylab.show()

    def test_hidden_layer_size(self):
        plt.figure(figsize=(16, 32))
        hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
        for i, n_h in enumerate(hidden_layer_sizes):
            plt.subplot(5, 2, i + 1)
            plt.title('Hidden Layer of size %d' % n_h)
            parameters = tc.nn_model(self.X, self.Y, n_h, num_iterations=5000)
            plot_decision_boundary(lambda x: tc.predict(parameters, x.T), self.X, self.Y)
            predictions = tc.predict(parameters, self.X)
            accuracy = float((np.dot(self.Y, predictions.T) + np.dot(1 - self.Y, 1 - predictions.T)) / float(self.Y.size) * 100)
            print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        pylab.show()

    def test_other_dataset(self):
        # Datasets
        noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

        datasets = {"noisy_circles": noisy_circles,
                    "noisy_moons": noisy_moons,
                    "blobs": blobs,
                    "gaussian_quantiles": gaussian_quantiles}

        ### START CODE HERE ### (choose your dataset)
        dataset = "noisy_moons"
        ### END CODE HERE ###

        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])

        # make blobs binary
        if dataset == "blobs":
            Y = Y % 2

        # Visualize the data
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        pylab.show()
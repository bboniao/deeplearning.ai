import unittest
import Neural_Networks_and_Deep_Learning.Building_your_Deep_Neural_Network_Step_by_Step as tc
from Neural_Networks_and_Deep_Learning.nn_testCases import *

class Building_your_Deep_Neural_Network_Step_by_Step_Test(unittest.TestCase):

    def test_initialize_parameters(self):
        parameters = tc.initialize_parameters(2,2,1)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test_initialize_parameters_deep(self):
        parameters = tc.initialize_parameters_deep([5, 4, 3])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test_linear_forward(self):
        A, W, b = linear_forward_test_case()

        Z, linear_cache = tc.linear_forward(A, W, b)
        print("Z = " + str(Z))

    def test_linear_activation_forward(self):
        A_prev, W, b = linear_activation_forward_test_case()

        A, linear_activation_cache = tc.linear_activation_forward(A_prev, W, b, activation="sigmoid")
        print("With sigmoid: A = " + str(A))

        A, linear_activation_cache = tc.linear_activation_forward(A_prev, W, b, activation="relu")
        print("With ReLU: A = " + str(A))

    def test_L_model_forward(self):
        X, parameters = L_model_forward_test_case()
        AL, caches = tc.L_model_forward(X, parameters)
        print("AL = " + str(AL))
        print("Length of caches list = " + str(len(caches)))

    def test_compute_cost(self):
        Y, AL = compute_cost_test_case()

        print("cost = " + str(tc.compute_cost(AL, Y)))

    def test_linear_backward(self):
        dZ, linear_cache = linear_backward_test_case()
        dA_prev, dW, db = tc.linear_backward(dZ, linear_cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

    def test_linear_activation_backward(self):
        AL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = tc.linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db) + "\n")

        dA_prev, dW, db = tc.linear_activation_backward(AL, linear_activation_cache, activation="relu")
        print("relu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

    def test_L_model_backward(self):
        X_assess, Y_assess, AL, caches = L_model_backward_test_case()
        grads = tc.L_model_backward(AL, Y_assess, caches)
        print("dW1 = " + str(grads["dW1"]))
        print("db1 = " + str(grads["db1"]))
        print("dA1 = " + str(grads["dA1"]))

    def test_update_parameters(self):
        parameters, grads = update_parameters_test_case()
        parameters = tc.update_parameters(parameters, grads, 0.1)

        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        print("W3 = " + str(parameters["W3"]))
        print("b3 = " + str(parameters["b3"]))
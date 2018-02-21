import unittest
import Neural_Networks_and_Deep_Learning.Deep_Neural_Network_Application_Image_Classification as tc
from Neural_Networks_and_Deep_Learning.dnn_app_utils import *

class Deep_Neural_Network_Application_Image_Classification_Test(unittest.TestCase):
    def test_two_layer_model(self):
        n_x = 12288  # num_px * num_px * 3
        n_h = 7
        n_y = 1

        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                               -1).T  # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        layers_dims = (n_x, n_h, n_y)
        train_x = train_x_flatten / 255.
        test_x = test_x_flatten / 255.
        parameters = tc.two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500,
                                     print_cost=True)
        predictions_train = predict(train_x, train_y, parameters)
        print("predictions_train = " + str(predictions_train))
        predictions_test = predict(test_x, test_y, parameters)
        print("predictions_test = " + str(predictions_test))

    def test_L_layer_model(self):
        layers_dims = [12288, 20, 7, 5, 1]
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                               -1).T  # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        train_x = train_x_flatten / 255.
        test_x = test_x_flatten / 255.
        parameters = tc.L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

        pred_train = predict(train_x, train_y, parameters)
        print("pred_train = " + str(pred_train))
        pred_test = predict(test_x, test_y, parameters)
        print("pred_test = " + str(pred_test))
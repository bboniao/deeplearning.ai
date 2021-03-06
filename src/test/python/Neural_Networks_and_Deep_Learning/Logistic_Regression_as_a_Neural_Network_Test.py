import unittest
import Neural_Networks_and_Deep_Learning.Logistic_Regression_as_a_Neural_Network as tc
from Neural_Networks_and_Deep_Learning.lr_utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np


class Python_Basics_with_Numpy_Test(unittest.TestCase):

    def setUp(self):
        self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y, self.classes = load_dataset()


    def test_train_set_x_orig(self):

        index = 25
        plt.imshow(self.train_set_x_orig[index])
        plt.show()

    def test_shape(self):
        m_train = self.train_set_y.shape[1]
        m_test = self.test_set_y.shape[1]
        num_px = self.train_set_x_orig.shape[1]

        print("Number of training examples: m_train = " + str(m_train))
        print("Number of testing examples: m_test = " + str(m_test))
        print("Height/Width of each image: num_px = " + str(num_px))
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("train_set_x shape: " + str(self.train_set_x_orig.shape))
        print("train_set_y shape: " + str(self.train_set_y.shape))
        print("test_set_x shape: " + str(self.test_set_x_orig.shape))
        print("test_set_y shape: " + str(self.test_set_y.shape))

    def test_flatten(self):
        train_set_x_flatten = self.train_set_x_orig.reshape(self.train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = self.test_set_x_orig.reshape(self.test_set_x_orig.shape[0], -1).T
        print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
        print("train_set_y shape: " + str(self.train_set_y.shape))
        print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
        print("test_set_y shape: " + str(self.test_set_y.shape))
        print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))


    def test_init_zeros(self):
        result = tc.init_zeros(4)
        print(result)


    def test_propagate(self):
        w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
        grads, cost = tc.propagate(w, b, X, Y)
        print("dw = " + str(grads["dw"]))
        print("db = " + str(grads["db"]))
        print("cost = " + str(cost))


    def test_optimize(self):
        w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
        params, grads, costs = tc.optimize(w, b, X, Y, iterate_num=100, learning_rate=0.009, print_cost=False)
        print("w = " + str(params["w"]))
        print("b = " + str(params["b"]))
        print("dw = " + str(grads["dw"]))
        print("db = " + str(grads["db"]))

    def test_predict(self):
        w = np.array([[0.1124579], [0.23106775]])
        b = 1.5593049248448891
        X = np.array([[1, 2], [3, 4]])
        print("predictions = " + str(tc.predict(w, b, X)))


    def test_model(self):
        d = tc.model(num_iterations=2000, learning_rate=0.005,
                  print_cost=True)
        index = 5
        print("y = " + str(self.test_set_y[0, index]) + ", you predicted that it is a \"" + self.classes[
            int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")
        plt.imshow(self.train_set_x_orig[index, :])
        plt.show()


    def test_iterate_num(self):
        d = tc.model(num_iterations=2000, learning_rate=0.005,
                     print_cost=True)
        costs = np.squeeze(d['costs'])
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(d["learning_rate"]))
        plt.show()

    def test_learning_rate(self):
        learning_rates = [0.01, 0.001, 0.0001]
        models = {}
        for i in learning_rates:
            print("learning rate is: " + str(i))
            models[str(i)] = tc.model(num_iterations=1500,
                                   learning_rate=i, print_cost=False)
            print('\n' + "-------------------------------------------------------" + '\n')

        for i in learning_rates:
            plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

        plt.ylabel('cost')
        plt.xlabel('iterations')

        legend = plt.legend(loc='upper center', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        plt.show()


    def test_predict_cat(self):
        model = tc.model()
        for i in range(6):
            print(tc.predict_cat(str(i) + ".jpg", model))
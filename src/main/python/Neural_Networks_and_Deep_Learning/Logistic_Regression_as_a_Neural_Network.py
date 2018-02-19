import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Neural_Networks_and_Deep_Learning.lr_utils import load_dataset
from Neural_Networks_and_Deep_Learning.Python_Basics_with_Numpy import  sigmoid
import os

#plt.show()


def init_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    """

    :param w: w.shape = (num_px * num_px * 3, 1)
    :param b: b = 0
    :param X: (num_px * num_px * 3, number of examples)
    :param Y: (1, number of examples)
    :return:
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, iterate_num, learning_rate, print_cost=False):
    costs = []
    for i in range(iterate_num):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction


def model(num_iterations=2000, learning_rate=0.5, print_cost=False):
    train_set_x_orig, train_y, test_set_x_orig, test_y, classes = load_dataset()
    assert (train_set_x_orig.shape == (209, 64, 64, 3))
    assert (train_y.shape == (1, 209))
    assert (test_set_x_orig.shape == (50, 64, 64, 3))
    assert (test_y.shape == (1, 50))
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    assert (train_set_x_flatten.shape == (12288, 209))
    assert (test_set_x_flatten.shape == (12288, 50))
    train_x = train_set_x_flatten/255
    test_x = test_set_x_flatten/255

    w,b = init_zeros(train_x.shape[0])
    param, grads, costs = optimize(w, b, train_x, train_y,num_iterations, learning_rate, print_cost)
    w = param["w"]
    b = param["b"]
    Y_prediction_test = predict(w, b, test_x)
    Y_prediction_train = predict(w, b, train_x)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def predict_cat(picname, model):
    root_path = os.path.split(os.path.realpath(__file__))[0]
    path = root_path + "/images/" + picname
    image = np.array(ndimage.imread(path, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    my_predicted_image = predict(model["w"], model["b"], my_image)
    return my_predicted_image
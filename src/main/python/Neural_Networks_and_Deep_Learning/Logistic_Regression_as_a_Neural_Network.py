import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Neural_Networks_and_Deep_Learning.lr_utils import load_dataset
from Neural_Networks_and_Deep_Learning.Python_Basics_with_Numpy import  sigmoid
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
    costs=[]
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

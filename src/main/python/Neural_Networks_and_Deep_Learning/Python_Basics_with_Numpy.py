import math


def basic_sigmoid(x):
    return 1. / (1 + math.exp(-x))


import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def simgoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)


def image2vector(image):
    return image.reshwape(image.shape[0] * image.shape[1] * image.shape[2], 1)


def normalizeRows(x):
    normalize_x = np.linalg.norm(x, axis=1, keepdims=True)
    return x / normalize_x


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum


def l1(yhat, y):
    return np.sum(np.abs(yhat - y))


def l2(yhat, y):
    return np.dot(yhat - y, yhat-y)
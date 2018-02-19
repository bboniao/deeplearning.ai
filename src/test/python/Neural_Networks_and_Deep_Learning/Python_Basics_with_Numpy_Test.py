import unittest
import Neural_Networks_and_Deep_Learning.Python_Basics_with_Numpy as tc
import numpy as np


class Test(unittest.TestCase):
    def test_basic_sigmiod(self):
        result = tc.basic_sigmoid(3)
        self.assertEqual(result, 0.9525741268224334)

    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        result = tc.sigmoid(x)
        y = [0.7310585786300049, 0.8807970779778823, 0.9525741268224334]
        for index in range(len(y)):
            self.assertEqual(result[index], y[index])

    def test_simgoid_derivative(self):
        x = np.array([1, 2, 3])
        result = tc.simgoid_derivative(x)
        y = np.array([0.19661193324148185, 0.10499358540350662, 0.045176659730912])
        for index in range(len(result)):
            self.assertEqual(result[index], y[index])

    def test_image2vector(self):
        image = np.array([[[0.67826139, 0.29380381],
                           [0.90714982, 0.52835647],
                           [0.4215251, 0.45017551]],

                          [[0.92814219, 0.96677647],
                           [0.85304703, 0.52351845],
                           [0.19981397, 0.27417313]],

                          [[0.60659855, 0.00533165],
                           [0.10820313, 0.49978937],
                           [0.34144279, 0.94630077]]])
        result = tc.image2vector(image)
        y = np.array([[0.67826139],
                      [0.29380381],
                      [0.90714982],
                      [0.52835647],
                      [0.4215251],
                      [0.45017551],
                      [0.92814219],
                      [0.96677647],
                      [0.85304703],
                      [0.52351845],
                      [0.19981397],
                      [0.27417313],
                      [0.60659855],
                      [0.00533165],
                      [0.10820313],
                      [0.49978937],
                      [0.34144279],
                      [0.94630077]])
        for index in range(len(result)):
            self.assertEqual(result[index], y[index])

    def test_normalizeRows(self):
        x = np.array([
            [0, 3, 4],
            [1, 6, 4]])
        result = tc.normalizeRows(x)
        y = np.array([
            [0.0, 0.6, 0.8],
            [0.13736056394868904, 0.8241633836921342, 0.5494422557947561]])
        for i in range(len(result)):
            for j in range(len(result[i])):
                self.assertEqual(result[i][j], y[i][j])

    def test_softmax(self):
        x = np.array([
            [9, 2, 5, 0, 0],
            [7, 5, 0, 0, 0]])
        result = tc.softmax(x)
        y = np.array([
            [0.9808976649146187, 0.0008944628906901777, 0.017965767417378736, 0.00012105238865619453,
             0.00012105238865619453],
            [0.87867985588699, 0.11891638717077183, 0.0008012523140793875, 0.0008012523140793875,
             0.0008012523140793875]])
        for i in range(len(result)):
            for j in range(len(result[i])):
                self.assertEqual(result[i][j], y[i][j])

    def test_l1(self):
        yhat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        result = tc.l1(yhat, y)
        self.assertEqual(result,1.1)

    def test_l2(self):
        yhat = np.array([.9, 0.2, 0.1, .4, .9])
        y = np.array([1, 0, 0, 1, 1])
        result = tc.l2(yhat, y)
        self.assertEqual(result, 0.43)


if __name__ == '__main__':
    unittest.main()

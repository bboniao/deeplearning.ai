import unittest
import Neural_Networks_and_Deep_Learning.Python_Basics_with_Numpy as tc


class Test(unittest.TestCase):
    def test_basic_sigmiod(self):
        result = tc.basic_sigmiod(3)
        self.assertEquals(result, 0.9525741268224334)


if __name__ == '__main__':
    unittest.main()
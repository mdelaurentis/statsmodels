import unittest
import numpy as np
import statsmodels.api as sm
import logging

class FtestTestCase(unittest.TestCase):

    def test_f_test_with_alphas(self):

        data = np.array([
                [ 0, 1, 5, 3, 9, 8, 7, 8],
                [ 1, 3, 2, 1, 6, 9, 8, 9],
                [ 9, 7, 8, 7, 1, 4, 2, 0],
                [ 1, 2, 3, 4, 5, 6, 7, 8],
                [ 1, 2, 1, 2, 1, 1, 1, 2]],
                        float)

        x = np.array([
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]])

        contrast = np.array([[0, 1]])

        model = sm.GLM(data[0], x)

        alpha = 0.0

        fitted = model.fit()

        alphas = np.array([ 0.0, 1.0, 10.0])

        np.testing.assert_almost_equal(
            [[23.68656716]], 
            fitted.f_test(contrast, smoothing=0).fvalue)
        np.testing.assert_almost_equal(
            [[22.10306407]], 
            fitted.f_test(contrast, smoothing=0.1).fvalue)
        np.testing.assert_almost_equal(
            [[13.8]], 
            fitted.f_test(contrast, smoothing=1.0).fvalue)
        np.testing.assert_almost_equal(
            [[2.90127971]], 
            fitted.f_test(contrast, smoothing=10.0).fvalue)

        np.testing.assert_almost_equal(
            [[[23.68656716]], 
             [[22.10306407]], 
             [[13.8]], 
             [[2.90127971]]],
            fitted.f_test(contrast, smoothing=[0, 0.1, 1.0, 10]).fvalue)
        
if __name__ == '__main__':
    unittest.main()

import unittest

from lecture_2.tests import test_lin_reg, test_mse
from lecture_3.tests import test_metrics


def create_suite():
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTest(test_loader.loadTestsFromTestCase(test_lin_reg.TestLinReg))
    test_suite.addTest(test_loader.loadTestsFromTestCase(test_mse.TestMse))
    test_suite.addTest(
        test_loader.loadTestsFromTestCase(test_metrics.TestConfusionMetrics)
    )

    return test_suite


if __name__ == "__main__":
    suite = create_suite()

    runner = unittest.TextTestRunner()
    runner.run(suite)

import unittest
from calculate_h4_compute_optimization import margin_target
from calculate_h4_per_model_power import model_power


class OptimizationTests(unittest.TestCase):
    def test_margin_targets_and_integer_boundaries(self):
        for q,factor,expected in [(.03211922343749999,1.986670409112242,44),
                                 (.0027108432291666667,1.2883801460247988,6),
                                 (.03666833576388888,2.125588135687874,50)]:
            n=margin_target(q,factor)
            self.assertEqual(n,expected)
            self.assertGreaterEqual(model_power(.1,1.25*q,factor,n),.9)
            self.assertLess(model_power(.1,1.25*q,factor,n-1),.9)

    def test_variance_inflation_does_not_reduce_target(self):
        q=.01525379774305555;factor=1.045139333320385
        self.assertEqual(margin_target(q,factor,1),18)
        self.assertEqual(margin_target(q,factor,1.25),22)

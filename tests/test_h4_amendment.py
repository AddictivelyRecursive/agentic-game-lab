import unittest
import math
from fit_glmm_and_curve import cochran_heterogeneity
from calculate_h4_per_model_power import model_power


class H4AmendmentTests(unittest.TestCase):
    def test_homogeneous_effects_have_zero_Q(self):
        result=cochran_heterogeneity([{'status':'estimated','endpoint_interaction':.1,
                                     'endpoint_standard_error':.02} for _ in range(6)])
        self.assertAlmostEqual(result['Q'],0,places=10)
        self.assertAlmostEqual(result['p_value'],1)
        self.assertEqual(result['I_squared_percent'],0)

    def test_heterogeneity_and_degenerate_errors_are_not_hidden(self):
        effects=[{'status':'estimated','endpoint_interaction':i*.1,'endpoint_standard_error':.02} for i in range(6)]
        result=cochran_heterogeneity(effects)
        self.assertLess(result['p_value'],.001);self.assertGreater(result['I_squared_percent'],90)
        effects[0]['endpoint_standard_error']=0
        self.assertEqual(cochran_heterogeneity(effects)['status'],'not_estimable')

    def test_worst_model_target_integer_boundary(self):
        q=.03666833576388888;factor=2.125588135687874
        self.assertLess(model_power(.1,q,factor,39),.9)
        self.assertGreaterEqual(model_power(.1,q,factor,40),.9)

    def test_extreme_noncentrality_returns_finite_power(self):
        p=model_power(.1,.0027108432291666667,1.2883801460247988,40)
        self.assertTrue(math.isfinite(p));self.assertGreater(p,.99)

"""A reviewed planning variance cannot bypass missing data or excessive fallback."""
import copy
import itertools
import unittest
from experiments.study_common import ROSTER, MODES, N_LEVELS, variance_ready_for_reviewed_power


class ZeroVariancePowerTests(unittest.TestCase):
    def setUp(self):
        self.variance = {'ready_for_power_calculation': False, 'models': list(ROSTER), 'cells': [
            {'model': m, 'reasoning_mode': mode, 'N': n, 'target': 5, 'completed': 5,
             'failed': 0, 'variance': .02, 'fallback_rate': .01}
            for m, mode, n in itertools.product(ROSTER, MODES, N_LEVELS)]}
        self.variance['cells'][0]['variance'] = 0
        self.power = {'confirmed_in_conversation': True, 'approval_reference': 'SYNTHETIC REVIEW',
                      'rationale': 'Synthetic external calculation', 'zero_variance_treatment': {
                          'method': 'common_max_observed_cell_variance', 'planning_variance': .02,
                          'raw_variances_retained': True}}

    def test_reviewed_rule_accepts_zeros_without_mutating_raw_variance(self):
        before = copy.deepcopy(self.variance)
        self.assertTrue(variance_ready_for_reviewed_power(self.variance, self.power))
        self.assertEqual(before, self.variance)

    def test_incomplete_failed_excessive_fallback_and_missing_cells_rejected(self):
        for key, value in [('completed', 4), ('failed', 1), ('fallback_rate', .121), ('variance', None)]:
            variance = copy.deepcopy(self.variance)
            variance['cells'][0][key] = value
            self.assertFalse(variance_ready_for_reviewed_power(variance, self.power))
        self.variance['cells'].pop()
        self.assertFalse(variance_ready_for_reviewed_power(self.variance, self.power))

    def test_unconfirmed_or_understated_planning_variance_rejected(self):
        self.power['confirmed_in_conversation'] = False
        self.assertFalse(variance_ready_for_reviewed_power(self.variance, self.power))
        self.power['confirmed_in_conversation'] = True
        self.power['zero_variance_treatment']['planning_variance'] = .001
        self.assertFalse(variance_ready_for_reviewed_power(self.variance, self.power))

    def test_entirely_uniform_pilot_cannot_supply_positive_working_variance(self):
        for c in self.variance['cells']:
            c['variance'] = 0
        self.power['zero_variance_treatment']['planning_variance'] = 0
        self.assertFalse(variance_ready_for_reviewed_power(self.variance, self.power))

import math
import unittest
import numpy as np
from calculate_h4_power import required_seeds, two_sided_power


class H4PowerTests(unittest.TestCase):
    def test_null_rejection_rate_and_exact_integer_boundary(self):
        self.assertAlmostEqual(two_sided_power(0, .01, 20), .05, places=12)
        for effect in (.05,.1,.15):
            n=required_seeds(effect,.016532)
            self.assertGreaterEqual(two_sided_power(effect,.016532,n),.9)
            if n>1:self.assertLess(two_sided_power(effect,.016532,n-1),.9)

    def test_contrast_is_endpoint_interaction_and_cancels_shared_model_effects(self):
        X=[]; weights=[]; values=[]
        for model in range(6):
            block=[]
            for free in (0,1):
                for n in (2,3,4,5):
                    X.append([1,n,free,n*free])
                    w=(1 if free else -1)*(n-3.5)/10
                    weights.append(w);block.append((w,n))
                    values.append(.2+model*.02-.01*n+.03*free+.1/3*n*free)
            self.assertAlmostEqual(math.fsum(w for w,n in block),0)
            self.assertAlmostEqual(math.fsum(w*n for w,n in block),0)
        np.testing.assert_allclose(np.asarray(weights),3*np.linalg.pinv(np.asarray(X))[3],atol=1e-14)
        self.assertAlmostEqual(np.dot(weights,values),.1,places=12)
        self.assertAlmostEqual(np.dot(weights,weights),.6,places=12)

    def test_smaller_interaction_requires_more_episodes(self):
        counts=[required_seeds(effect,.016532) for effect in (.05,.1,.15)]
        self.assertGreater(counts[0],counts[1]);self.assertGreater(counts[1],counts[2])

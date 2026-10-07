import unittest
from calculate_h4_cluster_power import cluster_power, cluster_required_seeds
from calculate_h4_power import required_seeds


class ClusterPowerTests(unittest.TestCase):
    def test_zero_floor_reproduces_previous_calculation(self):
        for effect in (.05,.1,.15):
            self.assertEqual(cluster_required_seeds(effect,.0025128,0),required_seeds(effect,.0025128))

    def test_unattainable_target_stops_without_unbounded_search(self):
        self.assertIsNone(cluster_required_seeds(.05,.0025,.0022368))
        self.assertLess(cluster_power(.05,.0025,.0022368,10**12),.9)

    def test_increasing_seeds_approaches_nonzero_floor(self):
        powers=[cluster_power(.1,.0025,.0022368,n) for n in (1,10,1000)]
        self.assertLess(powers[0],powers[1]);self.assertLess(powers[1],powers[2])
        self.assertLess(powers[-1],cluster_power(.1,.0025,.0022368))

    def test_finite_threshold_boundary_and_small_cluster_sensitivity(self):
        for df in (None,5):
            n=cluster_required_seeds(.15,.0025,.0001,df)
            self.assertGreaterEqual(cluster_power(.15,.0025,.0001,n,df),.9)
            if n>1:self.assertLess(cluster_power(.15,.0025,.0001,n-1,df),.9)
        self.assertLess(cluster_power(.15,.0025,.0022368,df=5),cluster_power(.15,.0025,.0022368))

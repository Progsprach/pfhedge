import pytest
import torch

from pfhedge.instruments import VarianceGammaStock


class TestVarianceGammaStock:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed):
        torch.manual_seed(seed)

        s = VarianceGammaStock()
        s.simulate(n_paths=1000)
        assert not s.spot.isnan().any()
        assert not s.variance.isnan().any()

    def test_repr(self):
        s = VarianceGammaStock()
        expect = "VarianceGammaStock(\
sigma=1., theta=0., kappa=1., dt=0.0040)"
        assert repr(s) == expect

    def test_simulate_shape(self):
        s = VarianceGammaStock(dt=0.1)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))
        assert s.variance.size() == torch.Size((10, 3))

        s = VarianceGammaStock(dt=0.1)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))
        assert s.variance.size() == torch.Size((10, 4))

import pytest
import torch

from pfhedge.instruments import RoughBergomiStock


class TestRoughBergomiStock:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed):
        torch.manual_seed(seed)

        s = RoughBergomiStock()
        s.simulate(n_paths=1000)

        assert not s.variance.isnan().any()

    def test_repr(self):
        s = RoughBergomiStock()
        expect = "RoughBergomiStock(\
alpha=-0.4000, rho=-0.9000, eta=1.9000, xi=0.0400, dt=0.0040)"
        assert repr(s) == expect

    def test_simulate_shape(self):
        s = RoughBergomiStock(dt=0.1)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))
        assert s.variance.size() == torch.Size((10, 3))

        s = RoughBergomiStock(dt=0.1)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))
        assert s.variance.size() == torch.Size((10, 4))

import pytest
import torch
from torch.testing import assert_close

from pfhedge.cost_functions import ZeroCostFunction,RelativeCostFunction,AbsoluteCostFunction,MixedCostFunction
class TestCostFunctions:

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)
    
    def test_relative_cost(self):
        cost = RelativeCostFunction(cost=0.1)
        spot = torch.tensor([1.0,2.0,10.0,3.0]).unsqueeze(0)
        trades = torch.tensor([1.0,1.0,0.99,1.09]).unsqueeze(0)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=True)
        expect = torch.tensor([0.14])
        assert_close(result,expect)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=False)
        expect = torch.tensor([0.04])
        assert_close(result,expect)
    def test_absolute_cost(self):
        cost = AbsoluteCostFunction(tolerance=0.2, abscost=0.1)
        spot = torch.tensor([1.0,2.0,10.0,3.0]).unsqueeze(0)
        trades = torch.tensor([1.0,1.0,0.99,1.09]).unsqueeze(0)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=True)
        expect = torch.tensor([0.25])
        assert_close(result,expect)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=False)
        expect = torch.tensor([0.15])
        assert_close(result,expect)
    def test_mixed_cost(self):
        cost = MixedCostFunction(cost=0.1,tolerance=0.2, abscost=0.1)
        spot = torch.tensor([1.0,2.0,10.0,3.0]).unsqueeze(0)
        trades = torch.tensor([1.0,1.0,0.99,1.09]).unsqueeze(0)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=True)
        expect = torch.tensor([0.39])
        assert_close(result,expect)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=False)
        expect = torch.tensor([0.19])
        assert_close(result,expect)
    def test_zero_cost(self):
        cost = ZeroCostFunction()
        spot = torch.tensor([1.0,2.0,10.0,3.0]).unsqueeze(0)
        trades = torch.tensor([1.0,1.0,0.99,1.09]).unsqueeze(0)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=True)
        expect = torch.tensor([0.0])
        assert_close(result,expect)
        result = cost.apply(trades=trades,spot=spot,apply_first_cost=False)
        expect = torch.tensor([0.0])
        assert_close(result,expect)

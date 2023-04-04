from typing import Optional
from typing import Union

import pytest
import torch
from torch.testing import assert_close
from pfhedge.instruments import BrownianStock,EuropeanOption
from pfhedge.instruments.clauses import add_cap_clause,add_knockin_clause,add_knockout_clause
class TestClauses:

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_knockout(self, device: Optional[Union[str, torch.device]] = "cpu"):
        stock = BrownianStock()
        derivative = EuropeanOption(stock).to(device)
        add_knockout_clause(der=derivative,barrier=2.5,rebate=0.5)
        stock.register_buffer(
            "spot",
            torch.tensor([[1.0, 3.0, 2.0], [1.0, 1.5, 2.0], [3.0, 1.5, 0.5]]).to(device))
        result = derivative.payoff()
        expect = torch.tensor([0.5, 1.0, 0.5]).to(device)
        assert_close(result,expect)

    @pytest.mark.gpu
    def test_knockout_gpu(self):
        self.test_knockout(device="cuda")

    def test_knockin(self, device: Optional[Union[str, torch.device]] = "cpu"):
        stock = BrownianStock()
        derivative = EuropeanOption(stock).to(device)
        add_knockin_clause(der=derivative,up=False, barrier=0.5,rebate=0.5)
        stock.register_buffer(
            "spot",
            torch.tensor([[1.0, 0.1, 4.0], [1.0, 1.5, 4.0], [1.0, 0.1, 1.0]]).to(device))
        result = derivative.payoff()
        expect = torch.tensor([3.0, 0.5, 0.0]).to(device)
        assert_close(result,expect)

    @pytest.mark.gpu
    def test_knockin_gpu(self):
        self.test_knockin(device="cuda")
    
    def test_barrier(self, device: Optional[Union[str, torch.device]] = "cpu"):
        stock = BrownianStock()
        derivative = EuropeanOption(stock).to(device)
        add_cap_clause(der=derivative, barrier=2.0)
        stock.register_buffer(
            "spot",
            torch.tensor([[1.0, 1.0, 4.0], [1.0, 1.0, 1.5], [1.0, 1.0, 1.0]]).to(device))
        result = derivative.payoff()
        expect = torch.tensor([1.0, 0.5, 0.0]).to(device)
        assert_close(result,expect)

    @pytest.mark.gpu
    def test_barrier_gpu(self):
        self.test_barrier(device="cuda")
    

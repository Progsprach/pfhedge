from abc import ABC, abstractmethod
from torch import Tensor,zeros
class CostFunction(ABC):
    @abstractmethod
    def apply(self,trades: Tensor, spot: Tensor, apply_first_cost: bool = True)->Tensor:
        return
class ZeroCostFunction(CostFunction):
    def apply(self, trades: Tensor, spot: Tensor, apply_first_cost: bool = True) -> Tensor:
        return zeros(trades.shape[0])
class RelativeCostFunction(CostFunction):
    def __init__(self,cost : float) -> None:
        super().__init__()
        self.cost = cost
    def apply(self, trades: Tensor, spot: Tensor, apply_first_cost: bool = True) -> Tensor:
        costs = self.cost*spot[...,1:]*trades.diff(dim=-1).abs()
        total = costs.sum(-1)
        if apply_first_cost:
            total += spot[...,0]*trades[...,0].abs()*self.cost
        return total

class AbsoluteCostFunction(CostFunction):
    def __init__(self,tolerance: float, abscost : float) -> None:
        super().__init__()
        self.tolerance = tolerance
        self.factor = abscost/tolerance
    def apply(self, trades: Tensor, spot: Tensor, apply_first_cost: bool = True) -> Tensor:
        costs = (spot[...,1:]*trades.diff(dim=-1).abs()).clamp(max=self.tolerance)*self.factor
        total = costs.sum(-1)
        if apply_first_cost:
            total += (spot[...,0]*trades[...,0].abs()).clamp(max=self.tolerance)*self.factor
        return total

class MixedCostFunction(CostFunction):
    def __init__(self, cost: float, tolerance: float, abscost : float) -> None:
        super().__init__()
        self.cost = cost
        self.tolerance = tolerance
        self.factor = abscost/tolerance
    def apply(self, trades: Tensor, spot: Tensor, apply_first_cost: bool = True) -> Tensor:
        costs = (spot[...,1:]*trades.diff(dim=-1).abs()).clamp(max=self.tolerance)*self.factor
        costs += self.cost*spot[...,1:]*trades.diff(dim=-1).abs()
        total = costs.sum(-1)
        if apply_first_cost:
            total += spot[...,0]*trades[...,0].abs()*self.cost
            total += (spot[...,0]*trades[...,0].abs()).clamp(max=self.tolerance)*self.factor
        return total
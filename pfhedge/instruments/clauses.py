import torch
from . import BaseDerivative
def add_cap_clause(der, barrier):
    strike = der.strike
    if der.call:
        def cap_clause(derivative: BaseDerivative, payoff: torch.Tensor):
            max_spot = derivative.ul().spot.max(-1).values
            capped_payoff = torch.full_like(payoff, barrier - strike)
            return torch.where(max_spot < barrier, payoff, capped_payoff)
    else:        
        def cap_clause(derivative: BaseDerivative, payoff: torch.Tensor):
            min_spot = derivative.ul().spot.min(-1).values
            capped_payoff = torch.full_like(payoff, strike - barrier)
            return torch.where(min_spot > barrier, payoff, capped_payoff)

    der.add_clause('cap',cap_clause)
def add_knockout_clause(der, barrier, up=True, rebate=0.0):
    if up:
        name = 'knockout_up'
        def knockout(derivative: BaseDerivative, payoff: torch.Tensor) -> torch.Tensor:
            max = derivative.ul().spot.max(-1).values
            return payoff.where(max <= barrier, torch.full_like(max,rebate))
    else:
        name = 'knockout_down'
        def knockout(derivative: BaseDerivative, payoff: torch.Tensor) -> torch.Tensor:
            min = derivative.ul().spot.min(-1).values
            return payoff.where(min >= barrier, torch.full_like(min,rebate))
    der.add_clause(name, knockout)
def add_knockin_clause(der, barrier, up=True, rebate=0.0):
    if up:
        name = 'knockin_up'
        def knockin(derivative: BaseDerivative, payoff: torch.Tensor) -> torch.Tensor:
            max = derivative.ul().spot.max(-1).values
            return payoff.where(max > barrier, torch.full_like(max,rebate))
    else:
        name = 'knockin_down'
        def knockin(derivative: BaseDerivative, payoff: torch.Tensor) -> torch.Tensor:
            min = derivative.ul().spot.min(-1).values
            return payoff.where(min < barrier, torch.full_like(min,rebate))
    der.add_clause(name, knockin)
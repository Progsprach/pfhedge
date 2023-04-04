from typing import List,Union
from pfhedge.instruments import BaseDerivative, EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption, MultiDerivative
from pfhedge.instruments import VarianceSwap, AsianOption, EuropeanForwardStartOption
from pfhedge.nn import BlackScholes
from pfhedge.cost_functions import CostFunction
def black_scholes_implemented(derivative: BaseDerivative):
    if derivative.__class__ == MultiDerivative:
        impl = [black_scholes_implemented(der) for der in derivative.derivatives]
        return not False in impl
    if derivative.__class__ in [AmericanBinaryOption,LookbackOption]:
        return derivative.call
    lst = [EuropeanOption,EuropeanBinaryOption]
    return derivative.__class__ in lst
def prepare_hedges(transaction_costs: Union[float, List[float]],*args):
    num_hedges = len(args)
    if num_hedges==0:
        raise ValueError("No hedging instruments given")
    transaction_costs = (transaction_costs,) * num_hedges if isinstance(transaction_costs, float) else transaction_costs
    hedge = []
    if len(transaction_costs) != num_hedges:
        raise ValueError("Mismatched transaction cost list")
    for i,der in enumerate(args):
        if der.is_listed:
            der.cost = transaction_costs[i]
        else:
            list_derivative(der, transaction_costs[i])
        hedge.append(der)
    return hedge
def prepare_features(derivative: BaseDerivative, prev_hedge: bool):
    features = None
    if black_scholes_implemented(derivative):
        features = BlackScholes(derivative).inputs()
    if derivative.__class__ == AsianOption:
        if derivative.geom:
            features = ["log_moneyness", "mean_log_moneyness", "expiry_time", "volatility"]
        else:
            features = ["log_moneyness", "log_mean_moneyness", "expiry_time", "volatility"]
    if derivative.__class__ == EuropeanForwardStartOption:
        features = ["log_moneyness", "expiry_time", "volatility"]
    if features is None:
        features= ['underlier_spot','volatility']
    if prev_hedge:
        features.append("prev_hedge")
    return features
        


def list_derivative(derivative: BaseDerivative,cost: CostFunction):
    pricer = None
    if derivative.__class__ == EuropeanOption or derivative.__class__ == EuropeanBinaryOption:
        pricer = lambda derivative: BlackScholes(derivative).price(
        log_moneyness=derivative.log_moneyness(),
        time_to_maturity=derivative.time_to_maturity(),
        volatility=derivative.ul().volatility)
    if derivative.__class__ == AmericanBinaryOption or derivative.__class__ == LookbackOption:
        pricer = lambda derivative: BlackScholes(derivative).price(
        log_moneyness=derivative.log_moneyness(),
        max_log_moneyness = derivative.max_log_moneyness(),
        time_to_maturity=derivative.time_to_maturity(),
        volatility=derivative.ul().volatility)        
    if derivative.__class__ == VarianceSwap:
        pricer = lambda varswap: varswap.ul().variance - varswap.strike
    if pricer==None:
        raise TypeError("Tried to list unsupported derivative type, must have Black-Scholes support or be variance swap")
    derivative.list(pricer,cost)
    
def make_linear_volatility(scale=0.1, bias=0.1):
    def sigma_fn(time,spot):
        return scale*spot+bias
    return sigma_fn
def make_cev_volatility(sigma=0.2, gamma=0.9):
    def sigma_fn(time,spot):
        return sigma*spot**(gamma-1)
    return sigma_fn
def make_time_volatility(sigma=0.2, alpha=1):
    def sigma_fn(time,spot):
        return sigma*(alpha*time).exp()
    return sigma_fn
def make_relative_cost(cost):
    def cost_fn(trade):
        return cost*trade
    return cost_fn
def make_absolute_cost(abscost, tolerance):
    def cost_fn(trade):
        return trade.clamp(0,tolerance)*abscost
    return cost_fn
def make_mixed_cost(cost,abscost,tolerance):
    def cost_fn(trade):
        return (cost*trade)+(trade.clamp(0,tolerance)*abscost)
    return cost_fn
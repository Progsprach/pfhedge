from typing import List
from collections.abc import Callable
from torch.nn import Module
from pfhedge.nn import (
    MultiLayerPerceptron,
    HedgeLoss,
    EntropicRiskMeasure,
    EntropicLoss,
    ExpectedShortfall,
)
from pfhedge.instruments import BaseInstrument, BasePrimary, BaseDerivative
from pfhedge.instruments import (
    BrownianStock,
    HestonStock,
    MertonJumpStock,
    RoughBergomiStock,
    LocalVolatilityStock,
    VarianceGammaStock,
    InverseGaussianStock,
    HestonJumpStock,
)
from pfhedge.instruments import (
    EuropeanOption,
    EuropeanBinaryOption,
    LookbackOption,
    AmericanBinaryOption,
    VarianceSwap,
    AsianOption,
    EuropeanForwardStartOption,
    FloatingAsianOption,
)
from pfhedge.instruments import MultiDerivative
from utils import list_derivative
from utils import make_linear_volatility, make_cev_volatility, make_time_volatility
from models import MultiLayerHybrid, NoTransactionBandNet
from quantum_circuits import (
    QuantumCircuit,
    SimpleQuantumCircuit,
    ReuploadingQuantumCircuit,
)

from pfhedge.instruments.clauses import add_cap_clause, add_knockin_clause, add_knockout_clause
from cost_functions import CostFunction,ZeroCostFunction,RelativeCostFunction,AbsoluteCostFunction,MixedCostFunction

def dict_without_keys(dictionary: dict, *args: tuple[str]):
    copy = dict()
    for key in dictionary.keys():
        if not key in args:
            copy[key] = dictionary[key]
    return copy


def make_variance_function(config: dict) -> Callable:
    options = {"linear": make_linear_volatility, "CEV": make_cev_volatility, 'time': make_time_volatility}
    variance_type = options[config.get("type", "CEV")]
    cfg = dict_without_keys(config, "type")
    return variance_type(**cfg)

def make_cost(config: dict) -> CostFunction:
    options = {"ZeroCostFunction": ZeroCostFunction,
               "RelativeCostFunction": RelativeCostFunction,
               "AbsoluteCostFunction": AbsoluteCostFunction,
               "MixedCostFunction": MixedCostFunction,
               }
    cost_type = options[config.get("type","ZeroCostFunction")]
    cfg = dict_without_keys(config,"type")
    return cost_type(**cfg)


def make_underlier(config: dict) -> BasePrimary:
    options = {
        "BrownianStock": BrownianStock,
        "HestonStock": HestonStock,
        "MertonJumpStock": MertonJumpStock,
        "RoughBergomiStock": RoughBergomiStock,
        "LocalVolatilityStock": LocalVolatilityStock,
        "VarianceGammaStock": VarianceGammaStock,
        "InverseGaussianStock": InverseGaussianStock,
        "HestonJumpStock": HestonJumpStock,
    }
    underlier_type = options[config.get("type", "BrownianStock")]
    cost = make_cost(config.get("cost",{}))
    if underlier_type == LocalVolatilityStock:
        sigma_fn = make_variance_function(config.get("sigma_fn", {}))
        cfg = dict_without_keys(config, "type", "cost", "sigma_fn")
        return underlier_type(sigma_fn=sigma_fn, cost=cost, **cfg)
    cfg = dict_without_keys(config, "type", "cost")
    return underlier_type(cost=cost, **cfg)


def add_clause(config: dict, derivative: BaseDerivative):
    options = {
        "cap": add_cap_clause,
        "knockout": add_knockout_clause,
        "knockin": add_knockin_clause,
    }
    clause_type = options.get(config.get("type", "none"), None)
    if clause_type is None:
        return
    cfg = dict_without_keys(config, "type")
    clause_type(derivative, **cfg)


def make_derivative(config: dict, underlier: BasePrimary) -> BaseDerivative:
    options = {
        "EuropeanOption": EuropeanOption,
        "EuropeanBinaryOption": EuropeanBinaryOption,
        "AmericanBinaryOption": AmericanBinaryOption,
        "LookbackOption": LookbackOption,
        "VarianceSwap": VarianceSwap,
        "AsianOption": AsianOption,
        "EuropeanForwardStartOption": EuropeanForwardStartOption,
        "FloatingAsianOption": FloatingAsianOption,
        "MultiDerivative": MultiDerivative,
    }
    derivative_type = options[config.get("type", "EuropeanOption")]
    if derivative_type == MultiDerivative:
        derivatives_cfg = config.get("derivatives", [])
        derivatives = [make_derivative(entry, underlier) for entry in derivatives_cfg]
        return MultiDerivative(underlier, derivatives)
    cfg = dict_without_keys(config, "type", "cost", "clauses")
    derivative = derivative_type(underlier=underlier, **cfg)
    for clause in config.get("clauses", []):
        add_clause(clause, derivative)
    return derivative


def make_hedge(config: dict, underlier: BasePrimary) -> List[BaseInstrument]:
    hedge = []
    if config.get("underlier", True):
        hedge.append(underlier)
    derivative_list = config.get("derivatives", [])
    for entry in derivative_list:
        der = make_derivative(entry, underlier)
        cost = make_cost(entry.get("cost",{}))
        list_derivative(der, cost=cost)
        hedge.append(der)
    return hedge


def make_circuit(config: dict) -> QuantumCircuit:
    options = {
        "SimpleQuantumCircuit": SimpleQuantumCircuit,
        "ReuploadingQuantumCircuit": ReuploadingQuantumCircuit,
    }
    model_type = options[config.get("type", "SimpleQuantumCircuit")]
    cfg = dict_without_keys(config, "type")
    return model_type(**cfg)

def make_model(config: dict, n_hedges: int, derivative: BaseDerivative) -> Module:
    options = {
        "MultiLayerPerceptron": MultiLayerPerceptron,
        "MultiLayerHybrid": MultiLayerHybrid,
    }
    model_type = options[config.get("type", "MultiLayerPerceptron")]
    NTB = config.get("NTB", True)
    if NTB:
        if n_hedges != 1:
            raise ValueError(
                "No Transaction Band not supported for more than one hedging instrument."
            )
        else:
            n_hedges = 2
    if model_type == MultiLayerPerceptron:
        cfg = dict_without_keys(config, "type", "NTB")
        model = model_type(out_features=n_hedges, **cfg)
    if model_type == MultiLayerHybrid:
        circuit_config = config.get("circuit", {})
        circuit = make_circuit(circuit_config)
        cfg = dict_without_keys(config, "type", "NTB", "circuit")
        model = MultiLayerHybrid(quantum=circuit, out_features=n_hedges, **cfg)
    if NTB:
        model = NoTransactionBandNet(derivative, model)
    return model


def make_criterion(config: dict) -> HedgeLoss:
    options = {
        "ExpectedShortfall": ExpectedShortfall,
        "EntropicRiskMeasure": EntropicRiskMeasure,
        "EntropicLoss": EntropicLoss,
    }
    loss_type = options[config.get("type", "ExpectedShortfall")]
    cfg = dict_without_keys(config, "type")
    return loss_type(**cfg)

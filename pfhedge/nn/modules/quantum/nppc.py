from typing import List
from copy import deepcopy

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Identity

from .jaxlayer import JaxLayer
from .circuits import QuantumCircuit

class NoPreprocessingCircuit(Sequential):
    """Variational quantum circuit with classical postprocessing layer."""
    def __init__(self,quantum: QuantumCircuit, out_features = 1, out_activation: Module = Identity()):
        layers: List[Module] = []
        layers.append(JaxLayer(quantum))
        layers.append(Linear(quantum.n_outputs,out_features))
        layers.append(deepcopy(out_activation))
        super().__init__(*layers)
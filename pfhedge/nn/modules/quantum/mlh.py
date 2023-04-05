from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from .circuits import QuantumCircuit


from .jaxlayer import JaxLayer

class MultiLayerHybrid(Sequential):
    """Multi layer perceptron with additional variational quantum circuit layer."""
    def __init__(
        self,
        quantum: QuantumCircuit,
        #quantum_nr: int = -1, 
        in_features: Optional[int] = None,
        out_features: int = 1,
        n_layers: int = 3,
        n_units: Union[int, Sequence[int]] = 16,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ):
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units
        #if quantum_nr<0:
        #    quantum_nr = n_layers+quantum_nr
        #if quantum_nr<0 or quantum_nr>n_layers:
        #    raise ValueError("Invalid position for quantum layer")

        layers: List[Module] = []
        for i in range(n_layers):
            if i == 0:
                if in_features is None:
                    layers.append(LazyLinear(n_units[0]))
                else:
                    layers.append(Linear(in_features, n_units[0]))
            else:
                layers.append(Linear(n_units[i - 1], n_units[i]))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units[-1], quantum.n_inputs))
        layers.append(deepcopy(activation))
        layers.append(JaxLayer(quantum))
        layers.append(Linear(quantum.n_outputs,out_features))
        layers.append(deepcopy(out_activation))

        super().__init__(*layers)
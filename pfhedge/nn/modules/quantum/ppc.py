from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Identity
from .mlh import MultiLayerHybrid
from quantum_circuits import QuantumCircuit

class PreprocessingCircuit(MultiLayerHybrid):
    """Variational quantum circuit with classical pre- and postprocessing layers."""
    def __init__(self,
        quantum: QuantumCircuit,
        in_features: int,
        out_features: int = 1,
        activation: Module = ReLU(),
        out_activation: Module = Identity()):

        super().__init__(quantum,in_features,out_features,0,[in_features],activation,out_activation)
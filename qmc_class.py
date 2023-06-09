from copy import deepcopy
from typing import List
import os

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import Module,Linear,Identity,ReLU,Sequential
from torch import reshape
from qiskit_aqt_provider import AQTProvider
from qiskit.primitives import BackendEstimator

def make_qiskit_layer(num_qubits,layers):
    provider = AQTProvider(os.environ['AQT_TOKEN'])
    backend = provider.get_backend("aqt_qasm_simulator_noise_1")
    estimator = BackendEstimator(backend)
    feature_map = ZZFeatureMap(feature_dimension=num_qubits)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=layers)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        estimator=estimator,
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    return TorchConnector(qnn)

#class QiskitPreprocessing(Sequential):
#    def __init__(self,n_qubits, n_layers, in_features, out_features = 1, activation: Module = ReLU(), out_activation: Module = Identity()):    
#        layers: List[Module] = []
#        layers.append(Linear(in_features=in_features,out_features=n_qubits))
#        layers.append(deepcopy(activation))
#        layers.append(make_qiskit_layer(n_qubits,n_layers))
#        layers.append(Linear(1,out_features))
#        layers.append(deepcopy(out_activation))
#        super().__init__(*layers)

class QiskitPreprocessing(Module):
    def __init__(self,n_qubits, n_layers, in_features, out_features = 1, activation: Module = ReLU(), out_activation: Module = Identity()):
        super().__init__()
        self.linear1 = (Linear(in_features=in_features,out_features=n_qubits))
        self.activation = (deepcopy(activation))
        self.qiskit = (make_qiskit_layer(n_qubits,n_layers))
        self.linear2 = (Linear(1,out_features))
        self.out_activation = (deepcopy(out_activation))
    def forward(self,input):
        input = self.linear1(input)
        input = self.activation(input)
        shp = input.shape
        input = reshape(input, (shp[0]*shp[1],shp[2]))
        input = self.qiskit(input)
        input = reshape(input, (shp[0],shp[1],-1))
        input = self.linear2(input)
        input = self.out_activation(input)
        return input


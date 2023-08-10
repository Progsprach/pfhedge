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
from qiskit_aqt_provider.primitives.estimator import AQTEstimator
from qiskit.algorithms.gradients import ParamShiftEstimatorGradient


def make_qiskit_layer(num_qubits,layers):
    token = ''
    provider = AQTProvider(os.environ['AQT_TOKEN'])
    backend = provider.get_backend("offline_simulator_noise")
    estimator = AQTEstimator(backend)
    grad_estimator = ParamShiftEstimatorGradient(estimator= estimator)
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
        input_gradients=True,
        gradient=grad_estimator
    )
    return TorchConnector(qnn)


import torch


n_qubits = 4
n_layers = 10
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = make_qiskit_layer(n_qubits, n_layers)
        self.fc2 = torch.nn.Linear(1, 1)

    def forward(self, X):
        X =  torch.relu(self.fc1(X))
        return self.fc2(X)

# Data, 2000 worked independent of n_qubits/n_layers

# 100 failed, 90 failed, 80 failed, 60 failed, 56 did not work
# 50 worked, 51 worked, 55 workes
n = 15
print(n*(2*n_qubits*n_layers+1))
x = torch.rand(n, n_qubits)
y = torch.sin(x[:, 0]).view(-1, 1)
print(y.shape)

model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

history = []
for epoch in range(2):
    y_predict = model(x)
    loss = criterion(y_predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(loss)
    if epoch % 10 == 0:
        pass
    print(f'{epoch}\t{loss}')


y = model(x).detach().numpy()


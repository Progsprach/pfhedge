import os
from torch.nn import Linear, ReLU, Identity, Module
from aqt_layer import QuantumLayer
from qiskit_aqt_provider import AQTProvider


class AQT_NN(Module):
    def __init__(self, n_qubits, n_layers, in_features, n_averages, backend, shots=200):
        super().__init__()
        self.linear1 = Linear(in_features=in_features, out_features=n_qubits)
        self.quantum = QuantumLayer(n_qubits, n_layers, backend, n_averages, shots)
        self.linear2 = Linear(in_features=n_qubits, out_features=1)
        self.activation = ReLU()
        self.out_activation = Identity()

    def forward(self, x):
        y = self.activation(self.linear1(x))
        y = self.quantum(y)        
        y = self.linear2(y)
        y = self.out_activation(y)
        return y
    
    
if __name__ == '__main__':
    import torch
    data = torch.zeros((10, 8, 3))
    nn = AQT_NN(5, 5, 3)
    out = nn(data)
    print(out)
    
    
    

        
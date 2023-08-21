
import os
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit_aqt_provider import AQTProvider


provider = AQTProvider(os.environ["AQT_TOKEN"])
backend = provider.get_backend("offline_simulator_no_noise")


def create_quantum_circuit(x, params):
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        qc.rx(x[i], i)
    n_layers = len(params)
    for angles in params:
        for i, angle in enumerate(angles):
            qc.rx(angle, i)
        for i in range(n_qubits):
            qc.cnot(i, (i+1)%n_qubits)
    for i in range(n_qubits):
        qc.measure(i, i)
    return qc


def get_nth_bit(num, n):
    mask = 2**n
    return (num & mask)//mask


def get_expvals(n_qubits, shots, counts):
    expvals = np.zeros(n_qubits)
    for label, count in counts.items():
        label = int(label, 2)
        for i in range(n_qubits):
            status = get_nth_bit(label, i)
            expvals[i] += (-1)**status*count
    expvals /= shots
    return expvals


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, backend=backend, n_averages=1, shots=200):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.n_averages = n_averages
        self.shots = shots
        params = torch.zeros((n_layers, n_qubits))
        self.params = torch.nn.Parameter(params)
        # TODO: Choose normal initialization
        # nn.init.zeros_(self.params)        
        nn.init.normal_(self.params, mean=0., std=1.)
        
    def forward(self, X):
        params = self.params.detach().numpy()
        n1, n2, n3 = X.shape
        out = np.zeros((n1, n2, n3))
        for i in range(n1):
            for j in range(n2):
                x = X[i, j, :].detach().numpy()
                expvals = np.zeros(self.n_qubits)
                for _ in range(self.n_averages):
                    qc = create_quantum_circuit(x, params)
                    counts = execute(qc, self.backend, shots=self.shots, with_progress_bar=False).result().get_counts()
                    expvals += get_expvals(self.n_qubits, self.shots, counts)
                expvals /= self.n_averages
                out[i, j, :] = get_expvals(self.n_qubits, self.shots, counts)
        return torch.Tensor(out)
    
    
if __name__ == '__main__':
    n_qubits = 5
    n_layers = 1
    layer = QuantumLayer(n_qubits, n_layers)
    
    x = np.pi*torch.tensor([[[1., 2., 1., 2., .2]]])
    out = layer(x)
    print(out.shape)
    
    




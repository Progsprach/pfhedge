"""Provides the quantum circuits via Pennylane."""
import pennylane as qml
from abc import ABC
import jax
class QuantumCircuit(ABC):
    """Parent class for all quantum circuits"""
    def __init__(self) -> None:
        super().__init__()
        self.n_inputs = 0
        self.n_outputs = 0
        self.weight_shape = ()

class SimpleQuantumCircuit(QuantumCircuit):
    """Basic quantum circuit using angle embedding and alternating rotation and entanglement layers."""
    def __init__(self, n_qubits=3, n_layers=4, n_measurements = 0):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        self.weight_shape = (n_layers,n_qubits)
        self.qnode = self._make_qnode(n_qubits,dev,n_measurements)
    def _make_qnode(self, n_qubits, dev, n_measurements):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
        return qnode
class ReuploadingQuantumCircuit(QuantumCircuit):
    """Quantum circuit which repeatedly uses angle embedding in between rotation and entanglement layers."""
    def __init__(self, n_qubits=4, n_uploads=3, n_layers=2, n_measurements = 0):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        self.weight_shape = (n_uploads*n_layers,n_qubits)
        self.qnode = self._make_qnode(n_qubits,dev,n_uploads,n_layers,n_measurements)
    def _make_qnode(self, n_qubits, dev,n_uploads,n_layers, n_measurements):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs, weights):
            for i in range(n_uploads):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.BasicEntanglerLayers(weights[i*n_layers:(i+1)*n_layers,...], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
        return qnode

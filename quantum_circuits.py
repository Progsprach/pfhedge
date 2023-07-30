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
    def __init__(self, n_qubits=3, n_layers=4, n_measurements = 0, rotation_matrix=None):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        self.weight_shape = (n_layers,n_qubits)
        if rotation_matrix:
            print('DEBUG: Using rotation matrix')
            self.qnode = self._make_qnode_rotation_control(n_qubits, dev, n_measurements, rotation_matrix)
        else:
            print('DEBUG: Not using rotation matrix')
            self.qnode = self._make_qnode(n_qubits,dev,n_measurements)

    def _make_qnode(self, n_qubits, dev, n_measurements):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
        return qnode
    
    def _make_qnode_rotation_control(self, n_qubits, dev, n_measurements, rotation_matrix):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs, weights):
            n1 = len(inputs)
            n2 = len(weights[0])
            n3 = len(rotation_matrix[0])
            if n1 != n2 or n1 != n3 or n2 != n3:
                raise Exception('Amount of layers not the same in all parameters')

            m2 = len(weights)
            m3 = len(rotation_matrix)
            if m2 != m3:
                raise Exception('Amount of qubits not the same in all parameters')

            n_qubits = len(inputs)
            n_layers = len(weights)

            qml.BasisState([0]*n_qubits, wires=range(n_qubits))
            qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation='X')
            for i_layer, (angle_vector, rotation_vector) in enumerate(zip(weights, rotation_matrix)):
                for j_qubit, (angle, rotation) in enumerate(zip(angle_vector, rotation_vector)):
                    if rotation == 'y':
                        operation = qml.RY
                    else:
                        operation = qml.RX
                    operation(angle, j_qubit)
                for k in range(n_qubits-1):
                    qml.CNOT([k, k+1])
                if n_qubits > 2:
                    qml.CNOT([n_qubits-1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
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
    

if __name__ == '__main__':
    n_qubits = 5
    n_layers = 5
    rotation_matrix = [['x', 'y', 'x', 'y', 'x'],
                       ['y', 'x', 'y', 'x', 'y'],
                       ['x', 'y', 'x', 'y', 'x'],
                       ['y', 'x', 'y', 'x', 'y'],
                       ['x', 'y', 'x', 'y', 'x']]
    import numpy as np
    quantum = SimpleQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers, rotation_matrix=rotation_matrix)
    inputs = np.random.randn(n_layers)
    weights = np.random.randn(n_layers, n_qubits)
    drawing = qml.draw(quantum.qnode)(inputs, weights)
    print(drawing)


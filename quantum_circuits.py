import pennylane as qml
from abc import ABC
from AQT_class import AQTDevice
class QuantumCircuit(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.n_inputs = 0
        self.n_outputs = 0
        self.weight_shapes = ()
        self.qnode = None
    def layer(self):
        return qml.qnn.TorchLayer(self.qnode, self.weight_shapes)


class SimpleQuantumCircuit(QuantumCircuit):
    def __init__(self, n_qubits=3, n_layers=4, n_measurements = 0):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        #dev = AQTDevice(wires=n_qubits, shots=40)
        dev = qml.device('default.mixed', wires=n_qubits, shots=50)
        #dev = qml.device('default.qubit.jax', wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = self._make_qnode(n_qubits,dev,n_measurements)
    def _make_qnode(self, n_qubits, dev, n_measurements):
        @qml.qnode(dev, interface='torch', diff_method='best')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
        return qnode
    

class ReuploadingQuantumCircuit(QuantumCircuit):
    def __init__(self, n_qubits=4, n_uploads=3, n_layers=2, n_measurements = 0):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        dev = qml.device('default.mixed', wires=n_qubits)
        self.weight_shapes = {"weights": (n_uploads*n_layers,n_qubits)} 
        self.qnode = self._make_qnode(n_qubits,dev,n_uploads,n_layers,n_measurements)
    def _make_qnode(self, n_qubits, dev,n_uploads,n_layers, n_measurements):
        @qml.qnode(dev, interface='torch', shots=50, diff_method='best')
        def qnode(inputs, weights):
            for i in range(n_uploads):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.BasicEntanglerLayers(weights[i*n_layers:(i+1)*n_layers,...], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
        return qnode

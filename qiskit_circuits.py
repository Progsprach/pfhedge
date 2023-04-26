import qiskit
from qiskit import QuantumCircuit
def make_basic_circuit(n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)
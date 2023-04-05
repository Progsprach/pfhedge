import pytest
import torch
from pfhedge.nn import QuantumCircuit, SimpleQuantumCircuit, ReuploadingQuantumCircuit
import jax.numpy as jnp

def test_simple_quantum_circuit():
    circuit = SimpleQuantumCircuit(n_qubits=2, n_layers=1, n_measurements=0)
    assert circuit.n_inputs == 2
    assert circuit.n_outputs == 2
    assert jnp.allclose(circuit.qnode(jnp.zeros((2,)),jnp.zeros((1,2))),jnp.ones((2,)))

def test_reuploading_quantum_circuit():
    circuit = ReuploadingQuantumCircuit(n_qubits=3, n_uploads=2,n_layers=1,n_measurements=2)
    assert circuit.n_inputs == 3
    assert circuit.n_outputs == 2
    assert jnp.allclose(circuit.qnode(jnp.zeros((3,)),jnp.zeros((2,1,3))),jnp.ones((2,)))
    
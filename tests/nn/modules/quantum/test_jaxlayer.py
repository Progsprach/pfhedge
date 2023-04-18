from typing import Optional
from typing import Union

import pytest
import torch

from pfhedge.nn import JaxLayer
from pfhedge.nn import SimpleQuantumCircuit

class TestJaxLayer:
    """
    pfhedge.nn.JaxLayer
    """

    @pytest.mark.parametrize("n_qubits", [4,5])
    @pytest.mark.parametrize("n_layers", [1,2])
    @pytest.mark.parametrize("n_measurements", [2,0])
    def test_shape(self, n_qubits: int, n_layers: int, n_measurements: int, device: Optional[Union[str, torch.device]] = "cpu"):
        circuit = SimpleQuantumCircuit(n_qubits,n_layers,n_measurements)
        N = 11
        M_1 = 13
        M_2 = 14


        n_out = n_qubits if n_measurements==0 else n_measurements
        m = JaxLayer(circuit).to(device)

        input = torch.zeros((N, n_qubits)).to(device)
        assert m(input).size() == torch.Size((N, n_out))

        input = torch.zeros((N, M_1, n_qubits)).to(device)
        assert m(input).size() == torch.Size((N, M_1, n_out))

        input = torch.zeros((N, M_1, M_2, n_qubits)).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, n_out))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_qubits", [4,5,6])
    @pytest.mark.parametrize("n_layers", [1,2])
    @pytest.mark.parametrize("n_measurements", [2,0])
    def test_shape_gpu(self, n_qubits: int, n_layers: int, n_measurements: int):
        self.test_shape(n_qubits, n_layers, n_measurements, device="cuda")

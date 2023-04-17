from typing import Optional
from typing import Union

import pytest
import torch

from pfhedge.nn import NoPreprocessingCircuit
from pfhedge.nn import SimpleQuantumCircuit

class TestNoPreprocessingCircuit:
    """
    pfhedge.nn.NoPreprocessingCircuit
    """

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        circuit = SimpleQuantumCircuit(3,4,2)
        N = 11
        M_1 = 13
        M_2 = 14
        H_out = 15

        input = torch.zeros((N, 3)).to(device)
        m = NoPreprocessingCircuit(quantum=circuit, out_features=H_out).to(device)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, 3)).to(device)
        m = NoPreprocessingCircuit(quantum=circuit, out_features=H_out).to(device)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, 3)).to(device)
        m = NoPreprocessingCircuit(quantum=circuit, out_features=H_out).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")

from typing import Optional
from typing import Union

import pytest
import torch

from pfhedge.nn import MultiLayerHybrid
from pfhedge.nn import SimpleQuantumCircuit

class TestMultiLayerHybrid:
    """
    pfhedge.nn.MultiLayerHybrid
    """


    @pytest.mark.parametrize("out_features", [1, 2])
    def test_out_features(
        self, out_features, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        circuit=SimpleQuantumCircuit(3,3)
        m = MultiLayerHybrid(quantum=circuit,out_features=out_features).to(device)
        assert m[-2].out_features == out_features

    @pytest.mark.gpu
    @pytest.mark.parametrize("out_features", [1, 2])
    def test_out_features_gpu(self, out_features):
        self.test_out_features(out_features, device="cuda")
    def test_shape(self):
        circuit=SimpleQuantumCircuit(3,3)
        N = 10
        H_in = 11
        M_1 = 12
        M_2 = 13
        H_out = 14

        input = torch.zeros((N, H_in))
        m = MultiLayerHybrid(circuit,H_in, H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, H_in))
        m = MultiLayerHybrid(circuit,H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, H_in))
        m = MultiLayerHybrid(circuit,H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))

    def test_shape_lazy(self):
        circuit=SimpleQuantumCircuit(3,3)
        N = 10
        H_in = 11
        M_1 = 12
        M_2 = 13
        H_out = 14

        input = torch.zeros((N, H_in))
        m = MultiLayerHybrid(circuit,out_features=H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, H_in))
        m = MultiLayerHybrid(circuit,out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, H_in))
        m = MultiLayerHybrid(circuit,out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))
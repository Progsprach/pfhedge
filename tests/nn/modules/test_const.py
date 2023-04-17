from typing import Optional
from typing import Union

import pytest
import torch
from torch.testing import assert_close

from pfhedge.nn import ConstantLayer


class TestConstantLayer:
    """
    pfhedge.nn.ConstantLayer
    """

    @pytest.mark.parametrize("n_paths", [1, 10])
    @pytest.mark.parametrize("n_features", [1, 10])
    def test(
        self, n_paths, n_features, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        m = ConstantLayer().to(device)
        input1 = torch.zeros((n_paths, n_features)).to(device)
        input2 = torch.ones((n_paths, n_features)).to(device)
        output1 = m(input1)
        output2 = m(input2)
        comp = torch.full_like(output1,output1[0][0].item()).to(device)
        assert_close(output1,output2)
        assert_close(output2,comp)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10])
    @pytest.mark.parametrize("n_features", [1, 10])
    def test_gpu(self, n_paths, n_features):
        self.test(n_paths, n_features, device="cuda")

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        N = 11
        H_in = 12
        M_1 = 13
        M_2 = 14

        input = torch.zeros((N, H_in)).to(device)
        m = ConstantLayer().to(device)
        assert m(input).size() == torch.Size((N, 1))

        input = torch.zeros((N, M_1, H_in)).to(device)
        m = ConstantLayer().to(device)
        assert m(input).size() == torch.Size((N, M_1, 1))

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        m = ConstantLayer().to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
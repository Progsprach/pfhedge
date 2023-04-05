import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

class ConstantLayer(Module):
    """Dummy model learning a constant hedge."""
    def __init__(self) -> None:
        super().__init__()
        self.bias = Parameter(torch.zeros(1,))
    def forward(self, input: Tensor)-> Tensor:
        shp = input.shape
        output = self.bias.unsqueeze(0).unsqueeze(0)
        return output.repeat(shp)
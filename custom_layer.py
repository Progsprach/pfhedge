import numpy as np
import torch
import torch.nn as nn


# def function(x, params):
#     a1, a2, a3, a4, a5 = params
#     # return x
#     return torch.sin(a1*x)*torch.sin(a2*x)*torch.sin(a3*x)*torch.sin(a4*x)*torch.sin(a5*x)


def function(x, params):
    out = 1
    for i in range(0, len(params), 1):
        out *= torch.sin(params[i]*x)
    return out


class NonLinearLayer(nn.Module):
    def __init__(self, function, n_params):
        super(NonLinearLayer, self).__init__()
        self.function = function
        params = torch.Tensor(n_params)
        self.params = nn.Parameter(params)
        nn.init.normal_(self.params, mean=0., std=1.)

    def forward(self, x):
        return self.function(x, self.params)
    
print(dir(torch.sin))
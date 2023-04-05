from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack

def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def t2j(x_torch):
    x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

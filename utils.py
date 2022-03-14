"""
    Miscellaneous utils
"""
import sys

import torch

BYTES_IN_MB = 1024*1024

"""
    Memory usage in MB
"""
def tensor_memory_usage(t: torch.Tensor):
    return sys.getsizeof(t.storage()) / BYTES_IN_MB

"""
    Model size in MB
"""
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / BYTES_IN_MB

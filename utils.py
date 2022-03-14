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

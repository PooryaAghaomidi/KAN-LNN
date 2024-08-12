import torch
import random
import numpy as np


def set_torch_seed(device, seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed_value)

    return generator


def set_torch_device():
    print(f'PyTorch version: {torch.__version__}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Device: {device}')

    return device

import torch
import sys


def normalize(x, clip=5):
    """
    Normalize the input tensor, used in state and reward normalization.
    """
    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)
    stds = torch.where(stds == 0, torch.tensor(1e-8), stds)

    x = (x - means) / stds
    x = torch.clamp(x, -clip, clip)
    return x

def get_name():
    for i, arg in enumerate(sys.argv):
        if arg == '--alg_name':
            if i + 1 < len(sys.argv):
                alg_name = sys.argv[i + 1]
                break
    else:
        raise ValueError("Algorithm name not found in command line arguments.")
    return alg_name
import torch

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

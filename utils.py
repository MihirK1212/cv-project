import torch
from einops import rearrange

def trunc_normal(tensor, mean=0., std=1.):
    size = tensor.shape
    numel = tensor.numel()
    truncation = 2 * std
    lower = mean - truncation
    upper = mean + truncation
    with torch.no_grad():
        normal_tensor = torch.randn(size)
        normal_tensor = torch.clamp(normal_tensor, lower, upper)
        normal_tensor = normal_tensor * std / normal_tensor.std()
        tensor.copy_(normal_tensor)
    return tensor

def reshape_channel_first(x, height, width):
    x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=height)
    return x

def reshape_channel_last(x):
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x
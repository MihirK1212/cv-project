import torch

class LayerNorm2d(torch.nn.Module):
    def __init__(self, features, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, features, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
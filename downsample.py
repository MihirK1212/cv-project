import torch

from norm import LayerNorm2d
import utils

class StemDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size):
        super(StemDownsample, self).__init__()
        dim = out_channels // 2
        self.proj = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                dim,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                stride=patch_size // 2,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.layer_norm = torch.nn.LayerNorm(out_channels)


    def forward(self, x):
        x = self.proj(x)

        height, width = x.shape[2], x.shape[3]
        x = utils.reshape_channel_last(x)
        x = self.layer_norm(x)
        x = utils.reshape_channel_first(x, height, width)
        
        return x

class TransitionDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size):
        super(TransitionDownsample, self).__init__()
        dim = out_channels // 2
        self.proj = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=patch_size,
            padding=(kernel_size - 1) // 2,
        )
        self.layer_norm = torch.nn.LayerNorm(out_channels)


    def forward(self, x):
        x = self.proj(x)

        height, width = x.shape[2], x.shape[3]
        x = utils.reshape_channel_last(x)
        x = self.layer_norm(x)
        x = utils.reshape_channel_first(x, height, width)
        
        return x
    
def get_downsample_layer(layer_num, in_channels, out_channels):
    if layer_num==0:
        stem_downsample_layer = StemDownsample(
            in_channels=in_channels, 
            out_channels=out_channels,
            patch_size=4,
            kernel_size=3
        )
        return stem_downsample_layer
    else:
        transition_downsample_layer = TransitionDownsample(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=2,
            kernel_size=2
        )
        return transition_downsample_layer
import torch

import downsample
from pacablock import PaCaBlock
from downsample import LayerNorm2d

class PaCaVIT(torch.nn.Module):
    def __init__(
        self,
        img_size = 224,
        num_blocks = 4,
        embed_dims=[96, 192, 320, 384],
        depths=[2, 2, 4, 2],
    ):
        super(PaCaVIT, self).__init__()

        self.num_blocks = num_blocks
        self.embed_dims = embed_dims
        self.depths = depths
        
        for layer_num in range(self.num_blocks):

            downsample_layer = downsample.get_downsample_layer(
                layer_num, 
                in_channels=(3 if layer_num==0 else embed_dims[layer_num-1]),
                out_channels=embed_dims[layer_num]
            )
            setattr(self, f'downsample_{layer_num}', downsample_layer)

            for depth_num in range(self.depths[layer_num]): 
                paca_block = PaCaBlock(
                    embed_dim=embed_dims[layer_num],
                    num_heads=8,
                    input_img_shape=(img_size//(4 * 2**layer_num), img_size//(4 * 2**layer_num)),
                    with_pos_embed=(depth_num == 0)
                )
                setattr(self, f'pacablock_{layer_num}_{depth_num}', paca_block)

            layer_norm = LayerNorm2d(embed_dims[layer_num])
            setattr(self, f'layer_norm_{layer_num}', layer_norm)

    def forward(self, x):
        
        for layer_num in range(self.num_blocks):

            stage = f'downsample_{layer_num}'
            downsample_layer = getattr(self, stage)
            x = downsample_layer(x) 

            for depth_num in range(self.depths[layer_num]):
                stage = f'pacablock_{layer_num}_{depth_num}'
                paca_block = getattr(self, stage)
                x = paca_block(x)

            stage = f'layer_norm_{layer_num}'
            layer_norm = getattr(self, stage)
            x = layer_norm(x)

        return x
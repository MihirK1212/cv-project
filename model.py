import torch

import downsample

class PaCaVIT(torch.nn.Module):
    def __init__(
        self,
        img_size = 224,
        num_blocks = 4,
        embed_dims=[96, 192, 320, 384],
    ):
        super(PaCaVIT, self).__init__()

        self.num_blocks = num_blocks
        self.embed_dims = embed_dims
        
        for layer_num in range(self.num_blocks):
            downsample_layer = downsample.get_downsample_layer(
                layer_num, 
                in_channels=(3 if layer_num==0 else embed_dims[layer_num-1]),
                out_channels=embed_dims[layer_num]
            )
            setattr(self, f'downsample_{layer_num}', downsample_layer)

    def forward(self, x):
        print()
        for i in range(self.num_blocks):
            stage = f'downsample_{i}'
            downsample_layer = getattr(self, stage)
            x = downsample_layer(x)
            print(f'x shape {stage}:', x.shape)
        return x

batch_size = 17
img_size = 224

x = torch.rand(batch_size, 3, img_size, img_size)
model = PaCaVIT()

output = model(x)
print('output shape:', output.shape)


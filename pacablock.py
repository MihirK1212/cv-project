import torch
from einops import rearrange

import utils
from norm import LayerNorm2d

device = utils.get_device()

class PaCaAttention(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        embed_dim,
        num_heads,
    ):
        super(PaCaAttention, self).__init__()

        self.num_heads = num_heads
        self.attn_drop = 0.0
        
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, 
            self.num_heads, 
            dropout=self.attn_drop
        )

        self.proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, height, width, clustering_model):

        x = utils.reshape_channel_last(x)
        
        c = clustering_model(x)
        c = torch.transpose(c, 1, 2)
        c = c.softmax(dim=-1)

        z = torch.einsum("bmn,bnc->bmc", c, x)

        x = rearrange(x, "B N C -> N B C").to(device)
        z = rearrange(z, "B M C -> M B C").to(device)

        x, attn = self.multihead_attn(x, z, z)
        
        x = rearrange(x, "N B C -> B N C").to(device)

        x = utils.reshape_channel_first(x, height, width)

        return x


class DWConv(torch.nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=True, with_shortcut=True):
        super().__init__()
        self.dwconv = torch.nn.Conv2d(
            embed_dim, embed_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=bias, groups=embed_dim
        )
        self.with_shortcut = with_shortcut

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        if self.with_shortcut:
            return x + shortcut
        return x

class FFN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        with_shortcut=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, with_shortcut=with_shortcut)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x, height, width):

        x = utils.reshape_channel_last(x)
        x = self.fc1(x)
        x = utils.reshape_channel_first(x, height, width)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)

        x = utils.reshape_channel_last(x)
        x = self.fc2(x)
        x = utils.reshape_channel_first(x, height, width)
        
        x = self.drop(x)
        return x

class PaCaBlock(torch.nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        input_img_shape,
        with_pos_embed,
        drop=0.0
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.with_pos_embed = with_pos_embed
        self.input_img_shape = input_img_shape

        if self.with_pos_embed:
            assert self.input_img_shape is not None
            self.pos_embed = torch.nn.Parameter(
                torch.zeros(1, self.input_img_shape[0] * self.input_img_shape[1], self.embed_dim)
            )
            self.pos_drop = torch.nn.Dropout(p=drop)
            utils.trunc_normal(self.pos_embed, std=0.02)

        self.layer_norm_1 = LayerNorm2d(self.embed_dim)

        self.paca_attn = PaCaAttention(
            num_tokens=self.input_img_shape[0]*self.input_img_shape[1], embed_dim=self.embed_dim, num_heads=self.num_heads
        )

        self.layer_norm_2 = LayerNorm2d(self.embed_dim)
        self.ffn = FFN(in_features=self.embed_dim)
        self.layer_norm_3 = LayerNorm2d(self.embed_dim)

    def forward(self, x, clustering_model):

        skip_connection_1 = x 

        if self.with_pos_embed:
            x = self.pos_drop(utils.reshape_channel_last(x) + self.pos_embed)
            x = utils.reshape_channel_first(x, self.input_img_shape[0], self.input_img_shape[1])

        x = self.layer_norm_1(x)

        x = self.paca_attn(x, self.input_img_shape[0], self.input_img_shape[1], clustering_model)
        
        x = x + skip_connection_1

        skip_connection_2 = x

        x = self.layer_norm_2(x)
        x = self.ffn(x, self.input_img_shape[0], self.input_img_shape[1])        
        x = self.layer_norm_3(x)

        x = x + skip_connection_2

        return x

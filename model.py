import torch

import downsample
from pacablock import PaCaBlock
from downsample import LayerNorm2d
from clustering import get_clustering_model
import config

class PaCaVIT(torch.nn.Module):
    def __init__(
        self,
        img_size = 224,
        num_blocks = 3,
        embed_dims=[96, 192, 384],
        depths=[2, 2, 2],
    ):
        super(PaCaVIT, self).__init__()

        self.num_blocks = num_blocks
        self.embed_dims = embed_dims
        self.depths = depths
        
        for block_num in range(self.num_blocks):

            downsample_layer = downsample.get_downsample_layer(
                block_num, 
                in_channels=(3 if block_num==0 else embed_dims[block_num-1]),
                out_channels=embed_dims[block_num]
            )
            setattr(self, f'downsample_{block_num}', downsample_layer)

            
            for depth_num in range(self.depths[block_num]): 

                clustering_model = get_clustering_model()
                setattr(self, f'clustering_{block_num}_{depth_num}', clustering_model)

                paca_block = PaCaBlock(
                    embed_dim=embed_dims[block_num],
                    num_heads=config.NUM_HEADS,
                    input_img_shape=(img_size//(4 * 2**block_num), img_size//(4 * 2**block_num)),
                    with_pos_embed=(depth_num == 0)
                )
                setattr(self, f'pacablock_{block_num}_{depth_num}', paca_block)

            layer_norm = LayerNorm2d(embed_dims[block_num])
            setattr(self, f'layer_norm_{block_num}', layer_norm)

        # final_img_shape = img_size//(4 * 2**(self.num_blocks-1))
        # final_dense_dim = self.embed_dims[-1]*final_img_shape*final_img_shape
        
        # self.pre_classifier = torch.nn.Linear(
        #     final_dense_dim, final_dense_dim
        # )
        # self.dropout_classificaition_1 = torch.nn.Dropout(config.DROPOUT_CLASSIFICATION)

        # self.classifier_hidden_1 = torch.nn.Linear(
        #     final_dense_dim, 1024
        # )
        # self.dropout_classificaition_2 = torch.nn.Dropout(config.DROPOUT_CLASSIFICATION)

        # self.classifier_hidden_2 = torch.nn.Linear(
        #     1024, 256
        # )
        # self.dropout_classificaition_3 = torch.nn.Dropout(config.DROPOUT_CLASSIFICATION)

        # self.classifier = torch.nn.Linear(
        #     256, config.NUM_CLASSES
        # )

    def forward(self, x):

        for block_num in range(self.num_blocks):

            stage = f'downsample_{block_num}'
            downsample_layer = getattr(self, stage)
            x = downsample_layer(x) 

            for depth_num in range(self.depths[block_num]):

                stage = f'clustering_{block_num}_{depth_num}'
                clustering_model = getattr(self, stage)

                stage = f'pacablock_{block_num}_{depth_num}'
                paca_block = getattr(self, stage)
                x = paca_block(x, clustering_model)

            stage = f'layer_norm_{block_num}'
            layer_norm = getattr(self, stage)
            x = layer_norm(x)

        return {"pred_logits": x}
        
        b, c, h, w = x.size()
        pooler = x.view(b, c * h * w)
        pooler = self.dropout_classificaition_1(torch.nn.ReLU()(self.pre_classifier(pooler)))
        pooler = self.dropout_classificaition_2(torch.nn.ReLU()(self.classifier_hidden_1(pooler)))
        pooler = self.dropout_classificaition_3(torch.nn.ReLU()(self.classifier_hidden_2(pooler)))

        output = self.classifier(pooler)

        return {"pred_logits": output}
    
def get_model():
    model = PaCaVIT()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=config.LR_SCHEDULER_GAMMA
    )
    return model, loss_function, optimizer, scheduler
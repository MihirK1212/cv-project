import torch
from einops import rearrange
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
import utils
import torch.nn as nn
import torch.nn.functional as F


class Clustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=100
    ):
        super(Clustering, self).__init__()
        self.num_clusters = num_clusters   
    
    def generate_one_hot_tensor(self, batch_size, N, M):
        tensor = torch.zeros(batch_size, N, M)
        indices = torch.randint(0, M, (batch_size, N))
        tensor.scatter_(2, indices.unsqueeze(2), 1)
        return tensor
    
    def forward(self, x):
        num_tokens =  x.shape[1]
        return self.generate_one_hot_tensor(x.shape[0], num_tokens, self.num_clusters)

class KMeansClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=100
    ):
        super(KMeansClustering, self).__init__()
        self.num_clusters = num_clusters   
        self.cluster_centers = None
        
    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        
        if self.cluster_centers is None:
            kmeans = KMeans(n_clusters=self.num_clusters)
        else:
            kmeans = KMeans(n_clusters=self.num_clusters, init=self.cluster_centers, n_init=1)
        
        cluster_assignments = kmeans.fit_predict(x.reshape(batch_size * num_tokens, -1).cpu().numpy())
        
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        
        tensor = tensor.reshape(batch_size, num_tokens, self.num_clusters)
        
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_).view(1, 1, -1)
            
        return tensor

class HierarchicalClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=100,
        linkage=None,
        distance_threshold=None
    ):
        super(HierarchicalClustering, self).__init__()
        self.num_clusters = num_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        
        if self.linkage is None:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, distance_threshold=self.distance_threshold)
            cluster_assignments = clustering.fit_predict(x.reshape(batch_size * num_tokens, -1))
        else:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, linkage=self.linkage)
            cluster_assignments = clustering.fit_predict(x.reshape(batch_size * num_tokens, -1))
        
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        
        tensor = tensor.reshape(batch_size, num_tokens, self.num_clusters)
        
        self.linkage = clustering.children_
        
        return tensor


class MLPClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=100
    ):
        super(MLPClustering, self).__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = 96

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_clusters)

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        self.fc1.in_features = input_dim
        x = x.reshape(-1, input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        cluster_probs = x.view(batch_size, num_tokens, self.num_clusters)

        return cluster_probs
    
class PaCaAttention(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        embed_dim,
        num_heads,
    ):
        super(PaCaAttention, self).__init__()

        self.clustering = MLPClustering()

        self.q = torch.nn.Linear(embed_dim, embed_dim)
        self.k = torch.nn.Linear(embed_dim, embed_dim)
        self.v = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.attn_drop = 0.0

        self.proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        height = x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        c = self.clustering(x)
        c = torch.transpose(c, 1, 2)
        c = c.softmax(dim=-1)

        z = torch.einsum("bmn,bnc->bmc", c, x)
        print('z shape:', z.shape)

        x = rearrange(x, "B N C -> N B C")
        z = rearrange(z, "B M C -> M B C")

        x, attn = torch.nn.functional.multi_head_attention_forward(
            query=x,
            key=z,
            value=z,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q.weight,
            k_proj_weight=self.k.weight,
            v_proj_weight=self.v.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_drop,
            out_proj_weight=self.proj.weight,
            out_proj_bias=self.proj.bias,
            use_separate_proj_weight=True,
            training=utils.is_training(),
            need_weights=False,
            average_attn_weights=False,
        )

        x = rearrange(x, "N B C -> B N C")

        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=height)
        return x


x = torch.rand(17, 96, 56, 56)

pct = PaCaAttention(num_tokens=x.shape[-1]*x.shape[-2], embed_dim=x.shape[1], num_heads=8)
output = pct(x)


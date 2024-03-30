import torch
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
        
        cluster_assignments = kmeans.fit_predict(x.reshape(batch_size * num_tokens, -1).detach().cpu().numpy())
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_).view(1, 1, -1).clone().to(utils.get_device())
        
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        
        tensor = tensor.reshape(batch_size, num_tokens, self.num_clusters)
        cluster_tensor = tensor.clone().to(utils.get_device())
        
        return cluster_tensor

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
    
def get_clustering_model():
    return Clustering()
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import utils
import torch.nn as nn
import torch.nn.functional as F
import config


device = utils.get_device()

class Clustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=config.NUM_CLUSTERS
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
        return self.generate_one_hot_tensor(x.shape[0], num_tokens, self.num_clusters).clone().to(device)

class KMeansClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=config.NUM_CLUSTERS
    ):
        super(KMeansClustering, self).__init__()
        self.num_clusters = num_clusters   
        self.cluster_centers = None
        
    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        if self.cluster_centers is None:
            kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto')
        else:
            kmeans = KMeans(n_clusters=self.num_clusters, init=self.cluster_centers, n_init=1)
        
        cluster_assignments = kmeans.fit_predict(x.reshape(batch_size * num_tokens, -1).detach().cpu().numpy())
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_).detach().cpu().numpy()
        
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        
        tensor = tensor.reshape(batch_size, num_tokens, self.num_clusters)
        cluster_tensor = tensor.clone().to(device)
        
        return cluster_tensor


class HierarchicalClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=config.NUM_CLUSTERS,
        linkage=None
    ):
        super(HierarchicalClustering, self).__init__()
        self.num_clusters = num_clusters
        self.linkage = linkage

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        
        if self.linkage is None:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters)
            cluster_assignments = clustering.fit_predict(x.reshape(batch_size * num_tokens, -1).detach().cpu().numpy())
        else:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, linkage=self.linkage)
            cluster_assignments = clustering.fit_predict(x.reshape(batch_size * num_tokens, -1).detach().cpu().numpy())
        
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        
        tensor = tensor.reshape(batch_size, num_tokens, self.num_clusters)
        cluster_tensor = tensor.clone().to(device)

        self.linkage = clustering.children_
        
        return cluster_tensor


class MLPClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=config.NUM_CLUSTERS,
        hidden_dim=256,
        num_layers=2,
        activation=torch.nn.ReLU(),
        input_dim=96,
    ):
        super(MLPClustering, self).__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.input_dim=input_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(torch.nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(torch.nn.Linear(hidden_dim, num_clusters))
            else:
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        x = x.reshape(batch_size * num_tokens, input_dim)

        mlp_output = self.mlp(x)

        mlp_output = mlp_output.view(batch_size, num_tokens, self.num_clusters)

        cluster_tensor = torch.nn.functional.softmax(mlp_output, dim=-1)
        print(cluster_tensor.shape)
        return cluster_tensor

    
class GMMClustering(torch.nn.Module):
    def __init__(
        self,
        num_clusters=config.NUM_CLUSTERS
    ):
        super(GMMClustering, self).__init__()
        self.num_clusters = num_clusters
        self.means = None
        self.covariances = None
        self.weights = None

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        x_reshaped = x.reshape(-1, input_dim)

        if self.means is None:
            gmm = GaussianMixture(n_components=self.num_clusters)
        else:
            gmm = GaussianMixture(n_components=self.num_clusters, means_init=self.means,
                                  covariances_init=self.covariances, weights_init=self.weights)

        gmm.fit(x_reshaped.detach().cpu().numpy())

        self.means = torch.tensor(gmm.means_).detach().cpu().numpy()
        self.covariances = torch.tensor(gmm.covariances_).detach().cpu().numpy()
        self.weights = torch.tensor(gmm.weights_).detach().cpu().numpy()

        cluster_assignments = gmm.predict(x_reshaped.detach().cpu().numpy())
        tensor = torch.zeros(batch_size * num_tokens, self.num_clusters)
        for i in range(batch_size * num_tokens):
            tensor[i, cluster_assignments[i]] = 1
        tensor = tensor.view(batch_size, num_tokens, self.num_clusters)
        cluster_tensor = tensor.clone().to(device)
        return cluster_tensor
    

def get_clustering_model():
    return KMeansClustering()
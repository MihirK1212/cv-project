import torch
from einops import rearrange
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    return tensor.to(get_device())

def reshape_channel_first(x, height, width):
    x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=height)
    return x

def reshape_channel_last(x):
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

def replace_elements(input_tensor, tolerance=1e-5):
    input_tensor = input_tensor.float()
    replaced_tensor = torch.where(input_tensor > tolerance, 1 / input_tensor, torch.ones_like(input_tensor) / tolerance)
    return replaced_tensor


def get_pairwise_inverse_euclidian_distance(tensor1, tensor2):
    m, n = tensor1.size(0), tensor2.size(0)
    d = tensor1.size(1)
    tensor1 = tensor1.unsqueeze(1).expand(m, n, d)
    tensor2 = tensor2.unsqueeze(0).expand(m, n, d)
    distances = torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=2))
    distances = replace_elements(distances)
    return distances

def get_metrics(targets, predictions):

    try:
        targets = targets.detach().cpu()
    except:
        pass

    try:
        predictions = predictions.detach().cpu()
    except:
        pass


    accuracy = accuracy_score(targets, predictions)
    print("Accuracy:", accuracy)

    precision = precision_score(targets, predictions, average="weighted")
    print("Precision:", precision)

    recall = recall_score(targets, predictions, average="weighted")
    print("Recall:", recall)

    f1 = f1_score(targets, predictions, average="weighted")
    print("F1-score:", f1)

    weighted_f1 = f1_score(targets, predictions, average="weighted")
    print("Weighted-F1:", weighted_f1)

    micro_f1 = f1_score(targets, predictions, average="micro")
    print("Micro-F1:", micro_f1)

    macro_f1 = f1_score(targets, predictions, average="macro")
    print("Macro-F1:", macro_f1)

    confusion_mat = confusion_matrix(targets, predictions)
    print("Confusion Matrix:")
    print(confusion_mat)

    return accuracy, precision, recall, f1, weighted_f1, micro_f1, macro_f1


def silhouette_value(samples, cluster_assignments, num_clusters):
    """
    Calculate the silhouette value for a given clustering.
    
    Parameters:
    samples (torch.Tensor): A tensor of shape (n, d) representing 'n' samples with 'd' features each.
    cluster_assignments (torch.Tensor): A tensor of shape (n,) representing the cluster assignment for each sample.
    num_clusters (int): The number of clusters 'm' (range of cluster assignments [0, m-1]).
    
    Returns:
    float: The mean silhouette value of the clustering.
    """
    
    # Ensure the tensors are on the same device
    samples = samples.to(cluster_assignments.device)
    
    # Calculate pairwise distances between samples
    n = samples.shape[0]
    pairwise_distances = torch.cdist(samples, samples)
    
    # Calculate intra-cluster distance (a) and nearest-cluster distance (b)
    a = torch.zeros(n)
    b = torch.full((n,), float('inf'))
    
    for i in range(n):
        # Samples in the same cluster as sample i
        same_cluster_mask = cluster_assignments == cluster_assignments[i]
        same_cluster_mask[i] = False  # Exclude the sample itself
        if same_cluster_mask.any():
            a[i] = pairwise_distances[i, same_cluster_mask].mean()
        
        # Calculate the average distance to the nearest other cluster
        for cluster in range(num_clusters):
            if cluster != cluster_assignments[i]:
                # Samples in the other cluster
                other_cluster_mask = cluster_assignments == cluster
                if other_cluster_mask.any():
                    mean_distance = pairwise_distances[i, other_cluster_mask].mean()
                    if mean_distance < b[i]:
                        b[i] = mean_distance
    
    # Calculate silhouette value for each sample
    silhouette_values = (b - a) / torch.maximum(a, b)
    
    # Calculate the mean silhouette value
    mean_silhouette_value = silhouette_values.mean().item()
    
    return mean_silhouette_value
import torch
from einops import rearrange
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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
    return tensor

def reshape_channel_first(x, height, width):
    x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=height)
    return x

def reshape_channel_last(x):
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x

def get_metrics(targets, predictions):
    
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
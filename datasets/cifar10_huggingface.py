import torch
from torchvision import datasets, transforms

import config

class CustomCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {'x': img, 'labels': target}

def get_dataset():
    # Transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = CustomCIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CustomCIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader
    training_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    testing_loader = validation_loader

    return training_loader, validation_loader, testing_loader
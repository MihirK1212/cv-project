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

    # def check_for_nan(loader, dataset_type):
    #     nan_found = False
    #     for i, batch in enumerate(loader):
    #         images = batch['x']
    #         if torch.isnan(images).any():
    #             print(f"NaN found in {dataset_type} loader in batch {i}")
    #             nan_found = True
    #     if not nan_found:
    #         print(f"No NaN values found in {dataset_type} loader")

    # Assuming train_loader and test_loader are already defined

    # # Check for NaN values in train loader
    # print("Checking train loader for NaN values:")
    # check_for_nan(training_loader, "train")

    # # Check for NaN values in test loader
    # print("Checking test loader for NaN values:")
    # check_for_nan(testing_loader, "test")

    return training_loader, validation_loader, testing_loader
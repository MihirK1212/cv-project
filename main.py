import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import model
import utils
import config

device = utils.get_device()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'labels': self.y[idx]}

def get_dataset(num_samples = 1000):
  X = [torch.rand(3, config.IMG_SIZE, config.IMG_SIZE).to(device).cuda() for _ in range(num_samples)]
  y = [torch.randint(0, config.NUM_CLASSES, (1,)).item() for _ in range(num_samples)]
  return X, y

def helper():

    X, y = get_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    valid_dataset = CustomDataset(X_valid, y_valid)
    test_dataset = CustomDataset(X_test, y_test)

    training_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    testing_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    paca_vit_model, loss_function, optimizer, scheduler = model.get_model()
    paca_vit_model.to(device)

    return (
        paca_vit_model,
        training_loader,
        validation_loader,
        testing_loader,
        loss_function,
        optimizer,
        scheduler
    )

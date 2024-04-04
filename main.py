import torch
from torch.utils.data import Dataset, DataLoader

import model
import utils
import config
from datasets import imagenet_200_local

device = utils.get_device()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'labels': self.y[idx]}

def helper():

    if config.DATASET == config.IMAGENET2000LOCAL:
        X_train, X_valid, X_test, y_train, y_valid, y_test = imagenet_200_local.get_dataset()
        print('TRAIN DATASET SIZE:', len(X_train))
        print('VALID DATASET SIZE:', len(X_valid))
        print('TEST DATASET SIZE:', len(X_test))

        assert len(X_train) == len(y_train)
        assert len(X_valid) == len(y_valid)
        assert len(X_test) == len(y_test)
        assert X_train[0].shape == (3, config.IMG_SIZE, config.IMG_SIZE)

        if not config.USE_RANDOM_DATASET:
            assert len(list(set(y_train))) == len(list(set(y_valid))) and len(list(set(y_train))) == config.NUM_CLASSES

        train_dataset = CustomDataset(X_train, y_train)
        valid_dataset = CustomDataset(X_valid, y_valid)
        test_dataset = CustomDataset(X_test, y_test)

        training_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        testing_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    

    else:
        raise ValueError

    
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

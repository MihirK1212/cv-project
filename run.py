import torch
from tqdm import tqdm

import utils
import config
import main
from model import PaCaVIT

device = utils.get_device()

x = torch.rand(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
model = PaCaVIT()
output = model(x)
print('output shape:', output.shape)

def train(
    model, training_loader, loss_function, optimizer, scheduler
):
    num_tr_steps = 0
    tr_loss = 0
    batch_predictions = []
    batch_targets = []

    model.train()

    for _, data in tqdm(enumerate(training_loader, 0)):
        torch.cuda.empty_cache()

        inputs = data['x'].to(device)
        targets = data['labels'].to(device, dtype=torch.long)

        outputs = model(inputs)

        loss = loss_function(outputs, targets)
        _, predictions = torch.max(outputs["pred_logits"], dim=1)

        num_tr_steps += 1
        tr_loss += loss.detach().item()
        batch_predictions.append(predictions.to(torch.long))
        batch_targets.append(targets.to(torch.long))

        optimizer.zero_grad()
        loss.backward()
        if config.APPLY_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if config.USE_LR_SCHEDULER:
        assert scheduler is not None
        scheduler.step()

    epoch_loss = tr_loss / num_tr_steps
    print(f"Training Loss for Epoch:", epoch_loss)

    predictions = torch.cat(batch_predictions)
    targets = torch.cat(batch_targets)
    print("Training Metrics:")
    return epoch_loss, *utils.get_metrics(
        targets, predictions
    )


def valid(
    model, validation_loader, loss_function
):
    num_tr_steps = 0
    tr_loss = 0
    batch_predictions = []
    batch_targets = []

    model.eval()

    for _, data in tqdm(enumerate(validation_loader, 0)):
        torch.cuda.empty_cache()
        
        inputs = data['x'].to(device)
        targets = data['labels'].to(device, dtype=torch.long)

        outputs = model(inputs)

        loss = loss_function(outputs, targets)
        _, predictions = torch.max(outputs["pred_logits"], dim=1)

        num_tr_steps += 1
        tr_loss += loss.item()
        batch_predictions.append(predictions.to(torch.long))
        batch_targets.append(targets.to(torch.long))

    epoch_loss = tr_loss / num_tr_steps
    print(f"Validation Loss for Epoch:", epoch_loss)
    predictions = torch.cat(batch_predictions)
    targets = torch.cat(batch_targets)
    print("Validation Metrics:")
    return epoch_loss, *utils.get_metrics(
        targets, predictions
    )


# (
#     paca_vit_model,
#     training_loader,
#     validation_loader,
#     testing_loader,
#     loss_function,
#     optimizer,
#     scheduler
# ) = main.helper()


# for epoch in range(config.NUM_EPOCHS):
        
#     print("###########################")
#     print()


#     print("######### Training #########")
#     train_loss, accuracy, precision, recall, f1, weighted_f1, micro_f1, macro_f1 = train(
#         paca_vit_model,
#         training_loader,
#         loss_function,
#         optimizer,
#         scheduler
#     )
#     print()

#     print("######### Validation #########")
#     valid_loss, accuracy, precision, recall, f1, weighted_f1, micro_f1, macro_f1 = valid(
#         paca_vit_model,
#         validation_loader,
#         loss_function
#     )
#     print()




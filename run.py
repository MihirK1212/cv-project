import torch
from tqdm import tqdm
import os
from datetime import datetime

import utils
import config
import main
from model import PaCaVIT

device = utils.get_device()
assert device == "cuda"

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

        loss = loss_function(outputs['pred_logits'], targets)
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

        loss = loss_function(outputs['pred_logits'], targets)
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

if __name__ == "__main__":

    (
        paca_vit_model,
        training_loader,
        validation_loader,
        testing_loader,
        loss_function,
        optimizer,
        scheduler
    ) = main.helper()
    

    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    log_file_path = f"./logs/train_{timestamp}.txt"

    config_lines = [
        f"SEED: {config.SEED}\n", 
        f"BATCH_SIZE: {config.BATCH_SIZE}\n",
        f"INITIAL_LR: {config.LEARNING_RATE}\n",
        f"LR_SCHEDULER_GAMMA: {config.LR_SCHEDULER_GAMMA}\n",
        f"WEIGHT_DECAY: {config.WEIGHT_DECAY}\n",
        f"APPLY_GRADIENT_CLIPPING: {config.APPLY_GRADIENT_CLIPPING}\n",
        f"USE_LR_SCHEDULER: {config.USE_LR_SCHEDULER}\n",
        "\n",
        f"DATASET: {config.DATASET}\n"
        "\n",
    ]
    with open(log_file_path, "w+") as file:
        file.writelines(config_lines)
    
    max_acc = 0

    for epoch in range(config.NUM_EPOCHS):

        log = open(log_file_path, "a")
        log.write(f"Epoch: {epoch+1}\n")            

        device = utils.get_device()
        print('DEVICE:', device)
            
        print("###########################")
        print()


        print("######### Training #########")
        train_loss, accuracy, precision, recall, f1, weighted_f1, micro_f1, macro_f1 = train(
            paca_vit_model,
            training_loader,
            loss_function,
            optimizer,
            scheduler
        )
        print()
    
        train_lines = [
            f"Training Metrics:\n"
            f" Loss: {train_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, WeightedF1: {weighted_f1}, MicroF1: {micro_f1}, MacroF1: {macro_f1}\n"
        ]
        log.writelines(train_lines)

        print("######### Validation #########")
        valid_loss, accuracy, precision, recall, f1, weighted_f1, micro_f1, macro_f1 = valid(
            paca_vit_model,
            validation_loader,
            loss_function
        )
        print()

        valid_lines = [
            f"Validation Metrics:\n"
            f" Loss: {valid_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, WeightedF1: {weighted_f1}, MicroF1: {micro_f1}, MacroF1: {macro_f1}\n"
        ]
        log.writelines(valid_lines)

        log.write("\n")
        log.close()

        if accuracy > max_acc and not config.USE_RANDOM_DATASET:
            max_acc = accuracy
            torch.save(paca_vit_model.state_dict(), os.path.join(config.MODEL_SAVE_BASE_PATH, 'paca_vit.pt'))
            paca_vit_model.save_clustering_models()
            

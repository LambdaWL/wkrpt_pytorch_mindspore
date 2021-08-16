import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from evaluation import Metrics


def run_torch_model_one_epoch(
    model, 
    loss_function, 
    dataloader, 
    device, 
    metrics: Metrics,
    optimizer=None, 
    max_grad_norm=5.0
):
    if optimizer is not None:
        optimizer.zero_grad()
    
    for (features, labels) in dataloader:
        if device == "GPU":
            features = features.cuda()  # (B, qlen, D)
            labels = labels.cuda()      # (B, qlen)
        
        logits = model(features)  # (B, qlen, C)
        loss = loss_function(logits.permute(0, 2, 1), labels)

        if optimizer is not None:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        metrics.update(
            loss=loss.item(),
            labels=labels.cpu().numpy(),
            predictions=torch.argmax(logits, dim=2).cpu().numpy(),
        )
    
    return metrics.evaluate()


def train_torch_model(
    model, 
    dataloader, 
    device, 
    epochs, 
    learning_rate=1e-3, 
    max_grad_norm=5.0, 
    padding_label=-1,
):
    if device == "GPU":
        model = model.cuda()
    model.train()
    
    loss_function = nn.CrossEntropyLoss(ignore_index=padding_label)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    metrics = Metrics(padding_label)

    train_kwargs = {
        "model": model, "loss_function": loss_function, "dataloader": dataloader,
        "device": device, "metrics": metrics, "optimizer": optimizer, 
        "max_grad_norm": max_grad_norm,
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        loss, accuracy, fscore = run_torch_model_one_epoch(**train_kwargs)
        print(
            f"Epoch {epoch}: Loss: {loss}, Acc: {accuracy}, F1 Score: {fscore}, "
            f"Total Time: {round(time.time() - epoch_start, 2)} s"
        )


def test_torch_model(model, dataloader, device, padding_label=-1):
    if device == "GPU":
        model = model.cuda()
    model.eval()

    loss_function = nn.CrossEntropyLoss(ignore_index=padding_label)
    metrics = Metrics(padding_label)

    test_kwargs = {
        "model": model, "loss_function": loss_function, "dataloader": dataloader,
        "device": device, "metrics": metrics,
    }

    epoch_start = time.time()
    loss, accuracy, fscore = run_torch_model_one_epoch(**test_kwargs)
    print(
        f"Test: Loss: {loss}, Acc: {accuracy}, F1 Score: {fscore}, "
        f"Total Time: {round(time.time() - epoch_start, 2)} s"
    )

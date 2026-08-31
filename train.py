"""Training loop with validation checkpointing and early stopping."""

from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    use_amp = device.type == "cuda"
    scaler = getattr(optimizer, "_emotion_scaler", None) if training else None
    model.train(training)
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def _class_weights(loader, device):
    subset = loader.dataset
    labels = [subset.dataset.samples[index][1] for index in subset.indices]
    counts = torch.bincount(torch.tensor(labels), minlength=7).float()
    weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    return weights.to(device)


def train_model(model, train_loader, validation_loader, device=None, epochs: int = 12,
                patience: int = 4, checkpoint_path: str | Path = "best_emotion_model.pth",
                history_path: str | Path = "training_history.json"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_loader, device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=2e-4, weight_decay=1e-4)
    if device.type == "cuda":
        optimizer._emotion_scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy, stale_epochs = -1.0, 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = _run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        history.append((train_loss, train_accuracy, validation_loss, validation_accuracy))
        print(f"Epoch {epoch:02d}/{epochs} - train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f} - "
              f"val loss: {validation_loss:.4f}, val acc: {validation_accuracy:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy, stale_epochs = validation_accuracy, 0
            torch.save({"model_state_dict": model.state_dict(), "val_accuracy": best_accuracy,
                        "epoch": epoch}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping activated.")
                break
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "framework": "PyTorch",
        "epochs_completed": len(history),
        "best_validation_accuracy": best_accuracy,
        "checkpoint": str(checkpoint_path),
        "history": [
            {"epoch": index, "train_loss": values[0], "train_accuracy": values[1],
             "validation_loss": values[2], "validation_accuracy": values[3]}
            for index, values in enumerate(history, start=1)
        ],
    }, indent=2), encoding="utf-8")
    return history
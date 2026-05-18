import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import build_dataloader
from model2 import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        batch_n = images.size(0)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_n
        acc_sum += accuracy(logits, targets) * batch_n
        n += batch_n

    return {"loss": loss_sum / n, "acc": acc_sum / n}


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        batch_n = images.size(0)

        logits = model(images)
        loss = criterion(logits, targets)

        loss_sum += loss.item() * batch_n
        acc_sum += accuracy(logits, targets) * batch_n
        n += batch_n

    return {"loss": loss_sum / n, "acc": acc_sum / n}


def main() -> None:
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(config.CHECKPOINT_DIR)

    train_root = Path(config.DATA_ROOT) / "train"
    val_root = Path(config.DATA_ROOT) / "train"

    train_loader = build_dataloader(
        train_root,
        train=True
    )
    val_loader = build_dataloader(
        val_root,
        train=False
    )

    model = UNet(3, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"epoch {epoch}/{config.NUM_EPOCHS}")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        print(
            f"  train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}"
        )
        print(f"  val   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_dir / "last.pt",
        )
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

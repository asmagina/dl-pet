import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_pet.engine.metrics import AverageMeter, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits, targets)
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

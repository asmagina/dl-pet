import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_pet.engine.metrics import AverageMeter, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_acc = accuracy(logits, targets)
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

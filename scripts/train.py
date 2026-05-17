import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dl_pet.config import load_config
from dl_pet.data.datamodule import build_dataloader, mvtec_split_root
from dl_pet.engine.eval_loop import evaluate
from dl_pet.engine.train_loop import train_one_epoch
from dl_pet.models.model import build_model
from dl_pet.utils.io import save_checkpoint
from dl_pet.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_root = mvtec_split_root(data_cfg["root"], data_cfg["category"], "train")
    val_root = mvtec_split_root(data_cfg["root"], data_cfg["category"], "test")

    train_loader = build_dataloader(
        train_root,
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train=True,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_root,
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train=False,
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    train_cfg = cfg["train"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    best_acc = -1.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"epoch {epoch}/{train_cfg['epochs']}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_every=train_cfg["log_every"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")
        print(f"  val   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            {"train": train_metrics, "val": val_metrics},
        )
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "val": val_metrics},
            )


if __name__ == "__main__":
    main()

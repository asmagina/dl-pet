import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dl_pet.config import load_config
from dl_pet.data.datamodule import build_dataloader, mvtec_split_root
from dl_pet.engine.eval_loop import evaluate
from dl_pet.models.model import build_model
from dl_pet.utils.io import load_checkpoint
from dl_pet.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a checkpoint on the test split")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    test_root = mvtec_split_root(data_cfg["root"], data_cfg["category"], "test")
    test_loader = build_dataloader(
        test_root,
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train=False,
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(cfg["eval"]["checkpoint"])
    load_checkpoint(checkpoint, model, device=device)

    metrics = evaluate(model, test_loader, criterion, device)
    print(f"test loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()

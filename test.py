from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from data import build_dataloader
from model2 import UNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_root = Path(config.DATA_ROOT) / "test-small"
    output_dir = Path("runs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(
        test_root,
        train=False
    )
    print(
        f"test images: start={config.TEST_START_INDEX}, "
        f"stop={config.TEST_STOP_INDEX}, count={len(loader.dataset)}"
    )

    ckpt = torch.load(config.BEST_CKPT, map_location=device, weights_only=False)
    model = UNet(3, 2).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    saved = 0
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            logits = model(images)
            masks = logits.argmax(dim=1).byte().cpu().numpy()

            for mask, name in zip(masks, names):
                out_name = Path(name).with_suffix(".png").name
                out_path = output_dir / out_name
                Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
                saved += 1

    print(f"saved {saved} predicted masks to {output_dir}")


if __name__ == "__main__":
    main()

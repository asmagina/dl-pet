from pathlib import Path
from PIL import Image
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import config

train_transform = A.Compose(
    [
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


eval_transform = A.Compose(
    [
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


class CarvanaSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.mask_dir = Path(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}_mask.gif"

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8)

        out = self.transform(image=image, mask=mask)
        mask = torch.as_tensor(out["mask"], dtype=torch.long)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        return out["image"], mask


class CarvanaTestDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = np.array(Image.open(path).convert("RGB"))
        out = self.transform(image=image)
        return out["image"], path.name


def build_dataloader(
    root: str | Path,
    train: bool,
    *,
    require_masks: bool = False,
) -> DataLoader:
    transform = train_transform if train else eval_transform
    root = Path(root)
    mask_dir = root.parent / "train_masks"
    has_masks = mask_dir.exists() and root.name == "train"

    if require_masks and not has_masks:
        raise FileNotFoundError(
            f"Training requires masks at {mask_dir}, but that directory was not found. "
            f"Expected Carvana layout: {root.parent}/train and {mask_dir}."
        )

    if has_masks:
        dataset = CarvanaSegmentationDataset(root, mask_dir, transform)
    else:
        dataset = CarvanaTestDataset(root, transform)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True if train else False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

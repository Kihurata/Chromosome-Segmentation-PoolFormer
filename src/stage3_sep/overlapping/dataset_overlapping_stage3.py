from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


class Stage3OverlappingMultiTaskDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        img_dir = self.root_dir / split / "images"
        fg_dir = self.root_dir / split / "foregrounds"
        ov_dir = self.root_dir / split / "overlaps"
        bd_dir = self.root_dir / split / "boundaries"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            fg_path = fg_dir / img_path.name
            ov_path = ov_dir / img_path.name
            bd_path = bd_dir / img_path.name
            if fg_path.exists() and ov_path.exists() and bd_path.exists():
                self.samples.append((str(img_path), str(fg_path), str(ov_path), str(bd_path)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fg_path, ov_path, bd_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        foreground = (np.array(Image.open(fg_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        overlap = (np.array(Image.open(ov_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        boundary = (np.array(Image.open(bd_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

        # Enforce target consistency to reduce contradictory supervision.
        overlap = (overlap & foreground).astype(np.uint8)
        boundary = (boundary & foreground).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[foreground, overlap, boundary])
            image = augmented["image"]
            foreground = augmented["masks"][0].float().clamp(0, 1)
            overlap = augmented["masks"][1].float().clamp(0, 1)
            boundary = augmented["masks"][2].float().clamp(0, 1)

        return image, foreground, overlap, boundary


def get_stage3_overlapping_multitask_transforms(img_size=384):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), rotate=(-25, 25), shear=(-8, 8), p=0.35),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ], p=0.35),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_transform

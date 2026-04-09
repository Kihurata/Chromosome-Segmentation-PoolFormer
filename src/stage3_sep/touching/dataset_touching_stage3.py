from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

class Stage3TouchingDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        img_dir = self.root_dir / split / "images"
        fg_dir = self.root_dir / split / "foregrounds"
        mk_dir = self.root_dir / split / "markers"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                continue
            fg_path = fg_dir / img_path.name
            mk_path = mk_dir / img_path.name
            if fg_path.exists() and mk_path.exists():
                self.samples.append((str(img_path), str(fg_path), str(mk_path)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fg_path, mk_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        foreground = (np.array(Image.open(fg_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        marker = (np.array(Image.open(mk_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[foreground, marker])
            image = augmented["image"]
            foreground = augmented["masks"][0].float()
            marker = augmented["masks"][1].float()

        return image, foreground, marker


def get_stage3_transforms(img_size=384):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform

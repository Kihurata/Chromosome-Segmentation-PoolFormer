import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

# Import mmpretrain to get backbone
try:
    from mmpretrain import get_model
except ImportError:
    print("Please install mmpretrain: pip install mmpretrain")
    sys.exit(1)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import ExperimentLogger

# ================== MODEL: PoolFormer-UNet ==================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PoolFormerUNet(nn.Module):
    def __init__(self, backbone_name, pretrained_path, num_classes=1):
        super().__init__()
        # Load backbone
        # PoolFormer m36 usually has channels: [64, 128, 320, 512] at strides [4, 8, 16, 32]
        # We need to verify this or get it dynamically. 
        # For 'poolformer_m36', the channels are [64, 128, 320, 512].
        
        self.backbone = get_model(backbone_name, pretrained=pretrained_path, backbone=dict(out_indices=(0, 1, 2, 3)))
        # Modify backbone to return features if not already (get_model usually returns classifier)
        # If it returns classifier, we need to extract backbone.
        if hasattr(self.backbone, 'backbone'):
            self.backbone = self.backbone.backbone
        
        # Encoder channels for m36
        enc_channels = [64, 128, 320, 512] 
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], 2, stride=2)
        self.conv1 = DoubleConv(enc_channels[3], enc_channels[2]) # cat(320, 320) -> 640 -> 320
        
        self.up2 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], 2, stride=2)
        self.conv2 = DoubleConv(enc_channels[2], enc_channels[1]) # cat(128, 128) -> 256 -> 128
        
        self.up3 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], 2, stride=2)
        self.conv3 = DoubleConv(enc_channels[1], enc_channels[0]) # cat(64, 64) -> 128 -> 64
        
        self.up4 = nn.ConvTranspose2d(enc_channels[0], 32, 2, stride=2) # 64 -> 32 (stride 4 original so need 2 steps or direct 4)
        # Note: Stage 0 is stride 4. So we need 2 upsamples to reach stride 1 (original size).
        # Actually PoolFormer Stem is stride 4.
        # Let's add a final upsampling block.
        
        self.final_conv = nn.Conv2d(enc_channels[0], num_classes, 1) 
        # Note: This outputs at stride 4. We can upsample output or add more decoder layers.
        # Let's simple bilinear upsample at the end to match input resolution.

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract features
        # mmpretrain backbone should return tuple of features if out_indices is set
        feats = self.backbone(x) 
        # feats[0]: stride 4 (64)
        # feats[1]: stride 8 (128)
        # feats[2]: stride 16 (320)
        # feats[3]: stride 32 (512)
        c1, c2, c3, c4 = feats
        
        # Decode
        x = self.up1(c4)
        # Resize if mismatch (due to odd input sizes)
        if x.shape[2:] != c3.shape[2:]:
            x =  nn.functional.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape[2:] != c2.shape[2:]:
            x =  nn.functional.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape[2:] != c1.shape[2:]:
            x =  nn.functional.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.conv3(x)
        
        # Output at stride 4
        out = self.final_conv(x)
        
        # Upsample to original input
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

# ================== DATASET ==================
# Reuse from previous step or imports
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []
        img_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')
        if os.path.exists(img_dir):
            files = sorted(os.listdir(img_dir))
            for f in files:
                self.images.append(os.path.join(img_dir, f))
                self.masks.append(os.path.join(mask_dir, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize(image.shape[1:], interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = np.array(mask)
            mask = (mask > 128).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage1_seg.yaml')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    logger = ExperimentLogger(cfg)
    print(f"[INFO] Stage 1 (PoolFormer) Experiment: {logger.exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    print(f"[INFO] Loading PoolFormer backbone: {cfg['model']['backbone']}")
    model = PoolFormerUNet(
        backbone_name=f"poolformer_{cfg['model']['backbone']}_3rdparty_in1k", # e.g. 'poolformer_m36_3rdparty_in1k'
        pretrained_path=cfg['model']['pretrained']
    ).to(device)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((cfg['model'].get('input_size', 320), cfg['model'].get('input_size', 320))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SegmentationDataset(cfg['training']['data_root'], transform=transform)
    if len(dataset) == 0:
        print("[WARN] No data found. Exiting.")
        return

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['wd']))
    criterion = nn.BCEWithLogitsLoss()

    # Loop
    epochs = cfg['training']['epochs']
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Val
        model.eval()
        val_loss = 0
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_iou = total_iou / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val IoU: {avg_iou:.4f}")
        
        # Log
        logger.log_epoch(epoch, {'train_loss': avg_loss, 'val_loss': avg_val_loss, 'val_iou': avg_iou})
        
        # Save Best (Maximizing IoU)
        logger.save_model(model, avg_iou, metric_name='iou', mode='max')

if __name__ == '__main__':
    main()

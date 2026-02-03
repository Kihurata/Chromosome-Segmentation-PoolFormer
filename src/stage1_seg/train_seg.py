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
        
        # Encoder channels - Determine dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 320, 320)
            if next(self.backbone.parameters()).is_cuda:
                dummy = dummy.cuda()
            # Handle potential device mismatch if backbone not on device yet (init usually cpu)
            
            feats = self.backbone(dummy)
            enc_channels = [f.shape[1] for f in feats]
            print(f"[INFO] Backbone channels found: {enc_channels}")
            
        if len(enc_channels) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(enc_channels)}") 
        
        # Decoder
        # c4=384, c3=192, c2=192, c1=96 (based on log)
        
        # UP1: c4 -> up -> cat(c3) -> conv
        self.up1 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], 2, stride=2)
        # Input to conv1 is cat(c3, up(c4)) = enc_channels[2] + enc_channels[2] = 2 * enc_channels[2]
        self.conv1 = DoubleConv(enc_channels[2] * 2, enc_channels[2]) 
        
        # UP2: c3 -> up -> cat(c2) -> conv
        self.up2 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], 2, stride=2)
        # Input to conv2 is cat(c2, up(c3)) = enc_channels[1] + enc_channels[1] = 2 * enc_channels[1]
        self.conv2 = DoubleConv(enc_channels[1] * 2, enc_channels[1]) 
        
        # UP3: c2 -> up -> cat(c1) -> conv
        self.up3 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], 2, stride=2)
        # Input to conv3 is cat(c1, up(c2)) = enc_channels[0] + enc_channels[0] = 2 * enc_channels[0]
        self.conv3 = DoubleConv(enc_channels[0] * 2, enc_channels[0])
        
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
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        # Structure: root_dir / split / image / *.png
        #            root_dir / split / mask / *.png
        # split can be 'train' or 'val'. If flat structure, handle accordingly.
        
        # Check if 'train' exists in root_dir, if so use it.
        if os.path.exists(os.path.join(root_dir, split)):
            work_dir = os.path.join(root_dir, split)
        else:
            # Maybe root_dir IS the split folder or flat
            work_dir = root_dir

        img_dir = os.path.join(work_dir, 'image') # Changed from 'images' to 'image' based on directory listing
        mask_dir = os.path.join(work_dir, 'mask') # Changed from 'masks' to 'mask'
        
        if os.path.exists(img_dir):
            files = sorted(os.listdir(img_dir))
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(img_dir, f))
                    # Assuming mask has same name or similar
                    self.masks.append(os.path.join(mask_dir, f))
        else:
            print(f"[WARN] Image directory not found: {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"[ERR] Loading {img_path}: {e}")
            # Return dummy or handle error
            return torch.zeros((3, 320, 320)), torch.zeros((1, 320, 320))

        if self.transform:
            # Check if transform supports both args (JointTransform)
            try:
                # Assuming JointTransform or similar callable accepting (image, mask)
                image, mask = self.transform(image, mask)
                # Ensure mask is correct shape/type if transform didn't handle it fully 
                # (transforms.py JointTransform returns tensors, so we might skip manual conversion below)
            except TypeError:
                # Fallback for standard torchvision transforms (image only)
                image = self.transform(image)
                # Resize mask to simple match input size (assumed 320 from config)
                target_size = image.shape[1:]
                mask = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
                mask = np.array(mask)
                mask = (mask > 128).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


# ================== LOSS ==================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Dice
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return bce + (1 - dice)

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
    
    backbone_type = cfg['model']['backbone']
    if backbone_type == 'm36':
        backbone_name = 'poolformer-m36_3rdparty_32xb128_in1k'
    else:
        backbone_name = f"poolformer_{backbone_type}_3rdparty_in1k"

    model = PoolFormerUNet(
        backbone_name=backbone_name, 
        pretrained_path=cfg['model']['pretrained']
    ).to(device)
    
    # Data & Augmentation
    try:
        from transforms import JointTransform
        # Augmentation for training
        train_transform = JointTransform(use_aug=True)
        # No aug for val, but need normalization/tensor conversion
        val_transform = JointTransform(use_aug=False)
    except ImportError:
        print("[WARN] transform.py not found using simple transforms")
        # Fallback
        train_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = train_transform
    
    # Initialize datasets using the new split logic
    train_dataset = SegmentationDataset(cfg['training']['data_root'], split='train', transform=train_transform)
    val_dataset = SegmentationDataset(cfg['training']['data_root'], split='val', transform=val_transform)
    
    if len(train_dataset) == 0:
        print("[WARN] No training data found.")
        return
        
    if len(val_dataset) == 0:
        print("[INFO] No validation data found. Splitting training data 80/20.")
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print(f"[INFO] Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['wd']))
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=float(cfg['training'].get('min_lr', 1e-6)))
    
    # Loss
    criterion = DiceBCELoss()
    
    # MP
    use_amp = cfg['training'].get('amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Loop
    epochs = cfg['training']['epochs']
    patience = cfg['training'].get('patience', 5)
    best_iou = 0.0
    counter = 0 # Early stopping counter
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Val
        model.eval()
        val_loss = 0
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
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
        
        # Step Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs} [LR: {current_lr:.2e}] - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val IoU: {avg_iou:.4f}")
        
        # Log
        logger.log_epoch(epoch, {'train_loss': avg_loss, 'val_loss': avg_val_loss, 'val_iou': avg_iou, 'lr': current_lr})
        
        # Save Best & Early Stopping
        is_best = logger.save_model(model, avg_iou, metric_name='iou', mode='max')
        
        if is_best:
            best_iou = avg_iou
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}. Best IoU: {best_iou:.4f}")
                break

if __name__ == '__main__':
    main()

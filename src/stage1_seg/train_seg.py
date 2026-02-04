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
    def __init__(self, backbone_name, pretrained_path, num_classes=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Load backbone
        self.backbone = get_model(backbone_name, pretrained=pretrained_path, backbone=dict(out_indices=(0, 1, 2, 3)))
        if hasattr(self.backbone, 'backbone'):
            self.backbone = self.backbone.backbone
        
        # Encoder channels - Determine dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 320, 320)
            if next(self.backbone.parameters()).is_cuda:
                dummy = dummy.cuda()
            
            feats = self.backbone(dummy)
            enc_channels = [f.shape[1] for f in feats]
            print(f"[INFO] Backbone channels found: {enc_channels}")
            
        if len(enc_channels) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(enc_channels)}") 
        
        # Decoder
        # c4=384, c3=192, c2=192, c1=96
        
        # UP1
        self.up1 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], 2, stride=2)
        self.conv1 = DoubleConv(enc_channels[2] * 2, enc_channels[2]) 
        
        # UP2
        self.up2 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], 2, stride=2)
        self.conv2 = DoubleConv(enc_channels[1] * 2, enc_channels[1]) 
        
        # UP3
        self.up3 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], 2, stride=2)
        self.conv3 = DoubleConv(enc_channels[0] * 2, enc_channels[0])
        
        self.final_conv = nn.Conv2d(enc_channels[0], num_classes, 1) 
        
        # Deep Supervision Heads
        if self.deep_supervision:
            # Aux head at conv1 (Stride 16)
            self.ds_head1 = nn.Conv2d(enc_channels[2], num_classes, 1)
            # Aux head at conv2 (Stride 8)
            self.ds_head2 = nn.Conv2d(enc_channels[1], num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        
        feats = self.backbone(x) 
        c1, c2, c3, c4 = feats
        
        # Decode
        x = self.up1(c4)
        if x.shape[2:] != c3.shape[2:]:
            x = nn.functional.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1(x)
        feat_aux1 = x # Output of Block 1
        
        x = self.up2(x)
        if x.shape[2:] != c2.shape[2:]:
            x = nn.functional.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)
        feat_aux2 = x # Output of Block 2
        
        x = self.up3(x)
        if x.shape[2:] != c1.shape[2:]:
            x = nn.functional.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.conv3(x)
        
        # Final
        out = self.final_conv(x)
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision and self.training:
            # Upsample aux outputs
            aux1 = self.ds_head1(feat_aux1)
            aux1 = nn.functional.interpolate(aux1, size=input_size, mode='bilinear', align_corners=False)
            
            aux2 = self.ds_head2(feat_aux2)
            aux2 = nn.functional.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)
            
            return {'out': out, 'aux1': aux1, 'aux2': aux2}
            
        return out

# ================== DATASET ==================
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        # Structure: root_dir / split / image / *.png
        #            root_dir / split / mask / *.png
        
        # Check if 'train' exists in root_dir, if so use it.
        if os.path.exists(os.path.join(root_dir, split)):
            work_dir = os.path.join(root_dir, split)
        else:
            # Maybe root_dir IS the split folder or flat
            work_dir = root_dir

        img_dir = os.path.join(work_dir, 'image')
        mask_dir = os.path.join(work_dir, 'mask')
        
        if os.path.exists(img_dir):
            files = sorted(os.listdir(img_dir))
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(img_dir, f))
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
            return torch.zeros((3, 320, 320)), torch.zeros((1, 320, 320))

        if self.transform:
            try:
                image, mask = self.transform(image, mask)
            except TypeError:
                image = self.transform(image)
                target_size = image.shape[1:]
                mask = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
                mask = np.array(mask)
                mask = (mask > 128).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

# ================== LOSS ==================
class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, gamma=2.0, alpha=0.25):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE / Focal logic
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss).mean()
        
        # Dice
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs_sigmoid.sum() + targets.sum() + smooth)  
        
        return focal_loss + (1 - dice)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage1_seg.yaml')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    logger = ExperimentLogger(cfg)
    print(f"[INFO] Stage 1 (PoolFormer) Experiment: {logger.exp_dir}")
    
    # Save transforms
    import shutil
    src_transforms = os.path.join(os.path.dirname(__file__), 'transforms.py')
    dst_transforms = os.path.join(logger.exp_dir, 'transforms.py')
    if os.path.exists(src_transforms):
        shutil.copy(src_transforms, dst_transforms)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    backbone_type = cfg['model']['backbone']
    if backbone_type == 'm36':
        backbone_name = 'poolformer-m36_3rdparty_32xb128_in1k'
    else:
        backbone_name = f"poolformer_{backbone_type}_3rdparty_in1k"

    deep_sup = cfg['training'].get('deep_supervision', False)
    print(f"[INFO] Deep Supervision: {deep_sup}")

    model = PoolFormerUNet(
        backbone_name=backbone_name, 
        pretrained_path=cfg['model']['pretrained'],
        deep_supervision=deep_sup
    ).to(device)
    
    # Data ... [Same as before] ...
    input_size = cfg['model'].get('input_size', 512)
    try:
        from transforms import JointTransform
        train_transform = JointTransform(use_aug=True, size=input_size)
        val_transform = JointTransform(use_aug=False, size=input_size)
    except ImportError:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = train_transform
    
    train_dataset = SegmentationDataset(cfg['training']['data_root'], split='train', transform=train_transform)
    val_dataset = SegmentationDataset(cfg['training']['data_root'], split='val', transform=val_transform)
    
    if len(train_dataset) == 0: return
    if len(val_dataset) == 0:
        val_size = int(0.2 * len(train_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['wd']))
    
    # Scheduler with Warmup
    max_epochs = cfg['training']['epochs']
    warmup_epochs = cfg['training'].get('warmup_epochs', 0)
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=float(cfg['training'].get('min_lr', 1e-6)))
    
    if warmup_epochs > 0:
        print(f"[INFO] Using Linear Warmup for {warmup_epochs} epochs")
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = cosine_scheduler
    
    criterion = DiceFocalLoss(gamma=2.0, alpha=0.25)
    
    use_amp = cfg['training'].get('amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    patience = cfg['training'].get('patience', 5)
    best_iou = 0.0
    counter = 0
    use_tta = cfg.get('inference', {}).get('tta', False)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_dict = model(imgs) # might be dict or tensor
                
                if isinstance(out_dict, dict):
                    # Deep Supervision Loss
                    loss_main = criterion(out_dict['out'], masks)
                    loss_aux1 = criterion(out_dict['aux1'], masks)
                    loss_aux2 = criterion(out_dict['aux2'], masks)
                    loss = loss_main + 0.5 * loss_aux1 + 0.25 * loss_aux2
                else:
                    loss = criterion(out_dict, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if use_tta:
                        # [TTA Logic same as before]
                        # 1. Original
                        out_orig = model(imgs)
                        if isinstance(out_orig, dict): out_orig = out_orig['out'] # Validation always uses final
                        pred = torch.sigmoid(out_orig)
                        
                        # 2. HFlip
                        imgs_hf = torch.flip(imgs, [3])
                        out_hf = model(imgs_hf)
                        if isinstance(out_hf, dict): out_hf = out_hf['out']
                        pred += torch.flip(torch.sigmoid(out_hf), [3])
                        
                        # 3. VFlip
                        imgs_vf = torch.flip(imgs, [2])
                        out_vf = model(imgs_vf)
                        if isinstance(out_vf, dict): out_vf = out_vf['out']
                        pred += torch.flip(torch.sigmoid(out_vf), [2])
                        
                        pred /= 3.0
                        loss = criterion(out_orig, masks)
                        outputs = out_orig 
                    else:
                        outputs = model(imgs)
                        if isinstance(outputs, dict): outputs = outputs['out']
                        loss = criterion(outputs, masks)
                        pred = torch.sigmoid(outputs)
                
                val_loss += loss.item()
                preds_bin = (pred > 0.5).float()
                intersection = (preds_bin * masks).sum()
                union = preds_bin.sum() + masks.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Epoch {epoch}/{max_epochs} [LR: {current_lr:.2e}] - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val IoU: {avg_iou:.4f}")
        logger.log_epoch(epoch, {'train_loss': avg_loss, 'val_loss': avg_val_loss, 'val_iou': avg_iou, 'lr': current_lr})
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

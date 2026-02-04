import os
import argparse
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from train_seg import PoolFormerUNet  # Import model definition
import matplotlib.pyplot as plt

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay binary mask on image.
    image: RGB numpy array
    mask: Binary numpy array (0 or 1)
    """
    mask = mask > 0.5
    overlay = image.copy()
    overlay[mask] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage1_seg.yaml')
    parser.add_argument('--weights', default=None, help='Path to .pth file. If None, tries to find best model in experiment dir.')
    parser.add_argument('--input_dir', default=None, help='Directory containing images to test. If None, uses val set from config.')
    parser.add_argument('--output_dir', default='test_results_vis', help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Setup
    print(f"[INFO] Loading Model Configuration...")
    backbone_type = cfg['model']['backbone']
    if backbone_type == 'm36':
        backbone_name = 'poolformer-m36_3rdparty_32xb128_in1k'
    else:
        backbone_name = f"poolformer_{backbone_type}_3rdparty_in1k"

    model = PoolFormerUNet(
        backbone_name=backbone_name, 
        pretrained_path=None # No need for pretrained weights when loading checkpoint
    ).to(device)

    # Load Weights
    if args.weights is None:
        # Find best model in experiments
        save_dir = cfg['training'].get('save_dir', 'experiments')
        stage = cfg.get('stage', 'unknown_stage')
        exp_root = os.path.join(save_dir, stage)
        
        # Find latest experiment
        if os.path.exists(exp_root):
             exps = sorted([os.path.join(exp_root, d) for d in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, d))])
             if exps:
                 latest_exp = exps[-1]
                 # Find best model file
                 best_models = [f for f in os.listdir(latest_exp) if f.startswith("best_model") and f.endswith(".pth")]
                 if best_models:
                     args.weights = os.path.join(latest_exp, best_models[0])
                     print(f"[INFO] Automatically found latest best model: {args.weights}")
                 else:
                     # Check for last_model.pth
                     if os.path.exists(os.path.join(latest_exp, 'last_model.pth')):
                         args.weights = os.path.join(latest_exp, 'last_model.pth')
                         print(f"[INFO] Best model not found, using last model: {args.weights}")
    
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights))
        print("[INFO] Model weights loaded successfully.")
    else:
        print(f"[ERR] Weights not found at {args.weights}")
        return

    model.eval()
    
    # Input Data
    input_size = cfg['model'].get('input_size', 512)
    tfs = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Directory to scan
    if args.input_dir:
        img_dir = args.input_dir
        mask_dir = None # Inference only usually, but we try to find mask if possible
    else:
        data_root = cfg['training']['data_root']
        # Try 'val' then 'train'
        if os.path.exists(os.path.join(data_root, 'val', 'image')):
             img_dir = os.path.join(data_root, 'val', 'image')
             mask_dir = os.path.join(data_root, 'val', 'mask')
        elif os.path.exists(os.path.join(data_root, 'train', 'image')):
             img_dir = os.path.join(data_root, 'train', 'image')
             mask_dir = os.path.join(data_root, 'train', 'mask')
        else:
             print("[ERR] Could not determine input directory.")
             return
             
    print(f"[INFO] Testing on images from: {img_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if args.num_samples:
        files = files[:args.num_samples]
        
    for i, f in enumerate(files):
        img_path = os.path.join(img_dir, f)
        
        # Load Image
        raw_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_img.size
        
        # Preprocess
        tensor_img = tfs(raw_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            out = model(tensor_img)
            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()
            
        # Post-process
        pred_np = pred.squeeze().cpu().numpy()
        pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_img = pred_img.resize((orig_w, orig_h), resample=Image.NEAREST)
        pred_np_resized = np.array(pred_img) / 255.0
        
        # Visualization
        raw_cv = np.array(raw_img)
        
        # Overlay Prediction (Green)
        vis_pred = overlay_mask(raw_cv, pred_np_resized, color=(0, 255, 0))
        
        # Check for GT Mask
        vis_gt = None
        if mask_dir:
            mask_path = os.path.join(mask_dir, f)
            if os.path.exists(mask_path):
                gt_mask = Image.open(mask_path).convert("L")
                gt_mask = np.array(gt_mask) > 128
                # Overlay GT (Red)
                vis_gt = overlay_mask(raw_cv, gt_mask, color=(255, 0, 0))
        
        # Combine Side-by-Side
        # Layout: Original | GT (if exists) | Prediction | Overlay
        
        h, w, _ = raw_cv.shape
        spacer = np.zeros((h, 10, 3), dtype=np.uint8) + 255 # White spacer
        
        combined_list = [raw_cv, spacer]
        if vis_gt is not None:
             combined_list.append(vis_gt)
             combined_list.append(spacer)
             
        combined_list.append(vis_pred)
             
        combined = np.hstack(combined_list)
        
        save_path = os.path.join(args.output_dir, f'vis_{f}')
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved result: {save_path}")

    print(f"[INFO] Done. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()

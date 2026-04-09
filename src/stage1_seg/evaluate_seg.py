import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from train_seg import PoolFormerUNet, SegmentationDataset

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate Precision, Recall, F1-Score, Accuracy, IoU for binary segmentation.
    preds: (N, 1, H, W) tensor with sigmoid values
    targets: (N, 1, H, W) tensor with ground truth (0 or 1)
    """
    preds_bin = (preds > threshold).float()
    
    # Pixel-wise metrics
    tp = (preds_bin * targets).sum().item()
    fp = (preds_bin * (1 - targets)).sum().item()
    fn = ((1 - preds_bin) * targets).sum().item()
    tn = ((1 - preds_bin) * (1 - targets)).sum().item()
    
    # Avoid division by zero
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    
    # Một dự đoán được coi là "Đúng mẫu" nếu IoU > 0.5 (ngưỡng thông dụng)
    is_correct = 1 if iou > 0.5 else 0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'IoU': iou,
        'Accuracy': accuracy,
        'Correct': is_correct
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage1_seg.yaml')
    parser.add_argument('--weights', default=None, help='Path to .pth file. If None, tries to find best model.')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Model Setup
    backbone_type = cfg['model']['backbone']
    if backbone_type == 'm36':
        backbone_name = 'poolformer-m36_3rdparty_32xb128_in1k'
    else:
        backbone_name = f"poolformer_{backbone_type}_3rdparty_in1k"

    deep_sup = cfg.get('training', {}).get('deep_supervision', False)
    model = PoolFormerUNet(
        backbone_name=backbone_name, 
        pretrained_path=None, 
        deep_supervision=deep_sup
    ).to(device)

    # Load Weights
    exp_dir = os.path.dirname(args.config) if 'experiments' in args.config else None
    
    if args.weights is None and exp_dir:
        # Find best model in the same dir as config
        best_models = [f for f in os.listdir(exp_dir) if f.startswith("best_model") and f.endswith(".pth")]
        if best_models:
            args.weights = os.path.join(exp_dir, best_models[0])
        elif os.path.exists(os.path.join(exp_dir, 'last_model.pth')):
            args.weights = os.path.join(exp_dir, 'last_model.pth')
    
    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Loading Weights: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        if exp_dir is None:
             exp_dir = os.path.dirname(args.weights)
    else:
        print(f"[ERR] Weights not found. Please provide --weights")
        return

    model.eval()

    # Dataset Setup
    input_size = cfg['model'].get('input_size', 512)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_root = cfg['training']['data_root']
    val_dataset = SegmentationDataset(data_root, split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[INFO] Evaluating on {len(val_dataset)} images...")

    all_results = []
    
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(val_loader)):
            img_name = os.path.basename(val_dataset.images[i])
            imgs, masks = imgs.to(device), masks.to(device)
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            out = model(imgs)
            if isinstance(out, dict): out = out['out']
            pred = torch.sigmoid(out)
            end_time.record()
            
            torch.cuda.synchronize()
            exec_time = start_time.elapsed_time(end_time) / 1000.0 # to seconds
            
            metrics = calculate_metrics(pred, masks)
            metrics['Image'] = img_name
            metrics['Model'] = 'PoolFormer'
            metrics['Time (s)'] = exec_time
            all_results.append(metrics)

    # DataFrame processing
    df = pd.DataFrame(all_results)
    
    # Reorder columns to match Benchmark style
    cols = ['Image', 'Model', 'Time (s)', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Accuracy', 'Correct']
    df = df[cols]
    
    # Save details
    output_path_details = os.path.join(exp_dir, 'evaluation_details_poolformer.csv')
    df.to_csv(output_path_details, index=False)
    
    # Create summary (Mean for metrics, Sum for Correct)
    summary_mean = df.groupby('Model')[['Time (s)', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Accuracy']].mean().reset_index()
    summary_sum = df.groupby('Model')['Correct'].sum().reset_index()
    summary = pd.merge(summary_mean, summary_sum, on='Model')
    
    # Rename 'Correct' to 'Correct Samples' in summary
    summary = summary.rename(columns={'Correct': 'Dự đoán đúng'})
    
    output_path_summary = os.path.join(exp_dir, 'evaluation_summary_poolformer.csv')
    summary.to_csv(output_path_summary, index=False)

    total_images = len(df)
    correct_count = summary['Dự đoán đúng'].values[0]

    # Display results
    print("\n" + "="*50)
    print(" POOLFORMER EVALUATION SUMMARY ")
    print("="*50)
    print(summary.to_string(index=False))
    print("="*50)
    print(f"✅ Details saved to: {output_path_details}")
    print(f"✅ Summary saved to: {output_path_summary}")

if __name__ == '__main__':
    main()

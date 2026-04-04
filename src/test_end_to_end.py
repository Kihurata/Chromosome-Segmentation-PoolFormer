import os
import cv2
import argparse
import numpy as np
import yaml
import torch
from mmpretrain import get_model, init_model
from mmpretrain.apis import ImageClassificationInferencer

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Testing on Stage 1 Output with Stage 2 Model")
    parser.add_argument('--image', type=str, required=True, help="Path to large image (background removed by Stage 1)")
    parser.add_argument('--config', type=str, default='configs/stage2_cls.yaml', help="Path to Stage 2 config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to Stage 2 checkpoint (.pth)")
    parser.add_argument('--out-dir', type=str, default='test_results_vis/end_to_end', help="OutputDir")
    parser.add_argument('--padding', type=int, default=10, help="Padding around bounding box")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load Stage 2 Config
    with open(args.config, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    classes = yaml_cfg['model']['classes']
    
    # 1. Initialize Stage 2 Model using Inferencer
    print(f"[INFO] Loading Stage 2 Model from {args.checkpoint}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build complete config to initialize model properly
    from stage2_cls.train_cluster import build_mm_config
    cfg = build_mm_config(yaml_cfg, args.out_dir, 'dummy', 'dummy')
    model = init_model(cfg, args.checkpoint, device=device)
    model.eval()
    
    # MMLab Inferencer
    inferencer = ImageClassificationInferencer(model, classes=classes)

    # 2. Read Large Image (Already background-removed from Stage 1)
    print(f"[INFO] Processing image: {args.image}")
    original_img = cv2.imread(args.image)
    if original_img is None:
        print(f"[ERROR] Could not read image at {args.image}")
        return
        
    vis_img = original_img.copy()
    
    # Convert to grayscale and threshold to get mask of chromosomes
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find Contours (the individual chromosomes or clusters)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Found {len(contours)} disconnected components (chromosomes).")

    # Define colors for classes
    # Assuming classes: ['single', 'overlapping', 'touching']
    colors = {
        'single': (0, 255, 0),        # Green
        'overlapping': (0, 0, 255),   # Red
        'touching': (0, 165, 255)     # Orange
    }

    # 3. Process each contour
    for i, contour in enumerate(contours):
        # Filter too small contours
        if cv2.contourArea(contour) < 50:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        pad = args.padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(original_img.shape[1], x + w + pad)
        y2 = min(original_img.shape[0], y + h + pad)
        
        # Crop patch
        patch = original_img[y1:y2, x1:x2]
        
        # Determine classification
        # We can pass the patch directly to inference (as numpy array)
        try:
            result = inferencer(patch)[0]
            pred_class = result['pred_class']
            pred_score = result['pred_score']
            
            # Predict color
            color = colors.get(pred_class, (255, 255, 255))
            
            # Draw bounding box on visualization image
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background for better text visibility
            label = f"{pred_class} ({pred_score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        except Exception as e:
            print(f"[WARNING] Inference failed on patch {i}: {e}")
            
    # 4. Save Final Visualization
    base_name = os.path.basename(args.image)
    save_path = os.path.join(args.out_dir, f"classified_{base_name}")
    cv2.imwrite(save_path, vis_img)
    print(f"\n[SUCCESS] Saved classified image to: {save_path}")

if __name__ == '__main__':
    main()

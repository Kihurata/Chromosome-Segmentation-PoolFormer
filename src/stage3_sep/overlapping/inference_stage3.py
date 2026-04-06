import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import warnings
warnings.filterwarnings("ignore")

from train_overlapping_stage3 import MultiHeadOverlapBoundaryNet

def get_transform(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def clean_mask(mask, morph_size=3):
    """Làm sạch các nhiễu nhỏ li ti trên mask bằng Morphological Opening"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cleaned

def predict_with_tta(model, img_tensor):
    """
    Áp dụng Test-Time Augmentation bằng cách dự đoán trên nhiều bản xoay lật 
    sau đó lấy trung bình để khử nhiễu vùng Overlap.
    Dùng 4 phép: Identity, Horizontal Flip, Vertical Flip, Rot180
    """
    fg_probs, ov_probs, bd_probs = [], [], []

    # 1. Identity
    def aug_id(x): return x
    def deaug_id(x): return x

    # 2. Horizontal Flip
    def aug_hf(x): return torch.flip(x, dims=[3])
    def deaug_hf(x): return torch.flip(x, dims=[3])

    # 3. Vertical Flip
    def aug_vf(x): return torch.flip(x, dims=[2])
    def deaug_vf(x): return torch.flip(x, dims=[2])

    # 4. Rotate 180 (vflip + hflip)
    def aug_rot180(x): return torch.flip(x, dims=[2, 3])
    def deaug_rot180(x): return torch.flip(x, dims=[2, 3])

    transforms = [(aug_id, deaug_id), (aug_hf, deaug_hf), (aug_vf, deaug_vf), (aug_rot180, deaug_rot180)]

    for aug, deaug in transforms:
        aug_img = aug(img_tensor)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fg_logits, ov_logits, bd_logits = model(aug_img)
            
        fg_prob = torch.sigmoid(fg_logits)
        ov_prob = torch.sigmoid(ov_logits)
        bd_prob = torch.sigmoid(bd_logits)
        
        fg_probs.append(deaug(fg_prob).squeeze().cpu().numpy())
        ov_probs.append(deaug(ov_prob).squeeze().cpu().numpy())
        bd_probs.append(deaug(bd_prob).squeeze().cpu().numpy())

    return np.mean(fg_probs, axis=0), np.mean(ov_probs, axis=0), np.mean(bd_probs, axis=0)

def postprocess_watershed(image_np, fg_mask, bd_mask, ov_mask):
    """
    Sử dụng Marker-controlled Watershed để tách các cụm nhiễm sắc thể dính nhau
    dựa trên dự đoán Boundary và Overlap.
    """
    # 1. Các vùng chắc chắn là Foreground: Có Foreground, KHÔNG nằm trên Biên và Vùng chồng lấn
    sure_fg = (fg_mask > 0) & (bd_mask == 0) & (ov_mask == 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # 2. Các vùng chắc chắn là Background:
    sure_bg = (fg_mask == 0).astype(np.uint8)
    
    # 3. Vùng chưa xác định (Unknown region - bao gồm cả biên và overlap)
    unknown = cv2.subtract(fg_mask, sure_fg)
    
    # 4. Gắn nhãn các vùng Marker
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Cộng 1 để sure background là 1 (chứ không phải 0)
    markers = markers + 1
    
    # Đánh dấu vùng chưa biết là 0 để Watershed tính toán
    markers[unknown == 1] = 0
    
    # Áp dụng Watershed algorithm
    img_8u = image_np.copy()
    markers = cv2.watershed(img_8u, markers)
    
    # Visualization: tạo các màu ngẫu nhiên cho từng vật thể
    instance_mask = np.zeros_like(img_8u)
    unique_markers = np.unique(markers)
    for m in unique_markers:
        if m <= 1: # 1 là background, -1 là boundary chia cắt
            continue
        color_instance = np.random.randint(50, 255, size=(3,))
        instance_mask[markers == m] = color_instance
    
    # Trộn ảnh gốc với màu instance (Blend)
    blended = cv2.addWeighted(img_8u, 0.5, instance_mask, 0.5, 0)
    blended[markers == -1] = [255, 0, 0] # Đường biên cắt màu đỏ
    
    return blended, markers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage3_sep.yaml")
    parser.add_argument("--weights", type=str, default="experiments/stage3_overlapping_multitask_best_checkpoint.pth")
    parser.add_argument("--input_dir", type=str, default="C:/data/stage3_separation/overlapping/val/images")
    parser.add_argument("--output_dir", type=str, default="test_results_vis/stage3")
    parser.add_argument("--num_samples", type=int, default=5, help="Số lượng ảnh để test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    img_size = cfg['model']['input_size']
    fg_thresh = float(cfg['loss'].get('fg_threshold', 0.5))
    ov_thresh = float(cfg['loss'].get('overlap_threshold', 0.45))
    bd_thresh = float(cfg['loss'].get('boundary_threshold', 0.35))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Model: PoolFormer-M36 Unet...")
    model = MultiHeadOverlapBoundaryNet(pretrained_weights=None).to(device)
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        fallback_weight = "experiments/stage3_overlapping_multitask_best.pth"
        if os.path.exists(fallback_weight):
            args.weights = fallback_weight
        else:
            print(f"Lỗi: Không tìm thấy file weights {args.weights}")
            return

    print(f"Loading Weights: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()

    transform = get_transform(img_size)

    input_paths = list(Path(args.input_dir).glob("*.png")) + list(Path(args.input_dir).glob("*.jpg"))
    random.shuffle(input_paths)
    input_paths = input_paths[:args.num_samples]

    if not input_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục {args.input_dir}")
        return

    print(f"Đang tiến hành chạy Inference và Watershed trên {len(input_paths)} ảnh...")

    for img_path in input_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý (Transform)
        augmented = transform(image=img_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(device)

        # Inference with TTA
        fg_prob, ov_prob, bd_prob = predict_with_tta(model, img_tensor)

        # Áp dụng ngưỡng (Threshold)
        fg_mask = (fg_prob > fg_thresh).astype(np.uint8)
        ov_mask = (ov_prob > ov_thresh).astype(np.uint8)
        bd_mask = (bd_prob > bd_thresh).astype(np.uint8)

        # Làm sạch nhiễu cho mask
        fg_mask = clean_mask(fg_mask, morph_size=3)
        ov_mask = clean_mask(ov_mask, morph_size=3)
        bd_mask = clean_mask(bd_mask, morph_size=3)

        # Resize lại đúng kích thước ảnh gốc cho Watershed
        orig_h, orig_w = img_rgb.shape[:2]
        fg_mask_rs = cv2.resize(fg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ov_mask_rs = cv2.resize(ov_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        bd_mask_rs = cv2.resize(bd_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Phân tách cụm dính bằng Watershed
        separated_img, markers = postprocess_watershed(img_rgb, fg_mask_rs, bd_mask_rs, ov_mask_rs)

        # ----- Trực quan hóa kết quả (Matplotlib) -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Kết quả Tách (Separation) & Dự đoán: {img_path.name}', fontsize=16)

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Ảnh gốc (Original Image)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(separated_img)
        axes[0, 1].set_title('Tách thể dính (Watershed)')
        axes[0, 1].axis('off')

        # Combine prob masks: Đỏ: Đường biên, Lục: Khu vực chồng lấn, Lam: Nhiễm sắc thể
        combined_prob = np.zeros((img_size, img_size, 3), dtype=np.float32)
        combined_prob[..., 0] = bd_prob  # R
        combined_prob[..., 1] = ov_prob  # G
        combined_prob[..., 2] = fg_prob  # B
        
        axes[0, 2].imshow(combined_prob)
        axes[0, 2].set_title('Probability Maps\n(Đỏ:Biên, Lục:Chồng chéo, Lam:Vật thể)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(fg_mask, cmap='gray')
        axes[1, 0].set_title('Mặt nạ Vật thể (Foreground)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ov_mask, cmap='gray')
        axes[1, 1].set_title('Mặt nạ Chồng chéo (Overlap)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(bd_mask, cmap='gray')
        axes[1, 2].set_title('Mặt nạ Đường biên (Boundary)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = out_dir / f"result_{img_path.name}"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Đã lưu kết quả tại: {save_path}")

if __name__ == '__main__':
    main()

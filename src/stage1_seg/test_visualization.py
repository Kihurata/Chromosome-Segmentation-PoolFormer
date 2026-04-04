import os
import argparse
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from train_seg import PoolFormerUNet  # Đảm bảo file này có trong cùng folder hoặc python path

def main():
    parser = argparse.ArgumentParser(description="Trích xuất ảnh tách nền từ Stage 1")
    parser.add_argument('--config', default='configs/stage1_seg.yaml')
    parser.add_argument('--weights', default=None, help='Đường dẫn tới file .pth. Nếu để trống sẽ tìm trong experiments.')
    parser.add_argument('--input_dir', default=None, help='Thư mục ảnh gốc. Nếu trống sẽ lấy từ config val set.')
    parser.add_argument('--output_dir', default='data/stage1_output_segmented', help='Thư mục lưu ảnh tách nền')
    parser.add_argument('--num_samples', type=int, default=None, help='Số lượng ảnh muốn xử lý (để trống là tất cả)')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Setup Model
    backbone_type = cfg['model']['backbone']
    backbone_name = 'poolformer-m36_3rdparty_32xb128_in1k' if backbone_type == 'm36' else f"poolformer_{backbone_type}_3rdparty_in1k"
    deep_sup = cfg.get('training', {}).get('deep_supervision', False)
    
    model = PoolFormerUNet(
        backbone_name=backbone_name, 
        pretrained_path=None, 
        deep_supervision=deep_sup
    ).to(device)

    # 3. Load Weights (Tự động tìm nếu không chỉ định)
    if args.weights is None:
        save_dir = cfg['training'].get('save_dir', 'experiments')
        stage = cfg.get('stage', 'stage1_seg')
        exp_root = os.path.join(save_dir, stage)
        if os.path.exists(exp_root):
             exps = sorted([os.path.join(exp_root, d) for d in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, d))])
             if exps:
                 latest_exp = exps[-1]
                 best_models = [f for f in os.listdir(latest_exp) if f.startswith("best_model") and f.endswith(".pth")]
                 if best_models:
                     args.weights = os.path.join(latest_exp, best_models[0])
                 elif os.path.exists(os.path.join(latest_exp, 'last_model.pth')):
                     args.weights = os.path.join(latest_exp, 'last_model.pth')
    
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[INFO] Đã tải trọng số từ: {args.weights}")
    else:
        print(f"[ERR] Không tìm thấy weight tại {args.weights}. Vui lòng kiểm tra lại.")
        return

    model.eval()
    
    # 4. Data Setup
    input_size = cfg['model'].get('input_size', 512)
    tfs = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.input_dir:
        img_dir = args.input_dir
    else:
        data_root = cfg['training']['data_root']
        img_dir = os.path.join(data_root, 'val', 'image')
             
    if not os.path.exists(img_dir):
        print(f"[ERR] Thư mục đầu vào không tồn tại: {img_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if args.num_samples:
        files = files[:args.num_samples]
        
    print(f"[INFO] Bắt đầu tách nền cho {len(files)} ảnh...")

    # 5. Inference & Save
    for f in files:
        img_path = os.path.join(img_dir, f)
        raw_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_img.size
        
        # Tiền xử lý
        tensor_img = tfs(raw_img).unsqueeze(0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            out = model(tensor_img)
            if isinstance(out, dict): out = out['out']
            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()
            
        # Hậu xử lý Mask
        pred_np = pred.squeeze().cpu().numpy()
        # Resize mask về kích thước gốc
        mask_resized = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Áp mask vào ảnh gốc (Tách nền)
        raw_cv = np.array(raw_img) # RGB
        segmented = raw_cv.copy()
        segmented[mask_resized == 0] = 0 # Chuyển nền về đen
        
        # Lưu kết quả
        save_path = os.path.join(args.output_dir, f)
        cv2.imwrite(save_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
        
    print(f"[SUCCESS] Đã lưu {len(files)} ảnh tách nền vào: {args.output_dir}")

if __name__ == '__main__':
    main()

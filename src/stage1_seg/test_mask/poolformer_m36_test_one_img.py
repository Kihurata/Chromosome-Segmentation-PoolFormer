# -*- coding: utf-8 -*-
"""
Batch test MMSeg cho mô hình đã train:
- Duyệt tất cả ảnh PNG trong các folder con của FOLDER_ROOT
- Mỗi ảnh tạo 1 thư mục riêng để lưu kết quả (binary, color, overlay)
"""

import os
import os.path as osp
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import mmcv

from mmengine import Config
from mmengine.runner import CheckpointLoader, load_checkpoint
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
from mmseg.apis import inference_model

# ====== CẤU HÌNH ======
CKPT_PATH   = r"D:\NCKH\runs\poolformer_m36_binseg\epoch_150.pth"
FOLDER_ROOT = r"D:\NCKH\Autokary2022_1600x1600\test_labelme"
SAVE_ROOT   = r"D:\NCKH\test_results_maskonly\test_mask_15_8"
# ======================

os.makedirs(SAVE_ROOT, exist_ok=True)
register_all_modules()

def load_cfg_from_ckpt(ckpt_path: str) -> Config:
    ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location='cpu')
    meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
    cfg_in_meta = meta.get('cfg', None)
    if cfg_in_meta is None:
        raise RuntimeError("Checkpoint không có 'meta.cfg'.")
    if isinstance(cfg_in_meta, dict):
        cfg = Config(cfg_in_meta)
    elif isinstance(cfg_in_meta, str):
        cfg = Config.fromstring(cfg_in_meta, file_format='.py')
    else:
        cfg = Config(cfg_in_meta)

    if not hasattr(cfg, 'test_pipeline'):
        cfg.test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='PackSegInputs')
        ]
    if 'test_cfg' not in cfg.model or cfg.model.get('test_cfg', None) is None:
        cfg.model['test_cfg'] = dict(mode='whole')
    else:
        cfg.model['test_cfg']['mode'] = 'whole'

    if not hasattr(cfg, 'dataset_meta'):
        cfg.dataset_meta = dict(
            classes=['background', 'foreground'],
            palette=[[0, 0, 0], [0, 255, 0]]
        )
    return cfg

def build_model(cfg: Config, ckpt_path: str):
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model.cfg = cfg
    model.dataset_meta = getattr(cfg, 'dataset_meta', {})
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def colorize_mask(mask: np.ndarray, palette):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    max_lab = int(mask.max())
    for lab in range(min(max_lab, len(palette)-1) + 1):
        out[mask == lab] = np.array(palette[lab], dtype=np.uint8)
    return out

def overlay_image(img_bgr: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5):
    mask_bgr = mmcv.rgb2bgr(mask_rgb)
    over = (img_bgr.astype(np.float32) * (1 - alpha) +
            mask_bgr.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return over

def main():
    cfg = load_cfg_from_ckpt(CKPT_PATH)
    model = build_model(cfg, CKPT_PATH)
    palette = model.dataset_meta.get('palette', [[0, 0, 0], [0, 255, 0]])
    classes = model.dataset_meta.get('classes', ['background', 'foreground'])

    # Duyệt toàn bộ folder con
    for root, dirs, files in os.walk(FOLDER_ROOT):
        for file in files:
            if file.lower().endswith('.png'):
                img_path = osp.join(root, file)
                img_name = osp.splitext(file)[0]

                save_dir_img = osp.join(SAVE_ROOT, img_name)
                os.makedirs(save_dir_img, exist_ok=True)

                # Inference
                result = inference_model(model, img_path)
                pred = result.pred_sem_seg.data
                if isinstance(pred, torch.Tensor):
                    pred = pred.squeeze().detach().cpu().numpy()
                else:
                    pred = np.array(pred)

                # Lưu mask nhị phân
                if pred.max() <= 1:
                    bin_mask = (pred > 0).astype(np.uint8) * 255
                    mmcv.imwrite(bin_mask, osp.join(save_dir_img, 'pred_mask_binary.png'))

                # Lưu mask màu
                color_mask = colorize_mask(pred.astype(np.int32), palette)
                mmcv.imwrite(color_mask[:, :, ::-1], osp.join(save_dir_img, 'pred_mask_color.png'))

                # Lưu overlay
                img_bgr = mmcv.imread(img_path)
                overlay = overlay_image(img_bgr, color_mask, alpha=0.5)
                mmcv.imwrite(overlay, osp.join(save_dir_img, 'overlay.png'))

                print(f"Done: {img_path}")

    print(f"Tất cả ảnh đã xử lý. Kết quả lưu tại: {SAVE_ROOT}")
    print(f"Classes: {classes}")
    print(f"Palette: {palette}")

if __name__ == "__main__":
    main()

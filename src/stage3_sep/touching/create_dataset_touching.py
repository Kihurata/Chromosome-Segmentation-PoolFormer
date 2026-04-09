import argparse
import csv
import json
import os
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create Stage 3 Touching Dataset')
    parser.add_argument('--stage1_train_img', default=r"C:\data\stage1_segmentation\train\image")
    parser.add_argument('--stage1_train_mask', default=r"C:\data\stage1_segmentation\train\mask")
    parser.add_argument('--stage1_val_img', default=r"C:\data\stage1_segmentation\val\image")
    parser.add_argument('--stage1_val_mask', default=r"C:\data\stage1_segmentation\val\mask")
    parser.add_argument('--labelme_root', default=r"C:\data_raw\Autokary2022_1600x1600\train_labelme")
    parser.add_argument('--out_dir', default=r"C:\data\stage3_separation\touching")
    parser.add_argument('--min_area', type=int, default=80)
    parser.add_argument('--padding', type=int, default=12)
    parser.add_argument('--erode_kernel', type=int, default=3)
    parser.add_argument('--erode_iters', type=int, default=2, help='Increase to make markers smaller/distinct')
    return parser.parse_args()

def polygon_to_mask(points, h, w):
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def load_instance_masks_from_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    instance_masks = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type", "") == "polygon":
            points = shape.get("points", [])
            if points:
                mask = polygon_to_mask(points, h, w)
                if mask.sum() > 0:
                    instance_masks.append(mask)
    return instance_masks

def is_touching_cluster(inst_masks, overlap_threshold=10):
    if len(inst_masks) < 2: return False
    for a, b in combinations(inst_masks, 2):
        if int((a & b).sum()) > overlap_threshold:
            return False
    return True

def build_json_lookup(labelme_root):
    return {p.stem: p for p in Path(labelme_root).rglob("*.json")}

def build_marker_map(inst_masks, component_mask, kernel_size=3, iters=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    marker = np.zeros_like(component_mask, dtype=np.uint8)
    for inst in inst_masks:
        masked_inst = (inst & component_mask).astype(np.uint8)
        if masked_inst.sum() < 20: continue
        eroded = cv2.erode(masked_inst, kernel, iterations=iters)
        if eroded.sum() == 0: eroded = masked_inst.copy()
        marker = np.maximum(marker, eroded)
    return marker

def process_split(split_name, img_dir, mask_dir, json_lookup, args, manifest_writer):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    out_base = Path(args.out_dir) / split_name
    for sub in ["images", "markers", "foregrounds"]: (out_base / sub).mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    saved = 0
    
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists(): continue
        
        stem = img_path.stem
        json_path = json_lookup.get(stem)
        
        # Nếu không có hoặc có "__" có thể xử lý dự phòng
        if not json_path:
            # Fallback nếu dùng format cũ Folder__File
            if "__" in stem:
                folder_name, base_name = stem.split("__", 1)
                fallback = Path(args.labelme_root) / folder_name / f"{base_name}.json"
                if fallback.exists(): json_path = fallback
        
        if not json_path or not json_path.exists(): continue

        image = np.array(Image.open(img_path).convert("RGB"))
        binary_mask = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        instance_masks = load_instance_masks_from_json(json_path)

        num_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        H, W = binary_mask.shape

        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < args.min_area: continue
            
            comp_mask = (cc_map == comp_id).astype(np.uint8)
            inst_in_comp = [inst for inst in instance_masks if (comp_mask & inst).sum() > 0]

            if len(inst_in_comp) < 2 or not is_touching_cluster(inst_in_comp): continue

            marker_map = build_marker_map(inst_in_comp, comp_mask, args.erode_kernel, args.erode_iters)
            
            x, y, w, h = stats[comp_id, :4]
            x1, y1 = max(0, x - args.padding), max(0, y - args.padding)
            x2, y2 = min(W, x + w + args.padding), min(H, y + h + args.padding)

            crop_rgb = image[y1:y2, x1:x2]
            crop_fg = comp_mask[y1:y2, x1:x2]
            crop_mk = marker_map[y1:y2, x1:x2]

            if (cv2.connectedComponents(crop_mk)[0] - 1) < 2: continue

            save_name = f"{stem}__obj{comp_id:03d}.png"
            Image.fromarray(crop_rgb).save(out_base / "images" / save_name)
            Image.fromarray((crop_fg * 255)).save(out_base / "foregrounds" / save_name)
            Image.fromarray((crop_mk * 255)).save(out_base / "markers" / save_name)
            
            manifest_writer.writerow([split_name, img_path.name, save_name, area, len(inst_in_comp)])
            saved += 1
    return saved

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_lookup = build_json_lookup(args.labelme_root)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source", "crop", "area", "num_inst"])
        train_saved = process_split("train", args.stage1_train_img, args.stage1_train_mask, json_lookup, args, writer)
        val_saved = process_split("val", args.stage1_val_img, args.stage1_val_mask, json_lookup, args, writer)

    print(f"Done! Train: {train_saved}, Val: {val_saved}")

if __name__ == "__main__":
    main()

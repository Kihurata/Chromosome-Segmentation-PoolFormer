import os
import cv2
import csv
import numpy as np
import random
import glob
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from PIL import Image

# =========================================================
# STAGE 3 SYNTHETIC OVERLAPPING DATASET CREATION
# ---------------------------------------------------------
# RGB canvas (345x345)
# -> overlap target
# -> boundary target
# -> foreground target
# =========================================================

TARGET_SIZE = (345, 345)
BOUNDARY_THICKNESS = 2
SAVE_DEBUG = True

def get_mask(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    return binary

def extract_chromosome(img):
    mask = get_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    crop_img = img[y:y+h, x:x+w]
    crop_mask = mask[y:y+h, x:x+w]
    return crop_img, crop_mask

def rotate_bound(image, angle, mask=None):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(image, M, (nW, nH))
    if mask is not None:
        rotated_mask = cv2.warpAffine(mask, M, (nW, nH))
        return rotated, rotated_mask
    return rotated

def check_intersection(canvas_mask, fg_mask, dx, dy):
    h, w = fg_mask.shape[:2]
    bg_h, bg_w = canvas_mask.shape[:2]
    
    y1, y2 = max(0, dy), min(bg_h, dy + h)
    x1, x2 = max(0, dx), min(bg_w, dx + w)
    
    fg_y1, fg_y2 = max(0, -dy), min(h, bg_h - dy)
    fg_x1, fg_x2 = max(0, -dx), min(w, bg_w - dx)
    
    if y1 >= y2 or x1 >= x2:
        return 0
        
    intersect = cv2.bitwise_and(canvas_mask[y1:y2, x1:x2], fg_mask[fg_y1:fg_y2, fg_x1:fg_x2])
    return cv2.countNonZero(intersect)

def paste_image_max_instance(bg, bg_mask, fg_img, fg_mask, x_offset, y_offset):
    h, w = fg_img.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + w)
    
    fg_y1, fg_y2 = max(0, -y_offset), min(h, bg_h - y_offset)
    fg_x1, fg_x2 = max(0, -x_offset), min(w, bg_w - x_offset)
    
    instance_mask_canvas = np.zeros_like(bg_mask, dtype=np.uint8)
    
    if y1 >= y2 or x1 >= x2:
        return bg, bg_mask, instance_mask_canvas
        
    bg_roi = bg[y1:y2, x1:x2]
    fg_roi = fg_img[fg_y1:fg_y2, fg_x1:fg_x2]
    
    alpha_mask = fg_mask[fg_y1:fg_y2, fg_x1:fg_x2] > 0
    if len(bg_roi.shape) == 3:
        alpha_3d = np.expand_dims(alpha_mask, axis=-1)
        bg[y1:y2, x1:x2] = np.where(alpha_3d, np.maximum(bg_roi, fg_roi), bg_roi)
    else:
        bg[y1:y2, x1:x2] = np.where(alpha_mask, np.maximum(bg_roi, fg_roi), bg_roi)
        
    if bg_mask is not None:
        bg_m_roi = bg_mask[y1:y2, x1:x2]
        fg_m_roi = fg_mask[fg_y1:fg_y2, fg_x1:fg_x2]
        bg_mask[y1:y2, x1:x2] = np.maximum(bg_m_roi, fg_m_roi)
        instance_mask_canvas[y1:y2, x1:x2] = fg_m_roi
        
    return bg, bg_mask, instance_mask_canvas

def generate_overlapping_instances(img1, img2, target_size=(345, 345)):
    c1, m1 = extract_chromosome(img1)
    c2, m2 = extract_chromosome(img2)
    
    angle1 = random.uniform(0, 360)
    angle2 = random.uniform(0, 360)
    c1, m1 = rotate_bound(c1, angle1, m1)
    c2, m2 = rotate_bound(c2, angle2, m2)
    
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    cx, cy = target_size[0]//2, target_size[1]//2
    x1 = cx - c1.shape[1]//2 + random.randint(-15, 15)
    y1 = cy - c1.shape[0]//2 + random.randint(-15, 15)
    canvas, canvas_mask, inst1_mask = paste_image_max_instance(canvas, canvas_mask, c1, m1, x1, y1)
    
    max_attempts = 50
    area1 = cv2.countNonZero(m1)
    area2 = cv2.countNonZero(m2)
    min_area = min(area1, area2)
    
    inst2_mask = None
    placed = False
    
    for _ in range(max_attempts):
        x2 = cx - c2.shape[1]//2 + random.randint(-int(c1.shape[1]*0.6), int(c1.shape[1]*0.6))
        y2 = cy - c2.shape[0]//2 + random.randint(-int(c1.shape[0]*0.6), int(c1.shape[0]*0.6))
        
        overlap_pixels = check_intersection(canvas_mask, m2, x2, y2)
        if min_area > 0 and (overlap_pixels / min_area) >= 0.10:
            canvas, canvas_mask, inst2_mask = paste_image_max_instance(canvas, canvas_mask, c2, m2, x2, y2)
            placed = True
            break
            
    if not placed:
        # Fallback to center completely
        x2 = cx - c2.shape[1]//2
        y2 = cy - c2.shape[0]//2
        canvas, canvas_mask, inst2_mask = paste_image_max_instance(canvas, canvas_mask, c2, m2, x2, y2)
        
    return canvas, [inst1_mask, inst2_mask]

def build_overlap_map(inst_masks):
    overlap_map = np.zeros_like(inst_masks[0], dtype=np.uint8)
    for a, b in combinations(inst_masks, 2):
        a_bin = (a > 0).astype(np.uint8)
        b_bin = (b > 0).astype(np.uint8)
        ov = (a_bin & b_bin)
        if int(ov.sum()) > 0:
            overlap_map = np.maximum(overlap_map, ov)
    return overlap_map

def build_boundary_map(inst_masks):
    boundary = np.zeros_like(inst_masks[0], dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for inst in inst_masks:
        bin_inst = (inst > 0).astype(np.uint8)
        eroded = cv2.erode(bin_inst, kernel, iterations=1)
        edge = (bin_inst - eroded).clip(0, 1).astype(np.uint8)
        if BOUNDARY_THICKNESS > 1:
            edge = cv2.dilate(edge, kernel, iterations=BOUNDARY_THICKNESS - 1)
        boundary = np.maximum(boundary, edge)

    return boundary

def save_debug(crop_rgb, crop_fg, crop_overlap, crop_boundary, out_path):
    h, w = crop_fg.shape
    canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)

    a = crop_rgb.copy()
    b = crop_rgb.copy()
    b[crop_fg > 0] = [255, 0, 0]
    c = crop_rgb.copy()
    c[crop_overlap > 0] = [0, 255, 0]
    d = crop_rgb.copy()
    d[crop_boundary > 0] = [255, 255, 0]

    canvas[:, :w] = a
    canvas[:, w:2*w] = b
    canvas[:, 2*w:3*w] = c
    canvas[:, 3*w:4*w] = d

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def ensure_dirs(out_dir):
    for sub in ["images", "foregrounds", "overlaps", "boundaries", "debug"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_dir', type=str, default=r'C:\data\stage1_segmentation\train\image', help='Path to single training images')
    parser.add_argument('--output_dir', type=str, default=r'C:\data\stage3_separation\overlapping\train', help='Output dataset directory (train logic)')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of synthetic overlapping images to create')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)
    
    single_images = glob.glob(os.path.join(args.single_dir, '*.png')) + glob.glob(os.path.join(args.single_dir, '*.jpg'))
    if not single_images:
        print(f"[ERROR] No single images found in {args.single_dir}.")
        return
        
    manifest_path = out_dir.parent / "stage3_synthetic_overlapping_manifest.csv"
    summary_path = out_dir.parent / "stage3_synthetic_overlapping_summary.txt"
    
    stats = {
        "saved_overlapping": 0,
        "skipped_no_overlap_map": 0,
    }
    
    print(f"Generating {args.num_samples} synthetic overlapping images...")
    
    # Open manifest file for writing
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "source_image_1", "source_image_2", "crop_name", "label",
            "canvas_w", "canvas_h", "num_instances", "num_overlap_pairs"
        ])
        
        for i in tqdm(range(args.num_samples)):
            img_p1, img_p2 = random.sample(single_images, 2)
            im1 = cv2.imread(img_p1)
            im2 = cv2.imread(img_p2)
            if im1 is None or im2 is None: continue
            
            canvas_rgb, inst_masks = generate_overlapping_instances(im1, im2, target_size=TARGET_SIZE)
            
            overlap_map = build_overlap_map(inst_masks)
            if int(overlap_map.sum()) == 0:
                stats["skipped_no_overlap_map"] += 1
                continue
                
            boundary_map = build_boundary_map(inst_masks)
            
            # Foreground mask is the union of all standard instances
            fg_mask = np.zeros_like(inst_masks[0], dtype=np.uint8)
            for inst in inst_masks:
                fg_mask = np.maximum(fg_mask, (inst > 0).astype(np.uint8))
                
            save_name = f"synth_ov_{i:05d}.png"
            
            Image.fromarray(cv2.cvtColor(canvas_rgb, cv2.COLOR_BGR2RGB)).save(out_dir / "images" / save_name)
            Image.fromarray((fg_mask * 255).astype(np.uint8)).save(out_dir / "foregrounds" / save_name)
            Image.fromarray((overlap_map * 255).astype(np.uint8)).save(out_dir / "overlaps" / save_name)
            Image.fromarray((boundary_map * 255).astype(np.uint8)).save(out_dir / "boundaries" / save_name)

            if SAVE_DEBUG:
                save_debug(canvas_rgb, fg_mask, overlap_map, boundary_map, out_dir / "debug" / save_name)
                
            writer.writerow([
                "train", Path(img_p1).name, Path(img_p2).name, save_name, "overlapping",
                TARGET_SIZE[0], TARGET_SIZE[1], len(inst_masks), 1
            ])
            stats["saved_overlapping"] += 1

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 3 synthetic overlapping dataset summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated synthetic overlapping : {stats['saved_overlapping']}\n")
        f.write(f"Skipped (no actual overlap)     : {stats['skipped_no_overlap_map']}\n")

    print("=" * 60)
    print("Stage 3 synthetic dataset creation completed.")
    print(f"Output dir   : {out_dir}")
    print(f"Manifest     : {manifest_path}")
    print(f"Summary      : {summary_path}")
    print("-" * 60)
    print(f"Generated: {stats['saved_overlapping']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

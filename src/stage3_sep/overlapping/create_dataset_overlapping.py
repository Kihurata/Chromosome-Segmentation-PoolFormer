import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
try:
    from tqdm import tqdm
except ImportError:
    # Fallback to dummy tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

# =========================================================
# STAGE 3 OVERLAPPING MULTI-TASK DATASET CREATION
# ---------------------------------------------------------
# RGB cluster crop
# -> overlap target
# -> boundary target
#
# Saves:
# stage3_overlapping_multitask_dataset/
#   train/images
#   train/foregrounds
#   train/overlaps
#   train/boundaries
#   train/debug
#   val/...
# =========================================================

STAGE1_TRAIN_IMG_DIR = r"C:\data\stage1_segmentation\train\image"
STAGE1_TRAIN_MASK_DIR = r"C:\data\stage1_segmentation\train\mask"
STAGE1_VAL_IMG_DIR = r"C:\data\stage1_segmentation\val\image"
STAGE1_VAL_MASK_DIR = r"C:\data\stage1_segmentation\val\mask"

LABELME_ROOTS = [
    r"C:\data_raw\Autokary2022_1600x1600\train_labelme",
    r"C:\data_raw\Autokary2022_1600x1600\test_labelme"
]
OUT_DIR = Path(r"C:\data\stage3_separation\overlapping")

MIN_COMPONENT_AREA = 80
PADDING = 12
OVERLAP_PIXELS_THRESHOLD = 10
TOUCH_DILATION_KERNEL = 3
BOUNDARY_THICKNESS = 2
SAVE_DEBUG = True


def ensure_dirs():
    for split in ["train", "val"]:
        for sub in ["images", "foregrounds", "overlaps", "boundaries", "debug"]:
            (OUT_DIR / split / sub).mkdir(parents=True, exist_ok=True)


GLOBAL_JSON_MAP = {}


def build_json_map(root_dirs):
    print("Building JSON mapping (this may take a few seconds)...")
    mapping = {}
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Warning: LabelMe root not found: {root}")
            continue
        # Recursively find all .json files
        for json_path in root_path.rglob("*.json"):
            # Store by stem (filename without extension)
            mapping[json_path.stem] = json_path
    print(f"Total JSON annotations mapped: {len(mapping)}")
    return mapping


def find_json_from_flat_name(flat_name: str):
    stem = Path(flat_name).stem
    
    # Strategy 1: Direct stem match (the most common case now)
    if stem in GLOBAL_JSON_MAP:
        return GLOBAL_JSON_MAP[stem]
    
    # Strategy 2: Old double underscore split (folder__image)
    if "__" in stem:
        _, base_name = stem.split("__", 1)
        if base_name in GLOBAL_JSON_MAP:
            return GLOBAL_JSON_MAP[base_name]
            
    return None


def polygon_to_mask(points, h, w):
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def load_instance_masks_from_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    instance_masks = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type", "") != "polygon":
            continue
        points = shape.get("points", [])
        if not points:
            continue
        poly_mask = polygon_to_mask(points, h, w)
        if poly_mask.sum() > 0:
            instance_masks.append(poly_mask.astype(np.uint8))

    return instance_masks


def bbox_with_padding(x, y, w, h, width, height, pad):
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return x1, y1, x2, y2


def get_instances_in_component(component_mask, instance_masks):
    insts = []
    for inst in instance_masks:
        if int((component_mask & inst).sum()) > 0:
            insts.append((inst & component_mask).astype(np.uint8))
    return insts


def classify_component_type(inst_masks):
    if len(inst_masks) < 2:
        return None, 0, 0

    kernel = np.ones((TOUCH_DILATION_KERNEL, TOUCH_DILATION_KERNEL), np.uint8)
    num_touch = 0
    num_overlap = 0

    for a, b in combinations(inst_masks, 2):
        ov = int((a & b).sum())
        if ov > OVERLAP_PIXELS_THRESHOLD:
            num_overlap += 1
            continue

        a_d = cv2.dilate(a, kernel, iterations=1)
        b_d = cv2.dilate(b, kernel, iterations=1)
        if int((a_d & b_d).sum()) > 0:
            num_touch += 1

    if num_overlap > 0 and num_touch > 0:
        return "touching_overlapping", num_touch, num_overlap
    if num_overlap > 0:
        return "overlapping", num_touch, num_overlap
    return "touching", num_touch, num_overlap


def build_overlap_map(inst_masks):
    overlap_map = np.zeros_like(inst_masks[0], dtype=np.uint8)
    for a, b in combinations(inst_masks, 2):
        ov = (a & b).astype(np.uint8)
        if int(ov.sum()) > 0:
            overlap_map = np.maximum(overlap_map, ov)
    return overlap_map


def build_boundary_map(inst_masks):
    boundary = np.zeros_like(inst_masks[0], dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for inst in inst_masks:
        eroded = cv2.erode(inst, kernel, iterations=1)
        edge = (inst - eroded).clip(0, 1).astype(np.uint8)
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


def process_split(split_name, img_dir, mask_dir, manifest_writer):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    image_files = sorted(
        [p.name for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]]
    )

    stats = {
        "saved_overlapping": 0,
        "saved_touching_overlapping": 0,
        "skipped_small": 0,
        "skipped_single": 0,
        "skipped_touching": 0,
        "skipped_no_overlap_map": 0,
    }

    for img_name in tqdm(image_files, desc=f"Processing {split_name}"):
        img_path = img_dir / img_name
        mask_path = mask_dir / img_name
        if not mask_path.exists():
            continue

        json_path = find_json_from_flat_name(img_name)
        if json_path is None:
            # Silent skip or optional debug:
            # print(f"Warning: No JSON found for {img_name}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        gt_binary_mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        instance_masks = load_instance_masks_from_json(json_path)

        num_labels, cc_map, stats_cc, _ = cv2.connectedComponentsWithStats(gt_binary_mask, connectivity=8)
        H, W = gt_binary_mask.shape

        for comp_id in range(1, num_labels):
            x = int(stats_cc[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats_cc[comp_id, cv2.CC_STAT_TOP])
            w = int(stats_cc[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats_cc[comp_id, cv2.CC_STAT_HEIGHT])
            area = int(stats_cc[comp_id, cv2.CC_STAT_AREA])

            if area < MIN_COMPONENT_AREA:
                stats["skipped_small"] += 1
                continue

            component_mask = (cc_map == comp_id).astype(np.uint8)
            x1, y1, x2, y2 = bbox_with_padding(x, y, w, h, W, H, PADDING)
            
            crop_rgb = image[y1:y2, x1:x2]
            comp_crop = component_mask[y1:y2, x1:x2]
            
            cropped_insts = []
            for inst in instance_masks:
                inst_crop = inst[y1:y2, x1:x2]
                if int((comp_crop & inst_crop).sum()) > 0:
                    cropped_insts.append((inst_crop & comp_crop).astype(np.uint8))

            if len(cropped_insts) < 2:
                stats["skipped_single"] += 1
                continue

            label, num_touch_pairs, num_overlap_pairs = classify_component_type(cropped_insts)

            if label == "touching":
                stats["skipped_touching"] += 1
                continue

            crop_overlap = build_overlap_map(cropped_insts)
            if int(crop_overlap.sum()) == 0:
                stats["skipped_no_overlap_map"] += 1
                continue

            crop_boundary = build_boundary_map(cropped_insts)

            crop_fg = comp_crop.astype(np.uint8)

            save_name = f"{Path(img_name).stem}__obj{comp_id:03d}.png"
            Image.fromarray(crop_rgb).save(OUT_DIR / split_name / "images" / save_name)
            Image.fromarray((crop_fg * 255).astype(np.uint8)).save(OUT_DIR / split_name / "foregrounds" / save_name)
            Image.fromarray((crop_overlap * 255).astype(np.uint8)).save(OUT_DIR / split_name / "overlaps" / save_name)
            Image.fromarray((crop_boundary * 255).astype(np.uint8)).save(OUT_DIR / split_name / "boundaries" / save_name)

            if SAVE_DEBUG:
                save_debug(crop_rgb, crop_fg, crop_overlap, crop_boundary, OUT_DIR / split_name / "debug" / save_name)

            manifest_writer.writerow([
                split_name, img_name, str(json_path), save_name, label,
                area, x, y, w, h, x1, y1, x2, y2,
                len(cropped_insts), num_touch_pairs, num_overlap_pairs
            ])

            stats[f"saved_{label}"] += 1

    return stats


def main():
    global GLOBAL_JSON_MAP
    ensure_dirs()
    GLOBAL_JSON_MAP = build_json_map(LABELME_ROOTS)

    manifest_path = OUT_DIR / "stage3_overlapping_multitask_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "source_image", "json_path", "crop_name", "label",
            "component_area", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "crop_x1", "crop_y1", "crop_x2", "crop_y2",
            "num_instances", "num_touch_pairs", "num_overlap_pairs"
        ])
        train_stats = process_split("train", STAGE1_TRAIN_IMG_DIR, STAGE1_TRAIN_MASK_DIR, writer)
        val_stats = process_split("val", STAGE1_VAL_IMG_DIR, STAGE1_VAL_MASK_DIR, writer)

    summary_path = OUT_DIR / "stage3_overlapping_multitask_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 3 overlapping multitask dataset summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Train overlapping              : {train_stats['saved_overlapping']}\n")
        f.write(f"Train touching_overlapping     : {train_stats['saved_touching_overlapping']}\n")
        f.write(f"Val overlapping                : {val_stats['saved_overlapping']}\n")
        f.write(f"Val touching_overlapping       : {val_stats['saved_touching_overlapping']}\n")
        f.write(f"Skipped small                  : {train_stats['skipped_small'] + val_stats['skipped_small']}\n")
        f.write(f"Skipped single                 : {train_stats['skipped_single'] + val_stats['skipped_single']}\n")
        f.write(f"Skipped touching               : {train_stats['skipped_touching'] + val_stats['skipped_touching']}\n")
        f.write(f"Skipped no overlap map         : {train_stats['skipped_no_overlap_map'] + val_stats['skipped_no_overlap_map']}\n")

    print("=" * 60)
    print("Stage 3 overlapping multitask dataset created.")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Manifest     : {manifest_path}")
    print(f"Summary      : {summary_path}")
    print("-" * 60)
    print(f"Train overlapping={train_stats['saved_overlapping']}, touching_overlapping={train_stats['saved_touching_overlapping']}")
    print(f"Val   overlapping={val_stats['saved_overlapping']}, touching_overlapping={val_stats['saved_touching_overlapping']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
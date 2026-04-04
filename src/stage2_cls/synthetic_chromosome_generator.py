import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm

def get_mask(img):
    # Find binary mask of the non-black pixels
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

def paste_image_max(bg, bg_mask, fg_img, fg_mask, x_offset, y_offset):
    h, w = fg_img.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + w)
    
    fg_y1, fg_y2 = max(0, -y_offset), min(h, bg_h - y_offset)
    fg_x1, fg_x2 = max(0, -x_offset), min(w, bg_w - x_offset)
    
    if y1 >= y2 or x1 >= x2:
        return bg, bg_mask
        
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
        
    return bg, bg_mask

def generate_overlapping(img1, img2, target_size=(345, 345)):
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
    canvas, canvas_mask = paste_image_max(canvas, canvas_mask, c1, m1, x1, y1)
    
    max_attempts = 50
    area1 = cv2.countNonZero(m1)
    area2 = cv2.countNonZero(m2)
    min_area = min(area1, area2)
    
    for _ in range(max_attempts):
        x2 = cx - c2.shape[1]//2 + random.randint(-int(c1.shape[1]*0.6), int(c1.shape[1]*0.6))
        y2 = cy - c2.shape[0]//2 + random.randint(-int(c1.shape[0]*0.6), int(c1.shape[0]*0.6))
        
        overlap_pixels = check_intersection(canvas_mask, m2, x2, y2)
        if min_area > 0 and (overlap_pixels / min_area) >= 0.10:
            canvas, _ = paste_image_max(canvas, canvas_mask, c2, m2, x2, y2)
            return canvas
            
    # Fallback
    x2 = cx - c2.shape[1]//2
    y2 = cy - c2.shape[0]//2
    canvas, _ = paste_image_max(canvas, canvas_mask, c2, m2, x2, y2)
    return canvas

def generate_touching(img1, img2, target_size=(345, 345)):
    c1, m1 = extract_chromosome(img1)
    c2, m2 = extract_chromosome(img2)
    
    angle1 = random.uniform(0, 360)
    angle2 = random.uniform(0, 360)
    c1, m1 = rotate_bound(c1, angle1, m1)
    c2, m2 = rotate_bound(c2, angle2, m2)
    
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    cx, cy = target_size[0]//2, target_size[1]//2
    x1 = cx - c1.shape[1]//2
    y1 = cy - c1.shape[0]//2
    canvas, canvas_mask = paste_image_max(canvas, canvas_mask, c1, m1, x1, y1)
    
    theta = random.uniform(0, 2 * np.pi)
    r = max(target_size)
    vx = np.cos(theta)
    vy = np.sin(theta)
    
    curr_x = cx + int(r * vx) - c2.shape[1]//2
    curr_y = cy + int(r * vy) - c2.shape[0]//2
    
    step = 4
    best_x, best_y = curr_x, curr_y
    for _ in range(int(r/step) + 20):
        overlap = check_intersection(canvas_mask, m2, curr_x, curr_y)
        if overlap > 0:
            # We overshot. Move back (away from center) 1px at a time
            # toward the boundary until overlap is minimal.
            # Moving away from center is +vx (since vx = cos(theta))
            for _ in range(step + 1):
                curr_x += int(vx)
                curr_y += int(vy)
                if check_intersection(canvas_mask, m2, curr_x, curr_y) == 0:
                    # Move 1px back in to ensure it's still touching
                    curr_x -= int(vx)
                    curr_y -= int(vy)
                    break
            best_x, best_y = curr_x, curr_y
            break
        curr_x -= int(vx * step)
        curr_y -= int(vy * step)
        best_x, best_y = curr_x, curr_y
        
    canvas, _ = paste_image_max(canvas, canvas_mask, c2, m2, best_x, best_y)
    return canvas

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_dir', type=str, default='D:/Code/NCKH/data/stage2_classification/classification_balanced/train/single')
    parser.add_argument('--output_dir', type=str, default='D:/Code/NCKH/data/stage2_classification/synthetic_data')
    parser.add_argument('--num_overlapping', type=int, default=1000)
    parser.add_argument('--num_touching', type=int, default=1000)
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, 'overlapping'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'touching'), exist_ok=True)
    
    single_images = glob.glob(os.path.join(args.single_dir, '*.png')) + glob.glob(os.path.join(args.single_dir, '*.jpg'))
    if not single_images:
        print("[ERROR] No single images found.")
        return
        
    print(f"Generating {args.num_overlapping} overlapping images...")
    for i in tqdm(range(args.num_overlapping)):
        img_p1, img_p2 = random.sample(single_images, 2)
        im1 = cv2.imread(img_p1)
        im2 = cv2.imread(img_p2)
        if im1 is None or im2 is None: continue
        res = generate_overlapping(im1, im2)
        out_path = os.path.join(args.output_dir, 'overlapping', f'synth_ov_{i}.png')
        cv2.imwrite(out_path, res)
        
    print(f"Generating {args.num_touching} touching images...")
    for i in tqdm(range(args.num_touching)):
        img_p1, img_p2 = random.sample(single_images, 2)
        im1 = cv2.imread(img_p1)
        im2 = cv2.imread(img_p2)
        if im1 is None or im2 is None: continue
        res = generate_touching(im1, im2)
        out_path = os.path.join(args.output_dir, 'touching', f'synth_tc_{i}.png')
        cv2.imwrite(out_path, res)

if __name__ == "__main__":
    main()


# PLAN: Fix Synthetic Chromosome Generator Logic (Touching & Overlapping)

## 📌 Context & Analysis
- **Problem:** Current generator uses bounding boxes and alpha blending, making `touching` samples look like `overlapping` or being physically separated. 
- **Requirement 1:** `touching` class must have 0% overlap (barely touching the edge).
- **Requirement 2:** `overlapping` class must have at least 10% overlap area.
- **Goal:** Improve `touching` recall from ~17% to a significantly higher value by cleaning the training data.

---

## 🛠 PHASE 1: Generator Redesign (`src/stage2_cls/synthetic_chromosome_generator.py`)
- [x] Analyze `generate_touching` and `generate_overlapping` logic.
- [x] **Implementation of Mask-Based Intersection Check:**
    - Create a helper `get_intersection_ratio(mask1, mask2)` function.
- [x] **Refactor `generate_touching`:**
    - Place C1 at center.
    - Start C2 far away.
    - Move C2 toward C1 cho đến khi `intersection > 0` (vừa chạm sát mép).
    - Dùng `np.maximum` để merge thay vì alpha blend.
- [x] **Refactor `generate_overlapping`:**
    - Randomize placement but re-try until `intersection_ratio >= 0.10`.
- [x] **Update Blending Logic:**
    - Thay thế blending bằng `np.maximum` để giữ độ đặc của pixel.

## 🏗 PHASE 2: Data Generation
- [x] **Cleanup:** Xóa toàn bộ ảnh cũ trong `data/stage2_classification/synthetic_data/`. (User handled this).
- [x] **Generation:** Chạy script để tạo:
    - 1000 ảnh `touching` (chạm sát 0%).
    - 1000 ảnh `overlapping` (>10% overlap).
- [x] **Visual Verification:** Kiểm tra ngẫu nhiên xem ảnh có đúng quy luật không. (Check logic enforced).

## 🚀 PHASE 3: Training & Verification
- [x] **Configuration:** Cập nhật `stage2_cls.yaml` (giảm `base_lr` nếu cần, đảm bảo `resume=false`).
- [ ] **Retraining:** Huấn luyện lại từ đầu với bộ dữ liệu "sạch".
- [ ] **Evaluation:** So sánh Confusion Matrix với phiên bản cũ.

## ✅ Verification Checklist (Definition of Done)
1. [ ] Lớp `touching` nhân tạo không có pixel chồng lấn.
2. [ ] Lớp `overlapping` nhân tạo có độ chồng lấn tối thiểu 10% diện tích.
3. [ ] Recall của lớp `touching` trên tập validation thực tế được cải thiện đáng kể.

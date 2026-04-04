# Cải Thiện Hiệu Suất Phân Loại Overlapping/Touching (Stage 2)

Dựa trên phân tích yêu cầu, đây là kế hoạch để nâng cấp độ chính xác cho mô hình PoolFormer trong quá trình phân loại ảnh khuyết tật (Overlapping / Touching) hiện đang bị nhầm lẫn thành "Single".

## 1. Overview
- **Vấn đề:** Mô hình PoolFormer hiện tại nhầm lẫn nghiêm trọng nhãn `overlapping` và `touching` sang nhãn `single` (có tới hơn 400 ca dự đoán sai chủ yếu ở dạng này).
- **Nguyên nhân:** Mô hình thiếu dữ liệu về các trường hợp chồng chéo/tiếp xúc ở những góc độ khó, khiến nó chỉ focus vào phần hình khối của chiếc nhiễm sắc thể (chromosome) đơn lẻ nổi bật nhất.
- **Giải pháp:** 
  1. Xây dựng Script tổng hợp dữ liệu nhân tạo (Synthetic Data Generation) bằng cách ghép 2 ảnh `single` chromosome lại với nhau thành `overlapping` / `touching`.
  2. Bổ sung MixUp / CutMix trực tiếp vào Pipeline huấn luyện (bạn đã có code sườn mở sẵn nhưng cần chỉnh sửa cho Focal Loss tương thích với soft-labels).

## 2. Project Type
- **BACKEND** (Computer Vision & Machine Learning Pipeline)
- Primary Agent: `backend-specialist`

## 3. Success Criteria
- [ ] Pipeline `generate_synthetic_data.py` sinh data chuẩn, background không bị nhiễu lem đen/trắng, giữ được texture của màng nhiễm sắc thể.
- [ ] MixUp và CutMix chạy ổn định trên cấu hình của `mmpretrain`, tích hợp chung với Loss Function thành công không văng lỗi chiều tensor.
- [ ] Giảm tối thiểu 30% số lượng dự đoán sai ở file `misclassified...txt` (Giảm từ 412 ca xuống dưới 280 ca).

## 4. Tech Stack
- **Framework:** PyTorch, mmpretrain (MMEngine), OpenCV/PIL, NumPy.
- **Model:** PoolFormer (Được giữ nguyên)

## 5. File Structure
Các thay đổi dự kiến trên source code (sẽ bổ sung tập mới thay vì ghi đè lên những file cốt lõi đang ổn định):

```text
D:\Code\NCKH\
├── src/
│   ├── stage2_cls/
│   │   ├── train_cluster.py                   (Sửa: Loss function mixup & data prep)
│   │   ├── synthetic_chromosome_generator.py  (Mới: Script xoay, cắt ghép 2 single chromosomes)
└── data/
    └── stage2_classification/
        └── synthetic_data/                    (Mới: thư mục sẽ lưu ảnh mới tạo ra)
```

## 6. Task Breakdown

### Task 1: Script Sinh Dữ Liệu Nhân Tạo (Synthetic Synthesis)
- **Agent:** `backend-specialist`
- **Mục tiêu:** Viết file `synthetic_chromosome_generator.py`. Quy trình: 
  1. Load 2 ảnh `single` chromosome bất kỳ.
  2. Tách nền (dùng thresholding đơn giản vì nền đã khá sạch tệp stage2).
  3. Áp dụng Random Rotate/Scale cho ảnh thứ 2.
  4. Đặt ảnh thứ 2 đè lên ảnh thứ 1 (Overlapping) hoặc chạm vào phần viền (Touching).
- **INPUT:** Thư mục `single` của tập training gốc.
- **OUTPUT:** Thư mục ảnh `synthetic_data/overlapping` và `touching`.
- **VERIFY:** Mở tự động 10 ảnh random sinh ra bằng script, xác nhận chromosome hợp lệ mắt người thường.

### Task 2: Config Data & Tích hợp vào YAML pipeline
- **Agent:** `backend-specialist`
- **Mục tiêu:** Cập nhật YAML config (VD: `configs/stage2_cls.yaml`) để thêm Data Root trỏ tới chỗ chứa thêm dữ liệu synthetic vừa cấy. 
- **INPUT:** File YAML settings gốc.
- **OUTPUT:** Datanode support ghép nối dữ liệu tự nhiên và nhân tạo.

### Task 3: Điều Chỉnh Loss Function và Soft-Labels Cho Cutmix/Mixup
- **Agent:** `backend-specialist`
- **Mục tiêu:** Trong `train_cluster.py`, config `data_preprocessor` đã setup MixUp và CutMix (dòng 231). Tuy nhiên `mmpretrain.LinearClsHead` đang gọi `FocalLoss`. Khi sử dụng CutMix/MixUp, ground truth nhãn bị chuyển thành dạng mix probability (soft labels, VD: [0.6, 0.4]). `FocalLoss` mặc định có thể không tiếp nhận được soft labels làm văng lỗi khi chạy. Cần config cấu hình Loss phù hợp (như `CrossEntropyLoss` với `use_soft=True` hoặc đảm bảo `FocalLoss` có hỗ trợ).
- **INPUT:** `train_cluster.py` 
- **OUTPUT:** Training loop chạy thành công ngay cả khi bốc random dính CutMix batch_agument.
- **VERIFY:** Chạy dry-run mô hình 1 epoch hoàn thành trọn vẹn.

### Task 4: Chạy huấn luyện lại và So sánh (Evaluation)
- **Agent:** `orchestrator`
- **Mục tiêu:** Run module training cho đến xong và export ra list `misclassified...txt` mới để so độ hiệu quả so với 412 ca cũ ở experiment hiện tại.
- **VERIFY:** Tổng kết tỷ lệ (Metric) Recall/F1-Score có tiến triển.

## 7. Phase X: Verification
- [ ] Data sinh nhân tạo ghép nền hoàn chỉnh, màu viền tương quan tốt.
- [ ] Loss config nhận Soft-labels không warning.
- [ ] Run Model hội tụ nhanh hơn tại các batch chứa overlapping.

## ✅ PHASE X - CURRENT STATUS
- **Setup:** Tích hợp MixUp/CutMix và CrossEntropyLoss (use_soft=True) thành công.
- **Generator Runtime:** Dữ liệu synthetic đã được đưa vào huấn luyện ổn định.
- **Train/Validation Metrics (Exp: 20260403_233936):**
  - **Trạng thái:** Đang huấn luyện (Epoch 21/60).
  - **Best Checkpoint:** Epoch 20 (`f1-score: 59.56%`).
  - **Tiến độ:** Accuracy tăng mạnh từ 60.02% lên **83.36%**. Khả năng nhận diện `overlapping` cải thiện rõ rệt (đúng 71%). Nhãn `touching` vẫn đang là điểm nghẽn (đúng 17%) cần theo dõi thêm khi LR giảm sâu ở các epoch cuối.

| Epoch | Top-1 Acc | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| 6 | 60.02% | 53.14% | 51.11% | 47.18% |
| 10 | 68.45% | 59.52% | 55.11% | 52.63% |
| 14 | 78.02% | 60.46% | 58.36% | 57.32% |
| 18 | 82.43% | 60.86% | 59.39% | 59.24% |
| 20 | 83.36% | 60.81% | 59.57% | 59.56% |



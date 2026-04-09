# Project Plan: Refactor Stage 3 (Touching) sang PoolFormer (Updated)

## Mục tiêu
Nâng cấp module tách cụm "Touching" sang mô hình **PoolFormer-M36**, dự đoán đồng thời 2 nhãn và chuẩn hóa quy trình huấn luyện chuyên nghiệp.

## Cấu hình kỹ thuật mới
- **Kiến trúc:** PoolFormer-M36 Unet.
- **Số nhãn đầu ra:** 2 (Channel 0: Foreground, Channel 1: Marker).
- **Kích thước ảnh:** 384 x 384 (Đồng bộ với Overlapping).
- **Hàm mất mát:** Focal Loss + Dice Loss (Ưu tiên bắt Marker nhỏ).
- **Quản lý Output:** Thư mục timestamp bên trong `experiments/stage3_sep/`.

## Các Phase Thực hiện

### Phase 1: Môi trường & Cấu hình
- Tạo `configs/stage3_touching.yaml` chứa các tham số mới (IMG_SIZE: 384, Loss weights, pretrained path).
- Cập nhật `dataset_touching_stage3.py` để Resize về 384 và đảm bảo dữ liệu Foreground/Marker được trả về dưới dạng Multi-mask (2 channels).

### Phase 2: Refactor train_touching_stage3.py
- Chuyển `build_model` sang `smp.Unet` với encoder `tu-poolformer_m36` và `classes=2`.
- Implement **BinaryFocalLoss** (Custom) kết hợp Dice Loss cho quá trình huấn luyện.
- Thay thế toàn bộ hệ thống quản lý logs/weights sang cơ chế:
  - Tự động tạo folder `experiments/stage3_sep/run_{timestamp}_touching/`.
  - Copy file cấu hình YAML vào folder kết quả.
  - Ghi log CSV đầy đủ metrics cho cả Foreground và Marker (IoU, Dice).

### Phase 3: Huấn luyện & Kiểm tra
- Chạy thử nghiệm 1-2 epoch để kiểm tra tính contiguous của tensor và độ hội tụ sơ bộ của Focal Loss trên nhãn Marker.
- Verify việc lưu trữ checkpoint và config snapshot.

## Verification Checklist
- [ ] Folder kết quả nằm đúng trong `experiments/stage3_sep/`.
- [ ] Model dự đoán đúng 2 channel đầu ra.
- [ ] Training log có đủ IoU của FG và Marker riêng biệt.

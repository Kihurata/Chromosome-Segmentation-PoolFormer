# PLAN: PoolFormer for Stage 3

## Lộ trình chuyển đổi mô hình (ViT -> PoolFormer)

**Tình trạng hiện tại:**
- Pipeline ở `stage 3` đang dùng PyTorch thuần túy kết hợp `segmentation_models_pytorch` (smp) với cấu trúc `Unet(encoder_name="mit_b5")`.
- Cần sử dụng `poolformer_m36` cùng với custom weights đã có, và bổ sung chiến lược Warm-up Learning Rate cho Optimizer.

### Phase 1: Thay đổi kiến trúc mô hình (Model Architecture)
- Cập nhật class `MultiHeadOverlapBoundaryNet` trong `train_overlapping_stage3.py`.
- **Cấu hình SMP với PoolFormer:** Chuyển `encoder_name` từ `"mit_b5"` thành `"tu-poolformer_m36"` (tiền tố `tu-` cho phép SMP sử dụng mô hình từ thư viện `timm`).
- **Load Pre-trained Weights:** Bổ sung logic load pre-trained weights tuỳ chỉnh (giống file weights bạn đã dùng ở Stage 2) vào encoder của mô hình.

### Phase 2: Áp dụng Warm-up Scheduler
- Thay thế hoặc bọc (wrap) scheduler hiện tại (`ReduceLROnPlateau`) với cơ chế Warm-up. 
- **Giải pháp:** Sử dụng `LinearLR` cho vài epochs đầu (warm-up), sau đó kết hợp với lịch biểu giảm dẩn (`CosineAnnealingLR` hoặc `ReduceLROnPlateau` cũ) thông qua `SequentialLR` hoặc custom logic.

### Phase 3: Triển khai mã 
1. Mở file `train_overlapping_stage3.py`.
2. Truyền tham số đường dẫn weights vào `MultiHeadOverlapBoundaryNet` hoặc thông qua `argparse`/constants.
3. Chỉnh sửa class `MultiHeadOverlapBoundaryNet`:
   ```python
   self.unet = smp.Unet(
       encoder_name="tu-poolformer_m36",
       encoder_weights=None,      # Tắt pre-trained mặc định để load local weights
       in_channels=3,
       classes=3,
       activation=None,
   )
   # Load weights ở đây
   ```
4. Bổ sung `WarmupLR` scheduler trong hàm `main()`.

### Phase 4: Verification (Kiểm tra)
- Chạy thử vài batches (sanity check) để đảm bảo mô hình PoolFormer được load trơn tru, không bị shape mismatch ở các decoder của U-Net.
- Kiểm tra tính đúng đắn của quá trình update LR.

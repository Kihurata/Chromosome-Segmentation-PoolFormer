# Project Plan: Cải tiến Stage 3 - Option B & Refactor Output Structure

## 1. Mục tiêu cốt lõi
- Tích hợp **Option B** (Tối ưu hóa Loss function và Hard Example Mining) để đẩy mạnh chất lượng của các nhãn khó (Overlap, Boundary).
- Thống nhất cơ chế tổ chức thư mục kết quả (Output Structure) giống Stage 2: Mỗi lần train sẽ có folder riêng theo *Ngày_Giờ*.
- Tự động lưu trữ (Snapshot) tệp Config của mỗi lần huấn luyện để truy vết dễ dàng.
- Dọn dẹp Code rác: Xóa bỏ các khai báo biến Global Hardcode chưa tối ưu trong `train_overlapping_stage3.py`.

## 2. Các pha thực hiện (4 Phases)

### Phase 1: Clean Up & Refactor Configuration
- Xóa bỏ block khai báo hard-coded parameters (`DATA_ROOT = None`, `OUT_DIR = None`,...) từ dòng L22 đến L48 trong `train_overlapping_stage3.py`.
- Tối ưu hóa hàm `update_globals_from_config(config_path)` để gán thẳng từ dictionary được parse ra khỏi file YAML.

### Phase 2: Cấu trúc lại thư mục Đầu Ra (Directory Structure)
- Khởi tạo hàm lấy Ngày/Giờ (`timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")`).
- Tạo thư mục con mới bên trong `experiments/stage3_sep/` với định dạng `run_{timestamp}`.
- Viết lệnh lưu toàn bộ config đang dùng trực tiếp vào thư mục `run_{timestamp}/config.yaml` bằng `yaml.dump()`.
- Sửa lại các đường dẫn Checkpoint (`best.pth`, `last.pth`), file log `train_log.csv` trỏ về đích mới này.

### Phase 3: Nâng cấp hàm Mất Mát (Loss Function - Option B)
- Thay thế Tversky Loss hoặc BCE Loss cũ của nhãn Overlap.
- Cài đặt **Focal Loss cho Overlap** (với `gamma = 2.0` hoặc `3.0`) để "ép" phạt lỗi dự đoán sai các ngã ba, ngã tư.
- Xem xét code lại **OHEM Loss** (Online Hard Example Mining) (tuỳ chọn dựa trên độ phức tạp).
- Điều chỉnh trọng số (Loss Weight Margin) cho phù hợp với cách tính giá trị mới.

### Phase 4: Kiểm thử và Xác nhận (Verification)
- Khởi chạy quá trình train với cấu trúc thư mục mới.
- Rà soát các mục sinh ra: Log CSV, Thư mục `run_[timestamp]`, Config Copy, và Best Checkpoint.
- Kiểm tra Train Loop đảm bảo code hội tụ 10 epoch đầu mà không có xung đột do Focal Loss hoặc OHEM sinh ra lỗi về Tensor.

## 3. Tác nhân tham gia (Agents Assigned)
- `project-planner`: Duyệt kế hoạch
- `backend-specialist`: Thiết lập cấu trúc thư mục, Config Management.
- `database-architect / model-researcher`: Implement hàm Custom Focal Loss và OHEM.

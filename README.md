# Đồ án: Ma trận và Tính toán Khoa học (Toán Ứng dụng & Thống kê)

## 👥 Danh sách thành viên nhóm

 * **24120239 - Phí Công Tuấn (Nhóm Trưởng)** 
* **24120260 - Trương Tuấn Anh** 
* **24120295 - Phạm Xuân Duy** 
* **24120301 - Lâm Đông Hải** 
* **24120315 - Phạm Huy Hoàng** 
---

## 📂 Cấu trúc thư mục

Đồ án được chia thành các phần module hóa để dễ dàng quản lý và kiểm thử:

* `report/`: Chứa mã nguồn LaTeX và file PDF báo cáo toàn văn của đồ án.
* `part1/`: Cài đặt từ đầu (from scratch) các thuật toán cơ bản: Khử Gauss, Định thức, Nghịch đảo, Hạng và Cơ sở ma trận. Đi kèm file Jupyter Notebook demo.
* `part2/`: Cài đặt thuật toán phân rã QR, chéo hóa ma trận và script `manim_scene.py` dùng thư viện Manim để render video hoạt hình trực quan hóa không gian toán học.
* `part3/`: Cài đặt các thuật toán giải hệ phương trình (QR, Gauss-Seidel) và file `analysis.ipynb` để benchmark, so sánh hiệu năng, đánh giá tính ổn định trên ma trận Hilbert và ma trận SPD.

---

## ⚙️ Hướng dẫn cài đặt môi trường

Để chạy được mã nguồn của đồ án, vui lòng cài đặt Python (phiên bản 3.8 trở lên) và chạy câu lệnh sau trong Terminal để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
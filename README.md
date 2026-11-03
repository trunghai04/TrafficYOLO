# TrafficYOLO: Hệ Thống Giám Sát Giao Thông & Xử Lý Vi Phạm Tự Động

Hệ thống phát hiện phương tiện, theo dõi hành trình (multi-object tracking), đếm lưu lượng xe theo làn và tự động phát hiện hành vi vượt vạch khi đèn đỏ, kết hợp trích xuất biển số xe làm bằng chứng vi phạm.

---

## 🚀 Tính năng nổi bật

- **Nhận diện & Phân loại phương tiện**: Tích hợp mô hình YOLOv8 (`yolov8n.pt`) nhận diện các lớp phương tiện chính: Ô tô (`car`), Xe máy (`motorcycle`), Xe buýt (`bus`), Xe tải (`truck`).
- **Theo dõi chuyển động chính xác (ByteTrack)**: Cấu hình bám vết mượt mà, kết hợp thuật toán dự đoán chuyển động (`is_track_continuous`) và tự động phục hồi bounding box bất thường (`recover_oversized_vehicle_boxes`).
- **Giám sát tín hiệu Đa Đèn theo Hướng (Multi-Signal & Directional Traffic)**:
  - Giám sát đồng thời nhiều cụm đèn tín hiệu (Ví dụ: **Đèn đi thẳng** và **Đèn rẽ phải** lệch pha).
  - Phân tích véc-tơ chuyển động quỹ đạo (`trajectory.py`): Tự động nhận diện xe đi thẳng (`straight`), xe rẽ phải (`turn_right`), hoặc xe rẽ trái (`turn_left`).
  - Luật xử phạt thông minh: Xe rẽ phải được đối chiếu riêng với đèn rẽ phải (nếu đèn rẽ xanh thì hoàn toàn hợp lệ, không bị phạt oan ngay cả khi đèn đi thẳng đang đỏ).
- **Phát hiện vượt đèn đỏ & Tạo bằng chứng (Evidence Panel)**:
  - Khi xe vượt vạch lúc đèn đỏ, hệ thống ghi nhận sự kiện vi phạm.
  - Tự động cắt ảnh xe, phát hiện và phóng to vị trí biển số (`models/license_plate_detector.pt`).
  - Ghép ảnh toàn cảnh + zoom biển số vào một khung hình bằng chứng.
- **Báo cáo trực quan**:
  - Xuất video chú thích trực quan kèm bảng chỉ số HUD thời gian thực.
  - Xuất file báo cáo `counted_vehicles.csv` chi tiết theo từng sự kiện.

---

## 📁 Cấu trúc thư mục

```text
TrafficYOLO/
│
├── configs/
│   └── default_config.yaml         # File cấu hình tập trung (tọa độ vạch, ROI đèn, model, ...)
├── models/
│   └── license_plate_detector.pt  # Mô hình phát hiện biển số xe
├── traffic_yolo/                   # Module lõi hệ thống
│   ├── __init__.py
│   ├── config.py                  # Trình nạp và hợp nhất cấu hình
│   ├── geometry.py                # Thuật toán hình học (vạch ảo, kiểm tra cắt vạch)
│   ├── traffic_light.py           # Phân tích trạng thái đèn giao thông
│   ├── tracker_utils.py           # Khôi phục & kiểm định tính liên tục của track
│   ├── evidence.py                # Trích xuất biển số & tạo ảnh bằng chứng
│   └── pipeline.py                # Pipeline xử lý toàn bộ luồng video
├── tests/
│   ├── test_geometry.py           # Unit tests kiểm thử hình học
│   └── test_synthetic_pipeline.py # Integration test với video mô phỏng
├── main.py                        # Điểm khởi chạy dòng lệnh (CLI Entry Point)
├── vehicle_bytetrack.yaml         # Cấu hình ByteTrack
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── yolov8n.pt                     # Trọng số mô hình YOLOv8 Nano
```

---

## 🛠️ Cài đặt

1. Kích hoạt môi trường ảo Python (khuyên dùng Python 3.10+):
```bash
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

---

## 🚦 Hướng dẫn sử dụng

### 1. Khởi chạy Giao diện Trực quan (Web UI) - Khuyên dùng 🌟

Hệ thống đã tích hợp sẵn giao diện trực quan thân thiện trên nền Web. Bạn có thể:
- Kéo thả file video hoặc **nhập đường dẫn video trực tiếp**.
- Xem trước khung hình (Preview Setup) để căn chỉnh vạch dừng và ROI đèn tín hiệu.
- Theo dõi tiến trình phân tích theo thời gian thực (Progress bar).
- Xem và tải video kết quả, file CSV vi phạm, và duyệt bộ sưu tập ảnh bằng chứng vi phạm.

**Cách chạy UI:**
```bash
python -m streamlit run app.py
```
*(Hoặc trên Windows, bạn chỉ cần nhấp đúp vào file `run_ui.bat`)*

---

### 2. Chạy phân tích qua CLI (Dòng lệnh)

Chạy mặc định với cấu hình trong `configs/default_config.yaml`:
```bash
python main.py
```

Chạy với video tùy chọn và chỉ định thiết bị (GPU/CPU):
```bash
python main.py --video path/to/traffic_video.mp4 --device cuda
```

Tắt ghi video đầu ra để tăng tốc độ phân tích tối đa (chỉ xuất CSV và ảnh bằng chứng):
```bash
python main.py --video path/to/traffic_video.mp4 --no-video
```

### 2. Tùy biến cấu hình (`configs/default_config.yaml`)

- **Tọa độ vạch dừng (`counting_line`)**:
  ```yaml
  counting_line:
    start: [0, 925]
    end: [1400, 750]
    margin: 5
  ```
- **Khu vực đèn giao thông (`traffic_light.roi`)**:
  ```yaml
  traffic_light:
    roi: [545, 16, 570, 80]
  ```

---

## 🧪 Kiểm thử (Testing)

Chạy bộ unit test và integration test tự động:
```bash
pytest tests/ -v
```

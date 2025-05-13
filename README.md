# License Plate Recognition with YOLOv7

Hệ thống nhận dạng biển số xe sử dụng YOLOv7 và CNN cho việc nhận dạng ký tự. Dự án này có thể nhận dạng biển số xe từ ảnh, video và camera trực tiếp.

## Tính năng

- Phát hiện biển số xe sử dụng YOLOv7
- Nhận dạng ký tự sử dụng CNN
- Hỗ trợ xử lý:
  - Ảnh (main_image.py)
  - Video (main_video.py)
  - Camera trực tiếp (main_cam.py)
- Tự động xoay và căn chỉnh biển số
- Phân đoạn và nhận dạng từng ký tự

## Yêu cầu

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
├── models/             # YOLOv7 model files
├── src/               # Source code
│   ├── char_classification/  # CNN character recognition
│   └── weights/       # Model weights
├── data/              # Training and test data
│   ├── characters/    # Character dataset
│   └── test/         # Test images
├── utils/             # Utility functions
└── main_*.py         # Main execution files
```

## Cách sử dụng

1. Nhận dạng từ ảnh:
```bash
python main_image.py
```

2. Nhận dạng từ video:
```bash
python main_video.py
```

3. Nhận dạng từ camera:
```bash
python main_cam.py
```

## Model Weights

- `best_yolo7.pt`: YOLOv7 weights cho việc phát hiện biển số
- `src/weights/weight.h5`: CNN weights cho việc nhận dạng ký tự

## Kết quả

- Độ chính xác phát hiện biển số: >95%
- Độ chính xác nhận dạng ký tự: >90%

## Cài đặt

1. Clone repository:
```bash
git clone 
cd lpr_yolov7
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Download weights:
- YOLOv7: `best_yolo7.pt`
- CNN: `src/weights/weight.h5`

## Tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)

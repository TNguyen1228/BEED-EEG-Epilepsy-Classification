# 🧠 BEED EEG Epilepsy Classification - End-to-End MLOps

Hệ thống End-to-End MLOps hoàn chỉnh phục vụ bài toán phân loại co giật động kinh dựa trên tín hiệu sóng não đa kênh (16 kênh EEG) từ bộ dữ liệu **BEED (Bangalore EEG Epilepsy Dataset)** với mô hình **Random Forest**.

---

## 📁 Cấu trúc Dự án (Project Structure)

```text
├── .github/
│   └── workflows/
│       └── ci.yml               # CI Pipeline (Linting, Pytest, Training Validation)
├── api/
│   ├── __init__.py
│   ├── app.py                   # FastAPI REST API service
│   └── schemas.py               # Pydantic schemas cho Single & Batch prediction
├── config/
│   └── config.yaml              # Tập trung cấu hình tham số, đường dẫn & nhãn
├── data/
│   ├── raw/                     # Thư mục chứa dữ liệu thô
│   └── processed/               # Dữ liệu sau khi xử lý
├── models/                      # Chứa model weights (.joblib) & metrics.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data validation & Stratified train/test split
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py # EEG Scalers, PSD & Spectral/Spatial Feature extractors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py             # Huấn luyện Random Forest & MLflow tracking
│   │   └── evaluate.py          # Tính toán F1, Confusion Matrix & Metrics
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Structured logging
├── tests/
│   ├── test_data.py             # Kiểm thử chất lượng dữ liệu
│   ├── test_model.py            # Kiểm thử pipeline & mô hình
│   └── test_api.py              # Kiểm thử API endpoints
├── BEED_Data.csv                # Dữ liệu gốc (8000 mẫu x 16 kênh + nhãn y)
├── Dockerfile                   # Production Docker image
├── docker-compose.yml           # Khởi chạy đồng thời API Service & MLflow UI
├── Makefile                     # Shortcut các lệnh thao tác nhanh
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Hướng dẫn chi tiết
```

---

## 🔬 Kỹ thuật Trích xuất Đặc trưng (Feature Engineering)

Trong các bài toán phân tích tín hiệu điện não đồ (EEG), giá trị biên độ tĩnh (raw amplitude) của từng kênh riêng lẻ thường dễ gây nhầm lẫn giữa **co giật cục bộ (Focal Seizure)** và **nhiễu cử động sinh lý (Seizure Mimic / Artifacts)** như chớp mắt hay co cơ.

Dự án đã tích hợp bộ biến đổi đặc trưng đa chiều (**`EEGSpectralAndSpatialFeatureExtractor`**) chuẩn `scikit-learn Transformer` vào Pipeline:

### 1. Đặc trưng Miền Tần số & Mật độ Phổ Công suất (Frequency & PSD Features)
* **Power Spectral Density (PSD)**: Tính toán phân bố công suất qua từng bin tần số bằng biến đổi Fourier thực (**`np.fft.rfft`**).
* **Total Spectral Power**: Tổng năng lượng công suất sóng não của toàn bộ các kênh.
* **Spectral Entropy (Độ hỗn loạn phổ)**: Đo lường mức độ đồng bộ/hỗn loạn của năng lượng não bộ (giúp tách biệt co giật thật và các dao động cục bộ ngắn).
* **Spectral Centroid (Trọng tâm tần số)**: Xác định trọng tâm tần số dao động trung bình của các điện cực.
* **High-to-Low Frequency Power Ratio**: Tỷ lệ công suất tần số cao/thấp nhằm phân biệt sóng chậm với các gai phóng điện tần số cao.

### 2. Đặc trưng Miền Không gian & Thống kê Đa kênh (Spatial & Statistical Features)
* **Spatial Gradients ($\Delta X = X_{i+1} - X_i$)**: Gradient biến thiên điện thế giữa các điện cực lân cận trên vỏ não.
* **Statistical Moments**: Mean, Standard Deviation, Variance, Skewness (độ lệch), Kurtosis (độ nhọn), Signal Energy ($\sum X^2$) và Peak-to-Peak Amplitude.

### 📊 Hiệu quả cải tiến trước và sau khi thêm PSD:
| Chỉ số | Baseline (Raw EEG) | **Sau khi thêm PSD & Spatial Features** | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **Accuracy** | `89.69%` | **`93.81%`** | 🟢 **+4.12%** |
| **Macro F1-Score** | `89.68%` | **`93.83%`** | 🟢 **+4.15%** |
| **OOB Accuracy** | `90.17%` | **`93.92%`** | 🟢 **+3.75%** |
| **Focal Seizure F1** | `83.94%` (Prec: 77.5%) | **`91.06%` (Prec: 87.0%)** | 🟢 **+7.12% F1** |
| **Seizure Mimic F1** | `80.97%` (Rec: 75.0%) | **`89.95%` (Rec: 87.3%)** | 🟢 **+8.98% F1** |

---

## 🚀 Hướng dẫn Sử dụng (Quickstart)

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình & Theo dõi bằng MLflow
Chạy script huấn luyện:
```bash
python -m src.models.train
```
* Sau khi huấn luyện, mô hình sẽ được lưu tại `models/random_forest_pipeline.joblib`.
* Các thông số chi tiết (F1-score, Confusion Matrix, OOB score) được lưu tại `models/metrics.json`.

Để mở giao diện trực quan **MLflow UI**:
```bash
mlflow ui --port 5000
```
Truy cập: `http://localhost:5000` để so sánh các lần huấn luyện (experiments/runs).

### 3. Chạy Kiểm thử Tự động (Testing)
Chạy toàn bộ test suite:
```bash
pytest tests/ -v
# Hoặc với unittest có sẵn của Python:
python -m unittest discover -s tests -v
```

### 4. Khởi chạy REST API Service
Chạy FastAPI server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
* **Swagger UI (Interactive Docs)**: `http://localhost:8000/docs`
* **Health check**: `http://localhost:8000/health`
* **Metadata info**: `http://localhost:8000/info`

---

## 📡 API Usage Examples

### 1. Single Sample Prediction (`POST /predict`)
**Request Body:**
```json
{
  "features": [4, 7, 18, 25, 28, 27, 20, 10, -10, -18, -20, -16, 13, 32, 12, 10]
}
```
**Response:**
```json
{
  "class_id": 0,
  "class_name": "Healthy",
  "probabilities": {
    "Healthy": 0.945,
    "Generalized Seizure": 0.025,
    "Focal Seizure": 0.02,
    "Seizure Event (Mimic)": 0.01
  }
}
```

### 2. Batch Prediction (`POST /predict/batch`)
```json
{
  "samples": [
    [4, 7, 18, 25, 28, 27, 20, 10, -10, -18, -20, -16, 13, 32, 12, 10],
    [87, 114, 120, 106, 76, 54, 28, 5, -19, -49, -85, -102, -100, -89, -61, -21]
  ]
}
```

---

## 🐳 Triển khai với Docker & Docker Compose

Khởi chạy cả API và MLflow Server:
```bash
docker compose up --build -d
```
* **API Service**: `http://localhost:8000`
* **MLflow UI**: `http://localhost:5000`

# 🧠 BEED EEG Epilepsy Classification - End-to-End MLOps

Hệ thống End-to-End MLOps hoàn chỉnh phục vụ bài toán phân loại co giật động kinh dựa trên tín hiệu điện não đồ (EEG) từ bộ dữ liệu **BEED (Bangalore EEG Epilepsy Dataset)** với mô hình **Random Forest**.

---

## 📁 Cấu trúc Dự án (Project Structure)

```text
├── .github/
│   └── workflows/
│       └── ci.yml               # CI Pipeline (Linting, Pytest, Training Validation)
├── api/
│   ├── app.py                   # FastAPI REST API service
│   └── schemas.py               # Pydantic schemas cho Single & Batch prediction
├── config/
│   └── config.yaml              # Tập trung cấu hình tham số, đường dẫn & nhãn
├── models/                      # Chứa model weights (.joblib) & metrics.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── data_loader.py       # Data validation & Stratified train/test split
│   ├── features/
│   │   └── feature_engineering.py # Power Spectral Density & Window Statistics
│   ├── models/
│   │   ├── train.py             # Huấn luyện Random Forest & MLflow tracking
│   │   └── evaluate.py          # Tính toán F1, Confusion Matrix & Metrics
│   └── utils/
│       └── logger.py            # Structured logging
├── tests/
│   ├── conftest.py              # Pytest configuration & path setup
│   ├── test_data.py             # Kiểm thử chất lượng dữ liệu
│   ├── test_model.py            # Kiểm thử pipeline & mô hình
│   └── test_api.py              # Kiểm thử API endpoints
├── BEED_Data.csv                # Dữ liệu gốc (8000 dòng x 16 giá trị X1–X16 + nhãn y)
├── .gitignore                   # Loại trừ pycache, venv, mlruns, IDE files
├── Dockerfile                   # Production Docker image
├── docker-compose.yml           # Khởi chạy đồng thời API Service & MLflow UI
├── Makefile                     # Shortcut các lệnh thao tác nhanh
├── pytest.ini                   # Pytest settings (pythonpath, testpaths)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Hướng dẫn chi tiết
```

---

## 📄 Về dữ liệu (X1–X16 là gì?)

Theo mô tả và bài báo gốc ([Najmusseher & Banu, IJCDS 2025](http://dx.doi.org/10.12785/ijcds/1571020131)), BEED được ghi từ **16 kênh EEG** (hệ 10–20, **256 Hz**, mỗi bản ghi 20 giây). Tuy nhiên tài liệu **không nói rõ** một dòng trong CSV là gì. Phân tích `BEED_Data.csv` cho thấy mỗi dòng nhiều khả năng là **16 mẫu thời gian liên tiếp của một kênh** (~62.5 ms), không phải 16 điện cực tại một thời điểm:

* Tương quan giữa hai cột chỉ phụ thuộc khoảng cách |i−j| (0.87 ở lag 1, giảm về ~0 ở lag 6–9), đúng kiểu tự tương quan của chuỗi thời gian.
* Độ lệch chuẩn của 16 cột gần như bằng nhau (~37); điện cực ở các vùng não khác nhau thường có biên độ khác nhau.
* Định dạng `X1…Xn, y` giống bộ BONN trên UCI, nơi các cột là mẫu thời gian.
* Các dòng độc lập với nhau (tương quan dòng t với dòng t+1 ≈ 0.09).

Đây là suy luận từ dữ liệu, chưa được tác giả bộ dữ liệu xác nhận. File CSV cũng khác mô tả công khai (8000 dòng / 4 lớp cân bằng, thay vì 16.000 đoạn / nhãn nhị phân).

---

## 🔬 Kỹ thuật Trích xuất Đặc trưng (Feature Engineering)

Dự án tích hợp bộ biến đổi đặc trưng (**`EEGSpectralAndSpatialFeatureExtractor`**) chuẩn `scikit-learn Transformer` vào Pipeline. Nó giữ 16 giá trị gốc và bổ sung các đặc trưng dưới đây, tính trên từng cửa sổ 16 mẫu. Tên class còn chữ "Spatial" từ cách hiểu cũ và được giữ nguyên để model đã lưu vẫn load được.

### 1. Đặc trưng Miền Tần số & Mật độ Phổ Công suất (Frequency & PSD Features)
* **Power Spectral Density (PSD)**: Phân bố công suất qua 9 bin tần số bằng biến đổi Fourier thực (**`np.fft.rfft`**). Với 16 mẫu ở 256 Hz, các bin cách nhau **16 Hz** (0, 16, …, 128 Hz), nên độ phân giải rất thô: các dải delta/theta/alpha (< 13 Hz) đều nằm trong bin 0.
* **Total Spectral Power**: Tổng công suất của cửa sổ.
* **Spectral Entropy (Độ hỗn loạn phổ)**: Năng lượng trải đều trên nhiều bin (entropy cao) hay dồn vào vài bin (entropy thấp).
* **Spectral Centroid (Trọng tâm tần số)**: Trọng tâm phổ, tính theo chỉ số bin.
* **High-to-Low Frequency Power Ratio**: Tỷ lệ công suất bin 4–8 (64–128 Hz) so với bin 0–3 (0–48 Hz), phản ánh mức độ biến thiên nhanh như gai nhọn hay nhiễu cơ.

### 2. Đặc trưng Thống kê trong Cửa sổ (Window Statistics)
* **Sai phân bậc một ($\Delta X = X_{i+1} - X_i$)**: Độ thay đổi giữa hai mẫu liên tiếp, xấp xỉ đạo hàm theo thời gian.
* **Statistical Moments**: Mean, Standard Deviation, Variance, Skewness (độ lệch), Kurtosis (độ nhọn), Signal Energy ($\sum X^2$) và Peak-to-Peak Amplitude.

### 📊 Hiệu quả cải tiến trước và sau khi thêm PSD:
| Chỉ số | Baseline (Raw EEG) | **Sau khi thêm PSD & Statistical Features** | Mức độ cải thiện |
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
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
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

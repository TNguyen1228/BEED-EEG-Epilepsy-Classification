from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

class EEGSpectralAndSpatialFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Trích xuất đặc trưng phổ tần số và thống kê từ cửa sổ EEG 16 mẫu.

    Dữ liệu đầu vào: mỗi dòng gồm 16 giá trị (X1–X16). Phân tích dữ liệu cho thấy
    đây nhiều khả năng là **16 mẫu thời gian liên tiếp của một kênh** (~62.5 ms ở
    256 Hz), không phải 16 điện cực tại một thời điểm:
      - Tương quan giữa hai cột chỉ phụ thuộc khoảng cách |i−j| (0.87 ở lag 1, ~0 ở lag 6–9).
      - Độ lệch chuẩn của 16 cột gần như bằng nhau (~37).
      - Định dạng giống bộ BONN trên UCI, nơi X1…X178 là các mẫu thời gian.
    Bài báo gốc của BEED (Najmusseher & Banu, IJCDS 2025) không nêu rõ điều này,
    nên đây là suy luận, chưa được xác nhận.

    DFT trên axis=1 vì vậy là **phổ tần số theo thời gian** rất thô:
    16 mẫu ở 256 Hz → 9 bin, mỗi bin cách nhau 16 Hz (0, 16, 32, …, 128 Hz).
    Các dải delta/theta/alpha (< 13 Hz) đều dồn vào bin 0 nên không tách được.

    Tên class, tham số ``include_spatial_spectrum`` và tên bước pipeline mang chữ
    "spatial" từ cách hiểu cũ; giữ nguyên để model .joblib đã lưu vẫn load được.
    """

    def __init__(self, include_spatial_spectrum: bool = True, include_statistics: bool = True):
        self.include_spatial_spectrum = include_spatial_spectrum
        self.include_statistics = include_statistics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_channels = X_arr.shape

        feature_blocks = [X_arr]  # 1. Giữ nguyên 16 mẫu gốc

        # ── 2. Đặc trưng phổ tần số (Power Spectral Density) ──
        if self.include_spatial_spectrum:
            # DFT trên cửa sổ 16 mẫu (axis=1) → 9 bin; ở 256 Hz bin k ≈ 16·k Hz
            # Bin 0 = DC (trung bình cửa sổ + mọi thành phần < ~8 Hz)
            fft_vals = np.fft.rfft(X_arr, axis=1)
            psd = (np.abs(fft_vals) ** 2) / n_channels

            # Phổ công suất từng bin (n_channels//2 + 1 = 9 bins cho 16 mẫu)
            feature_blocks.append(psd)

            # Tổng công suất phổ
            total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10
            feature_blocks.append(total_power)

            # Phổ chuẩn hóa (phân phối xác suất trên các bin)
            norm_psd = psd / total_power

            # Spectral Entropy — năng lượng trải đều trên nhiều bin (cao) hay dồn vào vài bin (thấp)
            spectral_entropy = -np.sum(
                norm_psd * np.log(norm_psd + 1e-12), axis=1, keepdims=True
            )
            feature_blocks.append(spectral_entropy)

            # Spectral Centroid — trọng tâm tần số, tính theo chỉ số bin
            freq_indices = np.arange(psd.shape[1]).reshape(1, -1)
            spectral_centroid = (
                np.sum(psd * freq_indices, axis=1, keepdims=True) / total_power
            )
            feature_blocks.append(spectral_centroid)

            # Tỷ lệ công suất tần số cao / thấp (High-to-Low Frequency Power Ratio)
            # Chia tại midpoint: bins 0–3 (0–48 Hz) = thấp, bins 4–8 (64–128 Hz) = cao
            n_bins = psd.shape[1]  # 9
            mid = n_bins // 2  # 4
            low_power = np.sum(psd[:, :mid], axis=1, keepdims=True) + 1e-10
            high_power = np.sum(psd[:, mid:], axis=1, keepdims=True) + 1e-10
            hl_ratio = high_power / low_power
            feature_blocks.append(hl_ratio)

        # ── 3. Đặc trưng thống kê trong cửa sổ (Window Statistics) ──
        if self.include_statistics:
            # Sai phân bậc một giữa các mẫu liên tiếp (xấp xỉ đạo hàm theo thời gian)
            diffs = np.diff(X_arr, axis=1)
            feature_blocks.append(diffs)

            # Các mô-men thống kê qua 16 mẫu
            mean_val = np.mean(X_arr, axis=1, keepdims=True)
            std_val = np.std(X_arr, axis=1, keepdims=True)
            var_val = std_val ** 2  # Tránh tính lại từ đầu
            ptp_val = np.ptp(X_arr, axis=1, keepdims=True)
            energy_val = np.sum(X_arr ** 2, axis=1, keepdims=True)

            # Skewness & Kurtosis (suppress warning khi std=0)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                skew_val = stats.skew(X_arr, axis=1, bias=False).reshape(-1, 1)
                kurt_val = stats.kurtosis(X_arr, axis=1, bias=False).reshape(-1, 1)

            # Thay thế NaN/inf (khi std=0 → skew & kurtosis không xác định)
            skew_val = np.nan_to_num(skew_val, nan=0.0, posinf=0.0, neginf=0.0)
            kurt_val = np.nan_to_num(kurt_val, nan=0.0, posinf=0.0, neginf=0.0)

            feature_blocks.extend([
                mean_val, std_val, var_val, ptp_val, energy_val, skew_val, kurt_val
            ])

        final_features = np.hstack(feature_blocks)
        return final_features

def build_preprocessing_pipeline(
    include_spatial_spectrum: bool = True,
    include_statistics: bool = True,
) -> Pipeline:
    """
    Xây dựng Scikit-Learn Pipeline gồm bộ trích xuất đặc trưng phổ & thống kê và Scaler.
    """
    pipeline = Pipeline([
        ("spectral_spatial_extractor", EEGSpectralAndSpatialFeatureExtractor(
            include_spatial_spectrum=include_spatial_spectrum,
            include_statistics=include_statistics,
        )),
        ("scaler", RobustScaler())
    ])
    logger.info("Built spectral & statistical EEG pipeline: %s", list(pipeline.named_steps.keys()))
    return pipeline


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
    Trích xuất đặc trưng Spatial-Spectral và Thống kê từ 16 kênh EEG.

    Dữ liệu đầu vào: mỗi mẫu gồm 16 giá trị (X1–X16) đại diện cho 16 điện cực
    EEG **sắp xếp theo vị trí không gian** tại một thời điểm.

    Vì trục dữ liệu là **không gian** (spatial), phép biến đổi Fourier rời rạc
    (DFT) trên axis=1 tạo ra **phổ tần số không gian (Spatial Frequency Spectrum)**:
      - Tần số không gian **thấp** → mẫu hình hoạt động **lan rộng** (broad/generalized)
      - Tần số không gian **cao**  → mẫu hình hoạt động **khu trú** (focal/localized)

    Đặc trưng này có ý nghĩa lâm sàng cho phân loại động kinh:
      - Generalized seizure → năng lượng tập trung ở spatial freq thấp
      - Focal seizure       → năng lượng tập trung ở spatial freq cao
    """

    def __init__(self, include_spatial_spectrum: bool = True, include_statistics: bool = True):
        self.include_spatial_spectrum = include_spatial_spectrum
        self.include_statistics = include_statistics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_channels = X_arr.shape

        feature_blocks = [X_arr]  # 1. Giữ nguyên 16 kênh gốc

        # ── 2. Đặc trưng phổ tần số không gian (Spatial Frequency Spectrum) ──
        if self.include_spatial_spectrum:
            # DFT trên trục không gian (axis=1): biến đổi 16 kênh → 9 spatial freq bins
            # Bin 0 = DC (trung bình toàn cục), bin 1..8 = tần số không gian tăng dần
            fft_vals = np.fft.rfft(X_arr, axis=1)
            spatial_psd = (np.abs(fft_vals) ** 2) / n_channels  # Spatial Power Spectral Density

            # Phổ công suất từng bin (n_channels//2 + 1 = 9 bins cho 16 kênh)
            feature_blocks.append(spatial_psd)

            # Tổng công suất phổ không gian
            total_power = np.sum(spatial_psd, axis=1, keepdims=True) + 1e-10
            feature_blocks.append(total_power)

            # Phổ chuẩn hóa (phân phối xác suất trên các spatial freq bins)
            norm_psd = spatial_psd / total_power

            # Spatial Spectral Entropy — đo mức phân tán năng lượng trên các tần số không gian
            # Entropy cao → hoạt động phân tán đều → Generalized | Entropy thấp → khu trú → Focal
            spatial_entropy = -np.sum(
                norm_psd * np.log(norm_psd + 1e-12), axis=1, keepdims=True
            )
            feature_blocks.append(spatial_entropy)

            # Spatial Spectral Centroid — trọng tâm tần số không gian
            freq_indices = np.arange(spatial_psd.shape[1]).reshape(1, -1)
            spatial_centroid = (
                np.sum(spatial_psd * freq_indices, axis=1, keepdims=True) / total_power
            )
            feature_blocks.append(spatial_centroid)

            # Tỷ lệ năng lượng spatial freq cao / thấp (High-to-Low Spatial Frequency Ratio)
            # Chia tại midpoint: bins 0–4 = broad patterns, bins 5–8 = focal patterns
            n_bins = spatial_psd.shape[1]  # 9
            mid = n_bins // 2  # 4
            low_spatial_power = np.sum(spatial_psd[:, :mid], axis=1, keepdims=True) + 1e-10
            high_spatial_power = np.sum(spatial_psd[:, mid:], axis=1, keepdims=True) + 1e-10
            hl_ratio = high_spatial_power / low_spatial_power
            feature_blocks.append(hl_ratio)

        # ── 3. Đặc trưng thống kê đa kênh (Cross-Channel Statistics) ──
        if self.include_statistics:
            # Gradient không gian giữa các điện cực lân cận
            diffs = np.diff(X_arr, axis=1)
            feature_blocks.append(diffs)

            # Các mô-men thống kê qua 16 kênh
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
    Xây dựng Scikit-Learn Pipeline tích hợp Spatial Spectral Feature Extractor và Scaler.
    """
    pipeline = Pipeline([
        ("spectral_spatial_extractor", EEGSpectralAndSpatialFeatureExtractor(
            include_spatial_spectrum=include_spatial_spectrum,
            include_statistics=include_statistics,
        )),
        ("scaler", RobustScaler())
    ])
    logger.info("Built Spatial-Spectral EEG pipeline: %s", list(pipeline.named_steps.keys()))
    return pipeline


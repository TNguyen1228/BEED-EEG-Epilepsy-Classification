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
    Extracts comprehensive Spectral (PSD, Spectral Entropy, Energy, Centroid)
    and Spatial (Moments, Gradients) features from 16-channel EEG signals.
    """
    def __init__(self, include_psd: bool = True, include_spatial: bool = True):
        self.include_psd = include_psd
        self.include_spatial = include_spatial

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_channels = X_arr.shape
        
        feature_blocks = [X_arr]  # 1. Giữ nguyên 16 kênh gốc
        
        # 2. Trích xuất đặc trưng Spectral & PSD qua Biến đổi Fourier (FFT)
        if self.include_psd:
            # Tính FFT thực (Real FFT) qua các kênh
            fft_vals = np.fft.rfft(X_arr, axis=1)
            psd = (np.abs(fft_vals) ** 2) / n_channels  # Mật độ phổ công suất (PSD)
            
            # Phổ công suất từng bin tần số (n_channels//2 + 1 bins = 9 bins)
            feature_blocks.append(psd)
            
            # Tổng công suất phổ (Total Spectral Power)
            total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10
            feature_blocks.append(total_power)
            
            # Mật độ phổ chuẩn hóa (Normalized PSD distribution)
            norm_psd = psd / total_power
            
            # Spectral Entropy (Độ hỗn loạn phổ - phân biệt co giật thật và nhiễu)
            spectral_entropy = -np.sum(norm_psd * np.log(norm_psd + 1e-12), axis=1, keepdims=True)
            feature_blocks.append(spectral_entropy)
            
            # Spectral Centroid (Trọng tâm tần số)
            freq_indices = np.arange(psd.shape[1]).reshape(1, -1)
            spectral_centroid = np.sum(psd * freq_indices, axis=1, keepdims=True) / total_power
            feature_blocks.append(spectral_centroid)
            
            # Tỷ lệ năng lượng tần số cao / tần số thấp (High-to-Low frequency power ratio)
            low_freq_power = np.sum(psd[:, :3], axis=1, keepdims=True) + 1e-10
            high_freq_power = np.sum(psd[:, 3:], axis=1, keepdims=True) + 1e-10
            hl_ratio = high_freq_power / low_freq_power
            feature_blocks.append(hl_ratio)

        # 3. Trích xuất đặc trưng Không gian & Thống kê đa kênh (Spatial & Statistical)
        if self.include_spatial:
            # Gradient không gian giữa các điện cực lân cận (Spatial differences)
            diffs = np.diff(X_arr, axis=1)
            feature_blocks.append(diffs)
            
            # Các mô-men thống kê
            mean_val = np.mean(X_arr, axis=1, keepdims=True)
            std_val = np.std(X_arr, axis=1, keepdims=True)
            var_val = np.var(X_arr, axis=1, keepdims=True)
            ptp_val = (np.max(X_arr, axis=1, keepdims=True) - np.min(X_arr, axis=1, keepdims=True))
            energy_val = np.sum(X_arr ** 2, axis=1, keepdims=True)
            
            # Skewness & Kurtosis
            skew_val = stats.skew(X_arr, axis=1, bias=False).reshape(-1, 1)
            kurt_val = stats.kurtosis(X_arr, axis=1, bias=False).reshape(-1, 1)
            
            # Thay thế NaN (nếu có khi std=0)
            skew_val = np.nan_to_num(skew_val, nan=0.0)
            kurt_val = np.nan_to_num(kurt_val, nan=0.0)
            
            feature_blocks.extend([mean_val, std_val, var_val, ptp_val, energy_val, skew_val, kurt_val])
            
        final_features = np.hstack(feature_blocks)
        return final_features

def build_preprocessing_pipeline(include_psd: bool = True, include_spatial: bool = True) -> Pipeline:
    """
    Xây dựng Scikit-Learn Pipeline tích hợp PSD Feature Extractor và Scaler.
    """
    pipeline = Pipeline([
        ("spectral_spatial_extractor", EEGSpectralAndSpatialFeatureExtractor(
            include_psd=include_psd,
            include_spatial=include_spatial
        )),
        ("scaler", RobustScaler())
    ])
    logger.info("Built enhanced PSD & Spatial EEG pipeline: %s", pipeline.named_steps.keys())
    return pipeline


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô hình ANFIS theo nhóm đặc trưng kết hợp nhánh chuỗi, phiên bản đã làm sạch.

Các điểm sửa chính so với script gốc:
- chia train/validation/test theo thứ tự thời gian
- bộ chuẩn hóa đặc trưng chỉ được fit trên train windows
- validation tách riêng khỏi test
- chọn run tốt nhất theo validation loss, không theo chỉ số test
- tâm mờ có thứ tự để LOW/HIGH giữ đúng ngữ nghĩa
- tham số hóa mục tiêu bảo toàn tính hợp lệ của OHLC
- xuất đầy đủ cả tiền đề lẫn hệ quả của luật
- đường dẫn dữ liệu trong `Dataset/` có thể cấu hình

Luồng xử lý mức cao của file này:
1. Nạp một file CSV có tối thiểu các cột Open/High/Low/Close.
2. Tạo một bộ đặc trưng "lõi" gọn cho nhánh mờ.
3. Xây dựng mục tiêu theo cách tham số hóa bảo đảm dựng lại OHLC hợp lệ.
4. Tạo các cửa sổ trượt và chia chúng theo thời gian thành train/validation/test.
5. Chỉ fit scaler trên train windows, sau đó dùng đúng scaler đó cho val/test.
6. Xây dựng mô hình lai:
   - nhánh mờ đọc bước thời gian cuối của đặc trưng lõi
   - nhánh chuỗi đọc toàn bộ cửa sổ
   - cả hai nhánh cùng dự đoán tham số mục tiêu rồi được cộng lại
7. Chọn run tốt nhất chỉ bằng validation loss.
8. Dựng lại giá OHLC từ các tham số mục tiêu đã dự đoán.
9. Xuất chỉ số đánh giá, luật, và một artifact giải thích cho một mẫu dữ liệu.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# RETURN_CLIP chặn hai mục tiêu có dấu:
# - close_ret: giá đóng cửa ngày kế tiếp so với giá đóng cửa hiện tại
# - open_gap : giá mở cửa ngày kế tiếp so với giá đóng cửa hiện tại
# Mục tiêu không phải là khẳng định "trong tài chính thì 50% là bất khả thi",
# mà là giữ mạng trong một miền số học ổn định khi huấn luyện.
RETURN_CLIP = 0.5

# BUFFER_CLIP chặn hai mục tiêu hình học nến không âm:
# - high_buffer: mức High cao hơn max(Open, Close) bao nhiêu
# - low_buffer : mức Low thấp hơn min(Open, Close) bao nhiêu
# Chúng được mô hình hóa trong không gian log để luôn dương sau khi dựng lại.
BUFFER_CLIP = 0.5

# CORE_FEATURE_NAMES là các đặc trưng đi vào nhánh mờ.
# Kể cả khi có thêm biến ngoại sinh, các khối ANFIS vẫn chỉ nhìn thấy
# sáu đặc trưng lõi có thể diễn giải này.
CORE_FEATURE_NAMES = [
    "close_ret",
    "open_gap",
    "high_buffer",
    "low_buffer",
    "range_pct",
    "volume_ret",
]
TARGET_NAMES = [
    "target_close_ret",
    "target_open_gap",
    "target_high_buffer",
    "target_low_buffer",
]

# Ta giữ thứ tự chuẩn này ở mọi nơi khi báo cáo các chỉ số theo giá.
PRICE_NAMES = ["Open", "High", "Low", "Close"]


def set_seed(seed: int) -> None:
    # Đặt tất cả seed ngẫu nhiên dễ thấy để những lần chạy lặp lại
    # tái lập được nhiều nhất có thể trong phạm vi TensorFlow và phần cứng hiện có.
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_runtime() -> None:
    # Trên Apple Silicon hoặc môi trường có GPU, bật chế độ memory growth giúp
    # TensorFlow không cố chiếm toàn bộ bộ nhớ ngay từ đầu.
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # Nếu không cấu hình được memory growth cho GPU thì cũng không nên
        # làm script dừng; mô hình vẫn có thể chạy trên CPU.
        pass


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    # Ta tham số hóa các đại lượng dương (khoảng cách giữa các tâm, độ rộng)
    # bằng softplus(raw). Khi đã biết giá trị khởi tạo dương mong muốn là x,
    # ta cần ánh xạ ngược để khởi tạo biến raw.
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 1e-6)
    return np.log(np.expm1(x))


def to_jsonable(value):
    # Tiện ích dùng khi lưu các tệp đầu ra. Nhiều đối tượng trong quá trình huấn luyện
    # chứa scalar hoặc mảng NumPy mà json.dump không tuần tự hóa trực tiếp được.
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class PreparedData:
    # Lớp dữ liệu này gom mọi thứ do pipeline dữ liệu tạo ra để phần còn lại
    # của script không phải truyền tay quá nhiều mảng riêng lẻ.
    stock_name: str
    dates: np.ndarray
    seq_feature_names: List[str]
    core_feature_names: List[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    current_close_train: np.ndarray
    current_close_val: np.ndarray
    current_close_test: np.ndarray
    current_ohlc_test: np.ndarray
    actual_next_ohlc_test: np.ndarray
    feature_scaler_mean: np.ndarray
    feature_scaler_scale: np.ndarray


@tf.keras.utils.register_keras_serializable(package="tsa")
class OrderedFeatureGroupANFIS(layers.Layer):
    """
    Khối ANFIS với các tâm được giữ theo thứ tự.

    Vì sao "có thứ tự" lại quan trọng:
    - Trong script gốc, nhãn LOW/HIGH có thể âm thầm mất nghĩa sau huấn luyện
      vì các tâm được phép vượt qua nhau.
    - Ở đây ta học một tâm gốc cộng với các khoảng dương, nên thứ tự được giữ lại.

    Lớp này trả về gì:
    - Một vector ẩn dày đặc được tạo từ các đầu ra luật có trọng số kiểu Sugeno.
    - Nếu cần, trả thêm cường độ kích hoạt đã chuẩn hóa và đầu ra hệ quả của từng luật.
    """

    def __init__(
        self,
        n_mfs: int = 2,
        output_dim: int = 4,
        name_prefix: str = "anfis",
        initial_centers: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.initial_centers = None if initial_centers is None else np.asarray(initial_centers, dtype=np.float32)

    def build(self, input_shape):
        # n_features là số chiều của nhóm đặc trưng cục bộ đi vào khối ANFIS này
        # (4 cho nhóm lợi suất/biến động giá, 2 cho nhóm chỉ báo).
        n_features = int(input_shape[-1])
        self.n_features = n_features

        # Số luật theo lưới đầy đủ: với 2 hàm thuộc và 4 đầu vào ta có 2^4 = 16 luật.
        self.n_rules = self.n_mfs ** n_features

        if self.initial_centers is None:
            initial_centers = np.sort(
                np.random.uniform(-1.0, 1.0, size=(n_features, self.n_mfs)).astype(np.float32),
                axis=1,
            )
        else:
            initial_centers = np.sort(self.initial_centers.astype(np.float32), axis=1)

        # Thay vì học từng tâm một cách độc lập, ta học:
        # - một tâm gốc
        # - các độ lệch dương tới những tâm tiếp theo
        # Cách này bảo đảm center[0] <= center[1] <= ... và giữ đúng thứ tự nhãn.
        base_init = initial_centers[:, :1]
        delta_init = np.diff(initial_centers, axis=1)
        delta_init = np.maximum(delta_init, 1e-3)
        width_init = np.full((n_features, self.n_mfs), 0.5, dtype=np.float32)

        # center_base lưu tâm ngoài cùng bên trái của mỗi đặc trưng.
        self.center_base = self.add_weight(
            name=f"{self.name_prefix}_center_base",
            shape=(n_features, 1),
            initializer=tf.constant_initializer(base_init),
            trainable=True,
        )
        if self.n_mfs > 1:
            # center_delta_raw được biến thành các khoảng dương nhờ softplus.
            self.center_delta_raw = self.add_weight(
                name=f"{self.name_prefix}_center_delta_raw",
                shape=(n_features, self.n_mfs - 1),
                initializer=tf.constant_initializer(inverse_softplus(delta_init)),
                trainable=True,
            )
        else:
            self.center_delta_raw = None

        # width_raw cũng được ánh xạ qua softplus để độ rộng Gaussian
        # luôn dương chặt trong suốt quá trình huấn luyện.
        self.width_raw = self.add_weight(
            name=f"{self.name_prefix}_width_raw",
            shape=(n_features, self.n_mfs),
            initializer=tf.constant_initializer(inverse_softplus(width_init)),
            trainable=True,
        )

        # Các tham số hệ quả kiểu Sugeno:
        # với mỗi luật và mỗi chiều đầu ra ẩn, ta học một hàm affine cục bộ
        # của các đầu vào.
        self.consequent_p = self.add_weight(
            name=f"{self.name_prefix}_consequent_p",
            shape=(self.n_rules, n_features, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.consequent_r = self.add_weight(
            name=f"{self.name_prefix}_consequent_r",
            shape=(self.n_rules, self.output_dim),
            initializer="zeros",
            trainable=True,
        )

        # Tính sẵn hàm thuộc mà mỗi luật dùng cho từng đặc trưng.
        # Nhờ vậy ta không phải dựng lại tổ hợp ở mỗi lần truyền thuận.
        self.rule_mf_indices = tf.constant(self._compute_rule_indices(n_features, self.n_mfs), dtype=tf.int32)
        super().build(input_shape)

    @staticmethod
    def _compute_rule_indices(n_features: int, n_mfs: int) -> np.ndarray:
        # Ví dụ với 2 đặc trưng, 2 hàm thuộc:
        # rule 0 -> [0, 0]
        # rule 1 -> [1, 0]
        # rule 2 -> [0, 1]
        # rule 3 -> [1, 1]
        # Thứ tự cụ thể không quá quan trọng về mặt khái niệm, miễn là ta
        # dùng nhất quán giữa cường độ kích hoạt và tham số hệ quả.
        indices = []
        for rule_idx in range(n_mfs ** n_features):
            rule_mfs = []
            temp = rule_idx
            for _ in range(n_features):
                rule_mfs.append(temp % n_mfs)
                temp //= n_mfs
            indices.append(rule_mfs)
        return np.asarray(indices, dtype=np.int32)

    def get_centers(self) -> tf.Tensor:
        # Khôi phục các tâm có thứ tự từ tâm gốc + các khoảng dương.
        if self.n_mfs == 1:
            return self.center_base
        gaps = tf.nn.softplus(self.center_delta_raw) + 1e-3
        offsets = tf.cumsum(gaps, axis=1)
        return tf.concat([self.center_base, self.center_base + offsets], axis=1)

    def get_widths(self) -> tf.Tensor:
        # Khôi phục các độ rộng Gaussian luôn dương.
        return tf.nn.softplus(self.width_raw) + 1e-3

    def call(self, inputs, return_details: bool = False):
        # inputs có dạng: (batch, n_features)
        batch_size = tf.shape(inputs)[0]

        # Mở rộng shape để có thể tính toàn bộ các hàm thuộc Gaussian
        # của mọi đặc trưng chỉ trong một biểu thức vector hóa.
        x_exp = tf.expand_dims(inputs, axis=2)
        centers = tf.expand_dims(self.get_centers(), axis=0)
        widths = tf.expand_dims(self.get_widths(), axis=0)

        # Lớp 1: mờ hóa.
        # memberships[b, i, j] = mức độ mà mẫu b thuộc hàm thuộc j của đặc trưng i.
        memberships = tf.exp(-tf.square(x_exp - centers) / (2.0 * tf.square(widths)))

        # Lớp 2: cường độ kích hoạt của luật.
        # Mỗi luật chọn một hàm thuộc cho mỗi đặc trưng rồi nhân các giá trị đó lại.
        firing = tf.ones((batch_size, self.n_rules), dtype=inputs.dtype)
        for feat_idx in range(self.n_features):
            mf_indices = self.rule_mf_indices[:, feat_idx]
            feat_memberships = memberships[:, feat_idx, :]
            rule_memberships = tf.gather(feat_memberships, mf_indices, axis=1)
            firing = firing * rule_memberships

        # Lớp 3: chuẩn hóa cường độ kích hoạt để tổng trọng số các luật bằng 1.
        firing_sum = tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum

        # Lớp 4: tính hệ quả affine kiểu Sugeno của từng luật.
        x_expanded = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=3)
        p_expanded = tf.expand_dims(self.consequent_p, axis=0)
        linear = tf.reduce_sum(x_expanded * p_expanded, axis=2)
        rule_outputs = linear + self.consequent_r

        # Lớp 5: cộng có trọng số các đầu ra của luật.
        output = tf.reduce_sum(tf.expand_dims(firing_norm, axis=2) * rule_outputs, axis=1)

        if return_details:
            return output, firing_norm, rule_outputs
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_mfs": self.n_mfs,
                "output_dim": self.output_dim,
                "name_prefix": self.name_prefix,
            }
        )
        return config


def parse_args() -> argparse.Namespace:
    # Giữ CLI tường minh để file này vừa có thể chạy như một tệp lệnh,
    # vừa đóng vai trò bộ chạy thí nghiệm có thể đọc hiểu được.
    parser = argparse.ArgumentParser(description="Cleaned feature-group ANFIS + sequence model")
    parser.add_argument("--data", type=str, default="Dataset/TSLA.csv", help="CSV path inside Dataset/")
    parser.add_argument("--output-dir", type=str, default="outputs_feature_group_clean", help="Directory for artifacts")
    parser.add_argument("--stock-name", type=str, default=None, help="Optional display name")
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--include-exog", action="store_true", help="Append exogenous columns to the sequence branch")
    parser.add_argument("--max-rows", type=int, default=None, help="Keep only the most recent N rows")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def load_market_dataframe(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    # Đọc CSV với giả định tối thiểu: chỉ bắt buộc có OHLC.
    df = pd.read_csv(path)
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    # Nếu thiếu Volume thì tạo một cột giả để phần sau của code vẫn dùng
    # được một công thức đặc trưng thống nhất.
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Date là tùy chọn, nhưng nếu có thì ta parse vì tinh thần của script
    # đã làm sạch này là tôn trọng thứ tự thời gian.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Luôn sắp xếp theo thời gian trước khi dựng window.
    df = df.sort_values("Date").reset_index(drop=True)

    # Chuyển mọi cột trừ Date sang số để tránh việc cột chuỗi lọt qua âm thầm.
    numeric_cols = [col for col in df.columns if col != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại các dòng có OHLC bất khả hoặc không dùng được.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)].reset_index(drop=True)

    # max_rows hữu ích khi cần kiểm tra nhanh trên đoạn dữ liệu mới nhất.
    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame, include_exog: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Hàm này định nghĩa đồng thời:
    # - các đặc trưng đi vào mô hình
    # - cách tham số hóa mục tiêu mà mô hình phải học
    #
    # Cách tham số hóa mục tiêu là thay đổi mô hình hóa quan trọng nhất:
    # thay vì dự đoán trực tiếp O/H/L/C một cách độc lập, ta dự đoán
    #   1. close ngày kế tiếp so với close hiện tại
    #   2. open ngày kế tiếp so với close hiện tại
    #   3. High cao hơn max(Open, Close) bao nhiêu
    #   4. Low thấp hơn min(Open, Close) bao nhiêu
    # Cách này bảo đảm nến OHLC dựng lại luôn hợp lệ về mặt hình học.
    work = df.copy()
    prev_close = work["Close"].shift(1)

    # Thân nến là đoạn giữa Open và Close.
    # Ta dùng body_high/body_low làm hình học gốc cho các mục tiêu râu nến.
    body_high = work[["Open", "Close"]].max(axis=1)
    body_low = work[["Open", "Close"]].min(axis=1)

    # Đặc trưng lõi 1: close hôm nay so với close hôm qua.
    work["close_ret"] = (work["Close"] / prev_close - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)

    # Đặc trưng lõi 2: open hôm nay so với close hôm qua.
    # Đây là một đặc trưng "gap" tự nhiên cho nến ngày.
    work["open_gap"] = (work["Open"] / prev_close - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)

    # Với các buffer của râu nến, ta ép phòng thủ các đẳng thức hình học:
    # High >= max(Open, Close), Low <= min(Open, Close).
    # Dữ liệu thị trường thật thường đã thỏa điều này, nhưng vẫn nên chặn
    # để các biểu thức log bên dưới không nhận tỉ số không hợp lệ.
    safe_high = np.maximum(work["High"].to_numpy(dtype=np.float32), body_high.to_numpy(dtype=np.float32))
    safe_low = np.minimum(work["Low"].to_numpy(dtype=np.float32), body_low.to_numpy(dtype=np.float32))
    safe_body_high = np.maximum(body_high.to_numpy(dtype=np.float32), 1e-8)
    safe_body_low = np.maximum(body_low.to_numpy(dtype=np.float32), 1e-8)

    # Đặc trưng lõi 3: độ dài râu trên trong không gian log.
    # Giá trị 0 nghĩa là High đúng bằng max(Open, Close).
    work["high_buffer"] = np.clip(np.log(safe_high / safe_body_high), 0.0, BUFFER_CLIP)

    # Đặc trưng lõi 4: độ dài râu dưới trong không gian log.
    # Giá trị 0 nghĩa là Low đúng bằng min(Open, Close).
    work["low_buffer"] = np.clip(np.log(safe_body_low / np.maximum(safe_low, 1e-8)), 0.0, BUFFER_CLIP)

    # Đặc trưng lõi 5: biên độ ngày dưới dạng tỉ lệ so với Close.
    work["range_pct"] = ((work["High"] - work["Low"]) / work["Close"]).clip(0.0, RETURN_CLIP)

    # Đặc trưng lõi 6: thay đổi của log-volume.
    # Ta dùng log1p trước vì khối lượng gốc có thể thay đổi rất mạnh về thang đo.
    work["volume_ret"] = np.log1p(work["Volume"].clip(lower=0)).diff().clip(-3.0, 3.0)

    # Các giá dịch sang ngày kế tiếp chính là mục tiêu có giám sát.
    next_close = work["Close"].shift(-1)
    next_open = work["Open"].shift(-1)
    next_high = work["High"].shift(-1)
    next_low = work["Low"].shift(-1)

    # Hình học thân nến của ngày kế tiếp.
    next_body_high = pd.concat([next_open, next_close], axis=1).max(axis=1)
    next_body_low = pd.concat([next_open, next_close], axis=1).min(axis=1)

    # Một lần nữa, ép hình học hợp lệ trước khi lấy log.
    safe_next_body_high = np.maximum(next_body_high.to_numpy(dtype=np.float32), 1e-8)
    safe_next_body_low = np.maximum(next_body_low.to_numpy(dtype=np.float32), 1e-8)
    safe_next_high = np.maximum(next_high.to_numpy(dtype=np.float32), safe_next_body_high)
    safe_next_low = np.minimum(next_low.to_numpy(dtype=np.float32), safe_next_body_low)

    # Mục tiêu 1: close ngày kế tiếp so với close hiện tại.
    work["target_close_ret"] = (next_close / work["Close"] - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)

    # Mục tiêu 2: open ngày kế tiếp so với close hiện tại.
    work["target_open_gap"] = (next_open / work["Close"] - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)

    # Mục tiêu 3/4: độ dài râu nến của ngày kế tiếp, trong không gian log và luôn không âm.
    work["target_high_buffer"] = np.clip(np.log(safe_next_high / safe_next_body_high), 0.0, BUFFER_CLIP)
    work["target_low_buffer"] = np.clip(np.log(safe_next_body_low / np.maximum(safe_next_low, 1e-8)), 0.0, BUFFER_CLIP)

    # Giữ lại giá thật của ngày kế tiếp để đánh giá cuối cùng sau khi dựng lại.
    work["next_Open"] = next_open
    work["next_High"] = next_high
    work["next_Low"] = next_low
    work["next_Close"] = next_close

    exog_cols: List[str] = []
    if include_exog:
        # Các biến ngoại sinh chỉ đi vào nhánh chuỗi.
        # Nhánh mờ được chủ ý giữ gắn với bộ sáu đặc trưng lõi gọn và dễ diễn giải.
        exog_cols = [col for col in work.columns if col.startswith("exog_")]
        if exog_cols:
            # Điền tiến/lùi là một lựa chọn thực dụng ở đây: nhờ đó nhánh chuỗi
            # có thể dùng bộ dữ liệu giàu hơn mà không gãy vì đặc trưng vĩ mô/sự kiện thưa.
            work[exog_cols] = work[exog_cols].replace([np.inf, -np.inf], np.nan)
            work[exog_cols] = work[exog_cols].ffill().bfill().fillna(0.0)

    # Loại các dòng vẫn chưa đủ điều kiện tạo một mẫu học có giám sát sạch.
    required_cols = CORE_FEATURE_NAMES + TARGET_NAMES + ["next_Open", "next_High", "next_Low", "next_Close", "Date"]
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=required_cols).reset_index(drop=True)

    # Nhánh chuỗi nhận các đặc trưng lõi cộng với đặc trưng ngoại sinh nếu có.
    seq_feature_cols = CORE_FEATURE_NAMES + exog_cols
    return work, CORE_FEATURE_NAMES, seq_feature_cols


def build_windows(
    engineered: pd.DataFrame,
    core_feature_names: Sequence[str],
    seq_feature_names: Sequence[str],
    look_back: int,
    train_ratio: float,
    val_ratio: float,
    stock_name: str,
) -> PreparedData:
    # Chuyển bảng đặc trưng đã dựng thành các cửa sổ trượt:
    # cửa sổ kết thúc ở thời điểm t -> mục tiêu mô tả cây nến tại thời điểm t+1.
    seq_values = engineered[list(seq_feature_names)].to_numpy(dtype=np.float32)
    targets = engineered[TARGET_NAMES].to_numpy(dtype=np.float32)
    current_close = engineered["Close"].to_numpy(dtype=np.float32)
    current_ohlc = engineered[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float32)
    next_ohlc = engineered[["next_Open", "next_High", "next_Low", "next_Close"]].to_numpy(dtype=np.float32)
    dates = engineered["Date"].to_numpy()

    X_raw: List[np.ndarray] = []
    y: List[np.ndarray] = []
    dates_used: List[np.datetime64] = []
    current_close_used: List[float] = []
    current_ohlc_used: List[np.ndarray] = []
    next_ohlc_used: List[np.ndarray] = []

    # Điểm kết thúc hợp lệ đầu tiên là look_back - 1.
    for idx in range(look_back - 1, len(engineered)):
        start = idx - look_back + 1
        X_raw.append(seq_values[start : idx + 1])
        y.append(targets[idx])
        dates_used.append(dates[idx])
        current_close_used.append(current_close[idx])
        current_ohlc_used.append(current_ohlc[idx])
        next_ohlc_used.append(next_ohlc[idx])

    X_raw_arr = np.asarray(X_raw, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    current_close_arr = np.asarray(current_close_used, dtype=np.float32)
    current_ohlc_arr = np.asarray(current_ohlc_used, dtype=np.float32)
    next_ohlc_arr = np.asarray(next_ohlc_used, dtype=np.float32)
    dates_arr = np.asarray(dates_used)

    # Cần đủ số lượng cửa sổ để phép chia theo thời gian còn có ý nghĩa.
    n_samples = len(X_raw_arr)
    if n_samples < 50:
        raise ValueError(f"Not enough usable windows after feature engineering: {n_samples}")

    # Chia theo thời gian, tuyệt đối không xáo trộn.
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n_samples - 1)

    if val_end <= train_end or n_samples - val_end < 1:
        raise ValueError("Invalid split sizes after windowing; adjust look_back/train_ratio/val_ratio.")

    X_train_raw = X_raw_arr[:train_end]
    X_val_raw = X_raw_arr[train_end:val_end]
    X_test_raw = X_raw_arr[val_end:]

    # Scaler chỉ được fit trên các cửa sổ của tập train.
    # Đây là một trong những sửa đổi phương pháp luận quan trọng nhất so với script gốc.
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))

    # Val/test chỉ được biến đổi bằng thống kê của tập train, không dùng thêm gì khác.
    X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape).astype(np.float32)
    X_val = scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape).astype(np.float32)

    return PreparedData(
        stock_name=stock_name,
        dates=dates_arr,
        seq_feature_names=list(seq_feature_names),
        core_feature_names=list(core_feature_names),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_arr[:train_end],
        y_val=y_arr[train_end:val_end],
        y_test=y_arr[val_end:],
        current_close_train=current_close_arr[:train_end],
        current_close_val=current_close_arr[train_end:val_end],
        current_close_test=current_close_arr[val_end:],
        current_ohlc_test=current_ohlc_arr[val_end:],
        actual_next_ohlc_test=next_ohlc_arr[val_end:],
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )


def compute_initial_centers(prepared: PreparedData, n_mfs: int) -> Tuple[np.ndarray, np.ndarray]:
    # Khởi tạo tâm mờ chỉ từ BƯỚC CUỐI của các cửa sổ train.
    # Cách này giữ lại trực giác ban đầu (khởi tạo theo dữ liệu) đồng thời
    # loại bỏ vấn đề rò rỉ dữ liệu có trong script cũ.
    last_step = prepared.X_train[:, -1, :]
    core_last = last_step[:, : len(prepared.core_feature_names)]
    returns_data = core_last[:, :4]
    indicator_data = core_last[:, 4:6]

    returns_centers = np.zeros((returns_data.shape[1], n_mfs), dtype=np.float32)
    indicator_centers = np.zeros((indicator_data.shape[1], n_mfs), dtype=np.float32)

    for feat_idx in range(returns_data.shape[1]):
        km = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        km.fit(returns_data[:, feat_idx : feat_idx + 1])
        returns_centers[feat_idx] = np.sort(km.cluster_centers_.reshape(-1))

    for feat_idx in range(indicator_data.shape[1]):
        km = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        km.fit(indicator_data[:, feat_idx : feat_idx + 1])
        indicator_centers[feat_idx] = np.sort(km.cluster_centers_.reshape(-1))

    return returns_centers, indicator_centers


def build_model(
    look_back: int,
    n_seq_features: int,
    n_mfs: int,
    lstm_units: int,
    dropout: float,
    learning_rate: float,
    returns_centers: np.ndarray,
    indicator_centers: np.ndarray,
) -> Model:
    # Kiến trúc có hai luồng thông tin:
    # 1. Nhánh mờ chỉ đọc bước cuối của sáu đặc trưng lõi.
    # 2. Nhánh chuỗi đọc toàn bộ cửa sổ, kể cả biến ngoại sinh nếu có.
    inputs = layers.Input(shape=(look_back, n_seq_features), name="sequence_input")

    # Nhánh mờ chỉ nên thấy các đặc trưng lõi có thể diễn giải, không phải phần
    # mở rộng ngoại sinh. Vì vậy ta cắt bước thời gian cuối và chỉ giữ
    # len(CORE_FEATURE_NAMES) kênh đầu tiên.
    last_core = layers.Lambda(lambda x: x[:, -1, : len(CORE_FEATURE_NAMES)], name="last_core_features")(inputs)

    # Nhóm 1: các đặc trưng kiểu lợi suất / có dấu.
    returns_features = layers.Lambda(lambda x: x[:, :4], name="returns_slice")(last_core)

    # Nhóm 2: các đặc trưng thiên về hình học nến / thang đo.
    indicator_features = layers.Lambda(lambda x: x[:, 4:6], name="indicator_slice")(last_core)

    # Mỗi khối ANFIS sinh ra một biểu diễn ẩn kích thước nhỏ.
    returns_anfis = OrderedFeatureGroupANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix="returns",
        initial_centers=returns_centers,
        name="anfis_returns",
    )(returns_features)
    indicators_anfis = OrderedFeatureGroupANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix="indicators",
        initial_centers=indicator_centers,
        name="anfis_indicators",
    )(indicator_features)

    # Ghép các biểu diễn mờ lại rồi ánh xạ chúng thành bốn tham số mục tiêu thô.
    # Đây là "ý kiến" của nhánh mờ về cây nến kế tiếp.
    fuzzy_hidden = layers.Concatenate(name="fuzzy_concat")([returns_anfis, indicators_anfis])
    fuzzy_hidden = layers.Dense(16, activation="tanh", name="fuzzy_hidden")(fuzzy_hidden)
    fuzzy_raw = layers.Dense(4, name="fuzzy_raw_params")(fuzzy_hidden)

    # Nhánh chuỗi học phần thông tin thời gian còn dư mà nhánh mờ cục bộ
    # tự nó không bắt được.
    sequence_hidden = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm_1",
    )(inputs)
    sequence_hidden = layers.Dropout(dropout, name="dropout_1")(sequence_hidden)
    sequence_hidden = layers.Bidirectional(
        layers.LSTM(max(lstm_units // 2, 8)),
        name="bilstm_2",
    )(sequence_hidden)
    sequence_hidden = layers.Dropout(dropout, name="dropout_2")(sequence_hidden)
    sequence_hidden = layers.Dense(16, activation="relu", name="sequence_hidden")(sequence_hidden)
    sequence_raw = layers.Dense(4, name="sequence_residual_params")(sequence_hidden)

    # Các tham số thô cuối cùng là tổ hợp cộng:
    # tiên nghiệm có cấu trúc từ nhánh mờ + phần hiệu chỉnh dư từ nhánh chuỗi.
    raw_params = layers.Add(name="raw_target_params")([fuzzy_raw, sequence_raw])

    # Chặn từng tham số mục tiêu vào miền hợp lệ của nó:
    # - lợi suất có dấu dùng tanh
    # - buffer râu nến không âm dùng sigmoid
    close_ret = layers.Lambda(lambda z: RETURN_CLIP * tf.tanh(z[:, 0:1]), name="close_ret_output")(raw_params)
    open_gap = layers.Lambda(lambda z: RETURN_CLIP * tf.tanh(z[:, 1:2]), name="open_gap_output")(raw_params)
    high_buffer = layers.Lambda(lambda z: BUFFER_CLIP * tf.sigmoid(z[:, 2:3]), name="high_buffer_output")(raw_params)
    low_buffer = layers.Lambda(lambda z: BUFFER_CLIP * tf.sigmoid(z[:, 3:4]), name="low_buffer_output")(raw_params)
    outputs = layers.Concatenate(name="target_output")([close_ret, open_gap, high_buffer, low_buffer])

    # Hàm mất mát chỉ là MSE trong không gian tham số mục tiêu.
    # Tính hợp lệ cứng về hình học đến từ chính cách tham số hóa, chứ không
    # phải từ một hạng phạt bổ sung.
    model = Model(inputs=inputs, outputs=outputs, name="clean_feature_group_anfis")
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def reconstruct_ohlc(current_close: np.ndarray, target_params: np.ndarray) -> np.ndarray:
    # Chuyển bốn tham số mục tiêu đã bị chặn về lại giá OHLC thật.
    #
    # target_params[:, 0] = lợi suất close so với close hiện tại
    # target_params[:, 1] = gap open so với close hiện tại
    # target_params[:, 2] = log buffer của râu trên
    # target_params[:, 3] = log buffer của râu dưới
    current_close = np.asarray(current_close, dtype=np.float32).reshape(-1, 1)
    close_ret = target_params[:, 0:1]
    open_gap = target_params[:, 1:2]
    high_buffer = target_params[:, 2:3]
    low_buffer = target_params[:, 3:4]

    # Trước hết dựng lại thân nến.
    pred_close = current_close * (1.0 + close_ret)
    pred_open = current_close * (1.0 + open_gap)

    # Sau đó dựng lại hình học râu nến hợp lệ quanh thân nến đó.
    body_high = np.maximum(pred_open, pred_close)
    body_low = np.minimum(pred_open, pred_close)
    pred_high = body_high * np.exp(high_buffer)
    pred_low = body_low * np.exp(-low_buffer)
    return np.concatenate([pred_open, pred_high, pred_low, pred_close], axis=1)


def evaluate_predictions(actual_ohlc: np.ndarray, pred_ohlc: np.ndarray, current_close: np.ndarray) -> Dict[str, object]:
    # Báo cáo cả chỉ số ở mức giá lẫn hai chỉ số kiểm tra hợp lý:
    # - độ chính xác hướng biến động
    # - tỉ lệ OHLC hợp lệ
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(PRICE_NAMES):
        actual = actual_ohlc[:, idx]
        pred = pred_ohlc[:, idx]
        mask = actual != 0
        mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0) if mask.any() else 0.0
        metrics[name] = {
            "RMSE": float(math.sqrt(mean_squared_error(actual, pred))),
            "MAE": float(mean_absolute_error(actual, pred)),
            "MAPE": mape,
            "R2": float(r2_score(actual, pred)),
        }

    # Độ chính xác hướng chỉ được báo cáo tương đối theo current close cho
    # Open/Close, vì như vậy phù hợp với cách tham số hóa mục tiêu hơn là
    # so Open với Open trước đó, v.v.
    close_da = float(
        np.mean(np.sign(actual_ohlc[:, 3] - current_close) == np.sign(pred_ohlc[:, 3] - current_close)) * 100.0
    )
    open_da = float(
        np.mean(np.sign(actual_ohlc[:, 0] - current_close) == np.sign(pred_ohlc[:, 0] - current_close)) * 100.0
    )
    validity_mask = (
        (pred_ohlc[:, 1] >= np.maximum(pred_ohlc[:, 0], pred_ohlc[:, 3]))
        & (pred_ohlc[:, 2] <= np.minimum(pred_ohlc[:, 0], pred_ohlc[:, 3]))
    )
    validity_rate = float(np.mean(validity_mask) * 100.0)

    return {
        "price_metrics": metrics,
        "close_direction_accuracy": close_da,
        "open_direction_accuracy": open_da,
        "ohlc_validity_rate": validity_rate,
    }


def build_decompose_model(model: Model) -> Model:
    """Mở lộ các nhánh bên trong để ta xem từng nhánh đóng góp gì.

    Mô hình gốc chỉ trả về các tham số mục tiêu cuối cùng sau khi bị chặn.
    Khi debug và giải thích, ta còn muốn thấy:
    1. các tham số ẩn do nhánh mờ sinh ra,
    2. phần hiệu chỉnh dư do nhánh chuỗi sinh ra,
    3. đầu ra cuối cùng sau khi hai phần này được kết hợp.
    """
    return Model(
        inputs=model.input,
        outputs={
            # Đầu ra của nhánh suy luận mờ tường minh.
            "fuzzy_raw_params": model.get_layer("fuzzy_raw_params").output,
            # Đầu ra của nhánh BiLSTM dùng để sửa phần mà nhánh mờ còn bỏ sót.
            "sequence_residual_params": model.get_layer("sequence_residual_params").output,
            # Tham số mục tiêu cuối cùng dùng để dựng lại giá OHLC.
            "target_output": model.output,
        },
        name="decompose_model",
    )


def membership_labels(n_mfs: int) -> List[str]:
    # Các nhãn này chỉ phục vụ báo cáo khả năng diễn giải.
    # Chúng dựa trên việc lớp ANFIS giữ thứ tự các tâm, nên hàm thuộc đầu tiên
    # thật sự là vùng "thấp nhất" trên trục của đặc trưng.
    if n_mfs == 2:
        return ["LOW", "HIGH"]
    if n_mfs == 3:
        return ["LOW", "MEDIUM", "HIGH"]
    return [f"MF_{idx + 1}" for idx in range(n_mfs)]


def extract_layer_rules(
    layer: OrderedFeatureGroupANFIS,
    feature_names: Sequence[str],
    latent_names: Sequence[str],
) -> Dict[str, object]:
    """Chuyển một khối ANFIS thành mô tả luật dễ lưu dưới dạng JSON."""

    # Kéo các tham số đã học từ biến TensorFlow sang mảng NumPy
    # để thuận tiện cho việc tuần tự hóa và quan sát.
    labels = membership_labels(layer.n_mfs)
    centers = layer.get_centers().numpy()
    widths = layer.get_widths().numpy()
    rule_indices = layer.rule_mf_indices.numpy()
    p = layer.consequent_p.numpy()
    r = layer.consequent_r.numpy()

    rules = []
    for rule_idx in range(layer.n_rules):
        antecedents = []
        for feat_idx, feature_name in enumerate(feature_names):
            # Mỗi luật chọn đúng một tập mờ cho mỗi đặc trưng.
            # Ví dụ: close_return là HIGH và low_return là LOW.
            mf_idx = int(rule_indices[rule_idx, feat_idx])
            antecedents.append(
                {
                    "feature": feature_name,
                    "label": labels[mf_idx],
                    "center": float(centers[feat_idx, mf_idx]),
                    "width": float(widths[feat_idx, mf_idx]),
                }
            )

        consequents = {}
        for out_idx, latent_name in enumerate(latent_names):
            # Hệ quả Sugeno: một hàm affine của đầu vào ứng với luật này.
            coeffs = {feature_names[i]: float(p[rule_idx, i, out_idx]) for i in range(len(feature_names))}
            bias = float(r[rule_idx, out_idx])
            terms = [f"{bias:+.6f}"] + [f"{coeff:+.6f}*{name}" for name, coeff in coeffs.items()]
            consequents[latent_name] = {
                "bias": bias,
                "coefficients": coeffs,
                "formula": " ".join(terms).replace("+ -", "- "),
            }

        # Giữ lại cả bản câu chữ dễ đọc lẫn bản cấu trúc của luật.
        rule_text = "IF " + " AND ".join(f"{item['feature']} is {item['label']}" for item in antecedents)
        rules.append(
            {
                "rule_index": rule_idx + 1,
                "text": rule_text,
                "antecedents": antecedents,
                "consequents": consequents,
            }
        )

    return {
        "n_rules": layer.n_rules,
        "rules": rules,
        "labels_note": "Ordered centers keep LOW/HIGH semantics inside this ANFIS block.",
    }


def extract_rules(model: Model, feature_names: Sequence[str]) -> Dict[str, object]:
    """Xuất riêng cả hai hệ con mờ để tiện quan sát về sau."""

    # Mô hình có hai khối mờ với hai nhóm đặc trưng khác nhau:
    # nhóm biến động giá kiểu lợi suất và nhóm đặc trưng kiểu chỉ báo tóm tắt.
    returns_layer = model.get_layer("anfis_returns")
    indicators_layer = model.get_layer("anfis_indicators")
    return {
        "explainability_scope": (
            "Rules below describe the internal fuzzy blocks. The final model output is the sum "
            "of a fuzzy branch and a sequence residual branch."
        ),
        "returns_anfis": extract_layer_rules(
            returns_layer,
            feature_names[:4],
            [f"returns_latent_{idx + 1}" for idx in range(returns_layer.output_dim)],
        ),
        "indicators_anfis": extract_layer_rules(
            indicators_layer,
            feature_names[4:6],
            [f"indicators_latent_{idx + 1}" for idx in range(indicators_layer.output_dim)],
        ),
    }


def analyze_sample(model: Model, sample: np.ndarray, feature_names: Sequence[str]) -> Dict[str, object]:
    """Mổ xẻ một mẫu dữ liệu xuyên suốt toàn bộ mô hình lai.

    Hàm này phục vụ giải thích nhiều hơn là huấn luyện. Ta ghi lại:
    - các đặc trưng lõi cuối cùng mà nhánh mờ nhìn thấy,
    - đầu ra thô của nhánh mờ,
    - phần hiệu chỉnh dư của nhánh chuỗi,
    - các tham số mục tiêu cuối cùng sau khi bị chặn,
    - những luật có ảnh hưởng mạnh nhất trong từng khối ANFIS.
    """
    returns_layer: OrderedFeatureGroupANFIS = model.get_layer("anfis_returns")
    indicators_layer: OrderedFeatureGroupANFIS = model.get_layer("anfis_indicators")
    decompose_model = build_decompose_model(model)

    sample = np.asarray(sample, dtype=np.float32)
    # Chỉ bước thời gian cuối của sáu đặc trưng lõi đi vào các khối ANFIS.
    # Toàn bộ chuỗi vẫn được nhánh BiLSTM dùng thông qua decompose_model.
    last_core = sample[:, -1, : len(CORE_FEATURE_NAMES)]
    returns_input = last_core[:, :4]
    indicators_input = last_core[:, 4:6]

    # Yêu cầu trực tiếp từ lớp ANFIS các chi tiết bên trong:
    # cường độ kích hoạt đã chuẩn hóa và đầu ra hệ quả của từng luật.
    _, returns_firing, returns_rule_outputs = returns_layer(returns_input, return_details=True)
    _, indicators_firing, indicators_rule_outputs = indicators_layer(indicators_input, return_details=True)

    # Cho mẫu đi qua mô hình phụ decomposition để nhìn rõ
    # đầu ra cuối cùng được tách thành phần mờ + phần dư chuỗi như thế nào.
    outputs = decompose_model.predict(sample, verbose=0)

    def top_rules(layer_firing, layer_rule_outputs, prefix: str, top_k: int) -> List[Dict[str, object]]:
        # Một luật có thể kích hoạt mạnh nhưng đóng góp nhỏ nếu đầu ra hệ quả
        # của nó gần 0. Để tránh nhầm lẫn đó, ta xếp hạng theo điểm đóng góp
        # đơn giản chứ không chỉ theo cường độ kích hoạt.
        firing_np = layer_firing.numpy()[0]
        rule_output_np = layer_rule_outputs.numpy()[0]
        contributions = np.abs(firing_np[:, None] * rule_output_np).sum(axis=1)
        top_indices = np.argsort(contributions)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(
                {
                    "rule_index": int(idx + 1),
                    "firing_strength": float(firing_np[idx]),
                    "contribution_score": float(contributions[idx]),
                    "latent_output": rule_output_np[idx].tolist(),
                    "prefix": prefix,
                }
            )
        return results

    return {
        "core_feature_names": list(feature_names),
        "last_core_features_scaled": last_core[0].tolist(),
        "fuzzy_raw_params": outputs["fuzzy_raw_params"][0].tolist(),
        "sequence_residual_params": outputs["sequence_residual_params"][0].tolist(),
        "bounded_target_params": outputs["target_output"][0].tolist(),
        "top_return_rules": top_rules(returns_firing, returns_rule_outputs, "returns", top_k=3),
        "top_indicator_rules": top_rules(indicators_firing, indicators_rule_outputs, "indicators", top_k=2),
    }


def train_one_run(
    prepared: PreparedData,
    n_mfs: int,
    lstm_units: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    run_seed: int,
    verbose: int,
) -> Tuple[Model, Dict[str, List[float]], float, float]:
    """Huấn luyện một lần khởi tạo mô hình và báo cáo kết quả theo validation."""

    # Đặt lại seed ở đây để mỗi run đều được kiểm soát và có thể tái lập.
    set_seed(run_seed)
    # Tâm mờ ban đầu chỉ được ước lượng từ dữ liệu train.
    # Cách này giữ thông tin validation/test nằm ngoài bước khởi tạo mô hình.
    returns_centers, indicator_centers = compute_initial_centers(prepared, n_mfs=n_mfs)
    model = build_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        n_mfs=n_mfs,
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate,
        returns_centers=returns_centers,
        indicator_centers=indicator_centers,
    )

    # Cả hai callback đều chỉ theo dõi validation loss.
    # Đây là một sửa đổi phương pháp luận then chốt so với bản gốc.
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    ]

    start = time.time()
    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        # Validation dùng một phần hold-out riêng theo thời gian, không dùng test.
        validation_data=(prepared.X_val, prepared.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    train_time = time.time() - start
    # Vì restore_best_weights=True, mô hình trả về đã mang trọng số tốt nhất
    # theo validation, không nhất thiết là trọng số của epoch cuối.
    best_val_loss = float(min(history.history["val_loss"]))
    return model, history.history, best_val_loss, train_time


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    """Chạy toàn bộ pipeline thí nghiệm từ CSV đến đánh giá cuối cùng."""

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    stock_name = args.stock_name or data_path.stem
    # Luồng tiền xử lý được viết tường minh có chủ đích để người đọc lần ra
    # mỗi tensor xuất phát từ đâu.
    df = load_market_dataframe(data_path, max_rows=args.max_rows)
    engineered, core_feature_names, seq_feature_names = engineer_features(df, include_exog=args.include_exog)
    prepared = build_windows(
        engineered=engineered,
        core_feature_names=core_feature_names,
        seq_feature_names=seq_feature_names,
        look_back=args.look_back,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stock_name=stock_name,
    )

    best_model: Optional[Model] = None
    best_history: Optional[Dict[str, List[float]]] = None
    best_val_loss = float("inf")
    best_run = -1
    best_train_time = 0.0
    run_summaries: List[Dict[str, object]] = []

    for run_idx in range(args.runs):
        # Các run khác nhau chỉ bởi seed / khởi tạo, không phải bởi cách chia dữ liệu.
        run_seed = args.seed + run_idx
        model, history, val_loss, train_time = train_one_run(
            prepared=prepared,
            n_mfs=args.n_mfs,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_seed=run_seed,
            verbose=args.verbose,
        )

        # Chỉ số validation được tính để so sánh giữa các run.
        # Chỉ số test được giữ nguyên, chưa đụng tới cho đến khi chốt run thắng.
        val_pred_targets = model.predict(prepared.X_val, verbose=0)
        val_pred_ohlc = reconstruct_ohlc(prepared.current_close_val, val_pred_targets)
        val_actual_ohlc = reconstruct_ohlc(prepared.current_close_val, prepared.y_val)
        val_metrics = evaluate_predictions(val_actual_ohlc, val_pred_ohlc, prepared.current_close_val)

        summary = {
            "run": run_idx + 1,
            "seed": run_seed,
            "best_val_loss": val_loss,
            "epochs_trained": len(history["loss"]),
            "training_time_sec": round(train_time, 2),
            "validation_close_r2": val_metrics["price_metrics"]["Close"]["R2"],
            "validation_ohlc_validity_rate": val_metrics["ohlc_validity_rate"],
        }
        run_summaries.append(summary)

        # Việc chọn mô hình dựa chặt chẽ trên validation loss.
        # Nhờ đó tránh được thiên lệch lạc quan do chọn theo điểm test.
        if val_loss < best_val_loss:
            best_model = model
            best_history = history
            best_val_loss = val_loss
            best_run = run_idx + 1
            best_train_time = train_time

    if best_model is None or best_history is None:
        raise RuntimeError("Training did not produce a best model.")

    # Chỉ đến đây ta mới đánh giá trên khối test đúng một lần, sau khi mọi lựa chọn đã cố định.
    test_target_pred = best_model.predict(prepared.X_test, verbose=0)
    test_price_pred = reconstruct_ohlc(prepared.current_close_test, test_target_pred)
    test_metrics = evaluate_predictions(prepared.actual_next_ohlc_test, test_price_pred, prepared.current_close_test)

    # Xuất cả các tệp giải thích ở mức toàn cục lẫn cục bộ.
    rules = extract_rules(best_model, core_feature_names)
    sample_analysis = analyze_sample(best_model, prepared.X_test[:1], core_feature_names)

    return {
        "stock_name": stock_name,
        "prepared": prepared,
        "best_model": best_model,
        "best_history": best_history,
        "best_run": best_run,
        "best_val_loss": best_val_loss,
        "best_train_time": best_train_time,
        "run_summaries": run_summaries,
        "test_target_pred": test_target_pred,
        "test_price_pred": test_price_pred,
        "test_metrics": test_metrics,
        "rules": rules,
        "sample_analysis": sample_analysis,
    }


def save_artifacts(args: argparse.Namespace, results: Dict[str, object]) -> Path:
    """Lưu mô hình đã huấn luyện cùng các tệp giải thích/đánh giá đi kèm."""

    output_root = Path(args.output_dir)
    stock_dir = output_root / results["stock_name"]
    stock_dir.mkdir(parents=True, exist_ok=True)

    model_path = stock_dir / f"{results['stock_name']}_clean_feature_group.keras"
    # Lưu toàn bộ mô hình Keras để giữ nguyên trọng số ANFIS tùy biến và cấu trúc mạng.
    results["best_model"].save(model_path)

    metrics_path = stock_dir / "metrics.json"
    rules_path = stock_dir / "rules.json"
    sample_path = stock_dir / "sample_analysis.json"
    config_path = stock_dir / "training_config.json"
    history_path = stock_dir / "history.json"

    # Lưu cấu hình thực tế dùng để chạy ngay cạnh các tệp đầu ra để tiện tái lập.
    config_payload = vars(args).copy()
    config_payload.update(
        {
            "stock_name": results["stock_name"],
            "seq_feature_names": results["prepared"].seq_feature_names,
            "core_feature_names": results["prepared"].core_feature_names,
        }
    )

    # Metrics trả lời câu hỏi "mô hình làm tốt đến đâu?";
    # rules/sample analysis trả lời câu hỏi "bên trong mô hình đang làm gì?"
    metrics_payload = {
        "stock_name": results["stock_name"],
        "best_run": results["best_run"],
        "best_val_loss": results["best_val_loss"],
        "best_train_time_sec": round(results["best_train_time"], 2),
        "run_summaries": results["run_summaries"],
        "test_metrics": results["test_metrics"],
        "explainability_note": results["rules"]["explainability_scope"],
    }

    history_payload = {
        "history": results["best_history"],
    }

    # Chuyển các đối tượng NumPy/TensorFlow sang container Python thuần trước khi xuất JSON.
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics_payload), handle, indent=2)
    with open(rules_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results["rules"]), handle, indent=2)
    with open(sample_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results["sample_analysis"]), handle, indent=2)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(config_payload), handle, indent=2)
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(history_payload), handle, indent=2)

    return stock_dir


def print_summary(results: Dict[str, object], artifact_dir: Path) -> None:
    """In ra phần tóm tắt ngắn gọn trên terminal sau khi đã lưu các tệp đầu ra."""

    metrics = results["test_metrics"]
    print("\n" + "=" * 88)
    print(f"CLEAN FEATURE-GROUP ANFIS SUMMARY - {results['stock_name']}")
    print("=" * 88)
    print(f"Best run (selected on validation only): {results['best_run']}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Artifacts: {artifact_dir}")
    print("")
    for name in PRICE_NAMES:
        item = metrics["price_metrics"][name]
        print(
            f"{name:5s} | RMSE={item['RMSE']:.4f} | MAE={item['MAE']:.4f} | "
            f"MAPE={item['MAPE']:.2f}% | R2={item['R2']:.4f}"
        )
    print("")
    print(f"Close directional accuracy: {metrics['close_direction_accuracy']:.2f}%")
    print(f"Open directional accuracy : {metrics['open_direction_accuracy']:.2f}%")
    print(f"OHLC validity rate        : {metrics['ohlc_validity_rate']:.2f}%")


def main() -> None:
    # Thứ tự chạy ở mức cao:
    # cấu hình runtime -> parse CLI -> cố định seed -> train/evaluate -> lưu -> tóm tắt.
    configure_runtime()
    args = parse_args()
    set_seed(args.seed)
    results = run_training(args)
    artifact_dir = save_artifacts(args, results)
    print_summary(results, artifact_dir)


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass, field

from functional_api.types import ArtifactPolicy, BenchmarkRequestBase, TrainRequestBase


@dataclass
class GRUTrainRequest(TrainRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    train_seed: int = 7
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/gru_train")
    )


@dataclass
class GRUBenchmarkRequest(BenchmarkRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/gru_benchmark")
    )


@dataclass
class LSTMTrainRequest(TrainRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    train_seed: int = 7
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/lstm_train")
    )


@dataclass
class LSTMBenchmarkRequest(BenchmarkRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/lstm_benchmark")
    )


@dataclass
class ANFISTrainRequest(TrainRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    train_seed: int = 7
    fast: bool = False
    n_rules: int | None = None
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/anfis_train")
    )


@dataclass
class ANFISBenchmarkRequest(BenchmarkRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    fast: bool = False
    n_rules: int | None = None
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/anfis_benchmark")
    )


@dataclass
class FeatureGroupTrainRequest(TrainRequestBase):
    split_ratio: str = "7/3"
    gap: int = 1
    look_back: int = 60
    n_mfs: int = 2
    epochs: int = 150
    batch_size: int = 32
    seed: int = 7
    lstm_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    include_exog: bool = False
    max_rows: int | None = None
    verbose: int = 0
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/feature_group_train")
    )


@dataclass
class FeatureGroupBenchmarkRequest(BenchmarkRequestBase):
    split_ratio: str = "7/3"
    gap: int = 1
    look_back: int = 60
    n_mfs: int = 2
    epochs: int = 150
    batch_size: int = 32
    lstm_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    include_exog: bool = False
    max_rows: int | None = None
    verbose: int = 0
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/feature_group_benchmark")
    )


@dataclass
class RegimeHybridTrainRequest(TrainRequestBase):
    output_dir: str = "./functional_api_outputs/regime_hybrid_train"
    look_back: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    n_mfs: int = 2
    epochs: int = 120
    batch_size: int = 32
    runs: int = 1
    seed: int = 42
    temporal_units: int = 48
    conv_filters: int = 48
    dropout: float = 0.15
    learning_rate: float = 1e-3
    component_loss_weight: float = 0.25
    include_exog: bool = False
    max_rows: int | None = None
    verbose: int = 0
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/regime_hybrid_train")
    )


@dataclass
class RegimeHybridBenchmarkRequest(BenchmarkRequestBase):
    output_dir: str = "./functional_api_outputs/regime_hybrid_benchmark"
    look_back: int = 60
    horizon: int = 1
    n_splits: int = 3
    gap: int = -1
    val_frac: float = 0.10
    test_frac: float = 0.10
    min_train_frac: float = 0.40
    max_train_size: int = 768
    n_mfs: int = 2
    epochs: int = 120
    batch_size: int = 32
    temporal_units: int = 48
    conv_filters: int = 48
    dropout: float = 0.15
    learning_rate: float = 1e-3
    component_loss_weight: float = 0.25
    include_exog: bool = False
    max_rows: int | None = None
    verbose: int = 0
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/regime_hybrid_benchmark")
    )

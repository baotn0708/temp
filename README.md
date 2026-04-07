## Everything is going wrong. It's all falling apart.

## Model design

The proposed model is **not**:
- deep model predicts,
- ANFIS predicts residual,
- final output = corrected residual chain.

Instead it is a jointly trained **regime-aware fuzzy hybrid**:

- **causal sequence branch**: lightweight TCN over the raw input window,
- **summary branch**: multi-scale summary over the same window,
- **fuzzy router**: Takagi-Sugeno-style regime gating on momentum / volatility / range / RSI features,
- **experts**: neural experts specialized by regime,
- **linear shortcut**: stable low-complexity path,
- **direction head**: auxiliary direction supervision.

This is closer to a **mixture-of-experts + fuzzy regime partition** design than to a naive ANFIS-after-DL stack.

## Files

- `hybrid_core.py`: feature engineering, purged splits, models, tuning helpers
- `benchmark.py`: leak-safe walk-forward benchmark
- `train.py`: train the selected model on the latest fold
- `benchmark_outputs/`: sample benchmark outputs generated in this session
- `train_outputs/`: sample train outputs generated in this session (`hybrid_state_dict.pt`, `hybrid_scaler.npz`, summary JSON)

## Example benchmark command

```bash
python benchmark.py \
  --files /path/to/AMZN.csv /path/to/JPM.csv /path/to/TSLA.csv \
  --names AMZN JPM TSLA \
  --output-dir ./benchmark_outputs
```

Quick smoke run:

```bash
python benchmark.py \
  --files /path/to/AMZN.csv /path/to/JPM.csv /path/to/TSLA.csv \
  --names AMZN JPM TSLA \
  --eval-seeds 7 \
  --fast \
  --output-dir ./benchmark_outputs
```

## Example training command

```bash
python train.py \
  --model hybrid \
  --files /path/to/AMZN.csv /path/to/JPM.csv /path/to/TSLA.csv \
  --names AMZN JPM TSLA \
  --output-dir ./train_outputs
```

## Clean GRU-only workflow

If you want a dedicated GRU-only pipeline with simple chronological splits, use the new scripts below.

The GRU-only setup now defaults to:
- `lookback = 60` trading days
- `horizon = 1` trading day
- target = next-day log-return computed separately for `Open`, `High`, `Low`, and `Close`
- outputs = 4 regression heads for `open/high/low/close`
- evaluation = reconstruct next-day OHLC prices algebraically from predicted log-returns, then compare directly with test OHLC prices
- metrics = `rmse`, `mape`, `mae`, `r^2`, `sign_acc` for each target price

Benchmark all three datasets in the project with a `7/2/1` split:

```bash
python3 gru_benchmark.py \
  --split-ratio 7/2/1 \
  --output-dir ./gru_benchmark_outputs
```

Benchmark all three datasets with a `7/3` split:

```bash
python3 gru_benchmark.py \
  --split-ratio 7/3 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.10 \
  --lr 1e-3 \
  --output-dir ./gru_benchmark_outputs_73
```

Train and save final GRU-only models for all three datasets:

```bash
python3 gru_train.py \
  --split-ratio 7/2/1 \
  --output-dir ./gru_train_outputs
```

Notes:
- `gru_benchmark.py` and `gru_train.py` default to `AMZN.csv`, `JPM.csv`, and `TSLA.csv` in the current directory.
- `7/2/1` uses validation for config selection and then refits on `train + val` before one final test evaluation.
- `gru_train.py` no longer calls the benchmark path internally, so training does not touch the test set before the final evaluation.
- `7/3` has no validation block, so the scripts now require an explicit fixed GRU config to keep the protocol leak-free.
- Benchmark output is now reported per target price (`open`, `high`, `low`, `close`) instead of aggregate `*_mean/std` columns.

## Clean ANFIS-only workflow

The ANFIS-only setup uses the same target protocol as the GRU-only setup:
- `lookback = 60` trading days
- `horizon = 1` trading day
- target = next-day log-return computed separately for `Open`, `High`, `Low`, and `Close`
- outputs = 4 prediction heads for `open/high/low/close`
- evaluation = reconstruct next-day OHLC prices algebraically from predicted log-returns, then compare directly with test OHLC prices
- metrics = `rmse`, `mape`, `mae`, `r^2`, `sign_acc` for each target price

Benchmark all three datasets in the project with a `7/2/1` split:

```bash
python3 anfis_benchmark.py \
  --split-ratio 7/2/1 \
  --output-dir ./anfis_benchmark_outputs
```

Benchmark all three datasets with a `7/3` split:

```bash
python3 anfis_benchmark.py \
  --split-ratio 7/3 \
  --n-rules 6 \
  --lr 1e-3 \
  --output-dir ./anfis_benchmark_outputs_73
```

Train and save final ANFIS-only models for all three datasets:

```bash
python3 anfis_train.py \
  --split-ratio 7/2/1 \
  --output-dir ./anfis_train_outputs
```

Notes:
- `anfis_benchmark.py` and `anfis_train.py` default to `AMZN.csv`, `JPM.csv`, and `TSLA.csv` in the current directory.
- `7/2/1` tunes the number of fuzzy rules on validation, then refits on `train + val` before one final test evaluation.
- `anfis_train.py` no longer calls the benchmark path internally, so training does not touch the test set before the final evaluation.
- `7/3` has no validation block, so the scripts now require an explicit fixed ANFIS config to keep the protocol leak-free.

## Clean LSTM-only workflow

The LSTM-only setup uses the same target protocol as the GRU-only setup:
- `lookback = 60` trading days
- `horizon = 1` trading day
- target = next-day log-return computed separately for `Open`, `High`, `Low`, and `Close`
- outputs = 4 prediction heads for `open/high/low/close`
- evaluation = reconstruct next-day OHLC prices algebraically from predicted log-returns, then compare directly with test OHLC prices
- metrics = `rmse`, `mape`, `mae`, `r^2`, `sign_acc` for each target price

Benchmark all three datasets in the project with a `7/2/1` split:

```bash
python3 lstm_benchmark.py \
  --split-ratio 7/2/1 \
  --output-dir ./lstm_benchmark_outputs
```

Benchmark all three datasets with a `7/3` split:

```bash
python3 lstm_benchmark.py \
  --split-ratio 7/3 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.10 \
  --lr 1e-3 \
  --output-dir ./lstm_benchmark_outputs_73
```

Train and save final LSTM-only models for all three datasets:

```bash
python3 lstm_train.py \
  --split-ratio 7/2/1 \
  --output-dir ./lstm_train_outputs
```

Notes:
- `lstm_benchmark.py` and `lstm_train.py` default to `AMZN.csv`, `JPM.csv`, and `TSLA.csv` in the current directory.
- `7/2/1` tunes the LSTM config on validation, then refits on `train + val` before one final test evaluation.
- `lstm_train.py` no longer calls the benchmark path internally, so training does not touch the test set before the final evaluation.
- `7/3` has no validation block, so the scripts now require an explicit fixed LSTM config to keep the protocol leak-free.

## Feature-group ANFIS + Sequence workflow

The model defined in `run_feature_group_anfis_clean.py` is now available through clean benchmark/train wrappers.

Protocol:
- `lookback = 60` trading days by default
- forecast horizon = next trading day
- split = `7/3` only
- scaler fit only on train windows
- a purge `gap` is inserted between train and test to reduce overlap leakage
- validation is not used
- the original target parameterization in `run_feature_group_anfis_clean.py` is kept unchanged
- benchmark/train report reconstructed next-day OHLC price metrics

Benchmark all three datasets:

```bash
python3 feature_group_anfis_benchmark.py \
  --split-ratio 7/3 \
  --output-dir ./feature_group_anfis_benchmark_outputs
```

Train and save final models:

```bash
python3 feature_group_anfis_train.py \
  --split-ratio 7/3 \
  --output-dir ./feature_group_anfis_train_outputs
```

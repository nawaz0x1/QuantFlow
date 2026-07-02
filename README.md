# 🌊 QuantFlow

**A Federated Post-Transformer Foundation Model for Probabilistic Time-Series Forecasting**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/) [![Status](https://img.shields.io/badge/status-research-orange)]()

QuantFlow is a probabilistic time-series forecasting framework built around **bidirectional Mamba state-space decoders**, an **inverted embedding strategy**, **quantile regression**, and **federated training**. It replaces quadratic Transformer attention with linear-complexity selective state-space modeling, while still capturing rich cross-variable interactions and producing calibrated uncertainty estimates rather than single-point forecasts.

![QuantFlow Architecture](figures/QuantFlow_Architecture.png)
_QuantFlow: overall architecture (left) and bidirectional Mamba decoder layer (right)_

## Table of Contents

- [Key Contributions](#key-contributions)
- [How It Works](#how-it-works)
- [Main Results](#main-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Ablation Study](#ablation-study)
- [Limitations](#limitations)
- [Citation](#citation)
- [Authors](#authors)

## Key Contributions

- **Bidirectional Mamba Decoders** Replaces $O(N^2)$ Transformer attention with 6 stacked forward/reverse Mamba layers, achieving linear complexity and stronger long-term dependency extraction.
- **Inverted Sequence Embedding** Encodes entire observation windows (100 steps) as tokens, forcing the model to explicitly learn cross-variable interactions and time-varying dynamics.
- **Probabilistic Quantile Forecasting** Moves beyond point estimates by simultaneously predicting 5 conditional quantiles ($0.10, 0.25, 0.50, 0.75, 0.90$) to model predictive uncertainty and risk.
- **TSMixup Augmentation** Expands temporal manifold coverage via Dirichlet-weighted interpolation, creating novel combinations of trends and periodic signals.

## How It Works

1. **Preprocess** Raw records are cleaned, chronologically ordered, enriched with cyclic calendar features (hour, day-of-week, day-of-year, etc.), and scaled per instance using Min-Max normalization.
2. **Augment** TSMixup blends multiple source windows via Dirichlet-sampled convex combinations to diversify the training distribution while preserving temporal structure.
3. **Embed** Each variable's full 100-step lookback window is projected into a 256-dimensional token (inverted embedding), so tokens represent _variables_, not timesteps.
4. **Decode** A stack of 6 bidirectional Mamba layers processes these variable tokens, combining forward and reverse passes at every layer, followed by a convolutional feed-forward block.
5. **Predict** A linear projection head outputs 5 quantile forecasts per variable, optimized jointly using pinball (quantile) loss, then de-normalized via reverse instance normalization.
6. **Federate** _(optional)_ The trained architecture can be distributed across clients with non-IID data partitions and aggregated with sample-weighted Federated Averaging.

## Main Results

### State-of-the-Art Comparison (Centralized)

_MSE on standard benchmarks (lower is better)._

| Model                | ETTm1 (MSE) | Weather (MSE) |
| -------------------- | ----------- | ------------- |
| FEDformer            | 0.448       | 0.308         |
| iTransformer         | 0.407       | 0.257         |
| PatchTST             | 0.400       | 0.258         |
| TimesNet             | 0.387       | 0.259         |
| TimeMixer            | 0.381       | 0.240         |
| Time-MoE Large       | 0.322       | 0.234         |
| **QuantFlow (Ours)** | **0.2834**  | **0.2218**    |

More extensive centralized results across 11 datasets (Bitcoin, Ethereum, Solana, SF Traffic, Electricity, ETTh1/ETTh2, ETTm1/ETTm2, Influenza, Weather) are reported in the accompanying paper, with R² > 0.9 on all datasets except Influenza.

## Installation

A CUDA-compatible GPU (e.g., AWS `g5.4xlarge` with an NVIDIA A10G) is recommended for compiling the Mamba kernels.

```bash
# Clone the repository
git clone https://github.com/nawaz0x1/QuantFlow.git
cd QuantFlow

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Place your raw time-series data (e.g., `data.csv`) in the project root. The preprocessor generates cyclic calendar features, applies instance-wise Min-Max scaling, and writes processed artifacts to `processed_data/`.

```bash
python src/preprocess.py
```

This produces:

- `processed_data/raw.csv` cleaned data with calendar features
- `processed_data/scaled.csv` Min-Max scaled data
- `processed_data/scaler.pkl` fitted scaler for inverse-transforming predictions

### 2. Model Training

Train the bidirectional Mamba architecture end-to-end. The script handles pinball loss optimization for quantile regression, early stopping, checkpointing, and automatically exports the trained model to ONNX format.

```bash
python train.py
```

This produces:

- `quantflow_checkpoints.pth` best model weights (by validation loss)
- `quantflow.onnx` ONNX export of the trained model for deployment/inference

### 3. Inference

```python
import torch
from src.config import ModelConfig
from src.architecture import QuantFlow

config = ModelConfig()
model = QuantFlow(config)
model.load_state_dict(torch.load("quantflow_checkpoints.pth"))
model.eval()

# x: [Batch, seq_len, n_vars]
with torch.no_grad():
    quantile_forecasts = model(x)  # [Batch, pred_len, n_vars, n_quantiles]
```

## Configuration

All model and training hyperparameters live in `src/config.py` as a `ModelConfig` dataclass:

| Parameter   | Default                          | Description                                             |
| ----------- | -------------------------------- | ------------------------------------------------------- |
| `seq_len`   | `10`                             | Input lookback window length                            |
| `pred_len`  | `1`                              | Forecast horizon                                        |
| `n_vars`    | `1`                              | Number of input variables (set automatically from data) |
| `quantiles` | `[0.10, 0.25, 0.50, 0.75, 0.90]` | Conditional quantiles predicted                         |
| `d_model`   | `256`                            | Embedding / hidden dimension                            |
| `d_state`   | `64`                             | Mamba SSM state dimension                               |
| `e_layers`  | `6`                              | Number of bidirectional Mamba decoder layers            |
| `d_ff`      | `1024`                           | Feed-forward inner dimension                            |
| `expand`    | `2`                              | Mamba block expansion factor                            |
| `dropout`   | `0.1`                            | Dropout rate                                            |
| `use_norm`  | `True`                           | Instance normalization / de-normalization               |
| `device`    | auto                             | `cuda` if available, else `cpu`                         |

Edit `src/config.py` directly, or instantiate and override programmatically:

```python
from src.config import ModelConfig

config = ModelConfig(seq_len=100, pred_len=1, d_model=256, e_layers=6)
```

## Data Format

`train.py` expects a pickled dictionary (`data.pkl`) with the following keys:

| Key        | Shape                  | Description                   |
| ---------- | ---------------------- | ----------------------------- |
| `X_trains` | `[N, seq_len, n_vars]` | Training input windows        |
| `y_trains` | `[N, n_vars]`          | Training targets              |
| `X_tests`  | `[M, seq_len, n_vars]` | Validation/test input windows |
| `y_tests`  | `[M, n_vars]`          | Validation/test targets       |
| `n_vars`   | `int`                  | Number of variables           |

## Ablation Study

Removing either core architectural component degrades performance across all datasets:

| Variant                     | Effect                                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Full QuantFlow**          | Best MSE / MAPE / R² across all 11 datasets                                                           |
| **w/o Bidirectional Mamba** | Notable degradation on datasets with strong temporal dynamics (SF Traffic, Weather, ETTh2, Influenza) |
| **w/o Inverted Embedding**  | Largest degradation overall. This component contributes the most to model performance                |

## Limitations

- **Irregular / epidemiological signals** Influenza forecasting remains the hardest benchmark (R² ≈ 0.53), reflecting outbreak irregularity, reporting noise, and unmodeled external factors (interventions, strain changes).
- **Long-horizon generalization** current experiments focus on one-step-ahead forecasting; extending to longer horizons is future work.
- **Federated calibration** quantile coverage and calibration error under federated non-IID training have not yet been rigorously evaluated.

## Citation

If you find this code useful in your research, please cite:

```bibtex
will be added
```

## Authors

- **Shah Nawaz Haider** & **Steve Austin** _Department of Computer Science and Engineering, University of Science and Technology Chittagong (USTC)_
- **Arnab Barua** & **Sarowar Morshed Shawon** _Department of Electrical and Electronic Engineering, USTC_

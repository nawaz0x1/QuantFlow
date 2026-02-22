import warnings

warnings.filterwarnings("ignore")
import torch
from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    seq_len: int = 10
    pred_len: int = 1
    n_vars: int = 1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])

    # Model Parameters
    d_model: int = 256
    d_state: int = 64
    e_layers: int = 6
    d_ff: int = 1024  # 4 * d_model
    expand: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

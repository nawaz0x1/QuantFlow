# QuantFlow: Foundational Federated Time-Series Model

QuantFlow is a scalable, privacy-preserving forecasting framework built on a **post-Transformer Mamba architecture**. It addresses the high computational complexity and memory demands of traditional Transformer models while enabling decentralized training through **Federated Learning (FL)**.

---

## Key Features

* **Post-Transformer Architecture:** Utilizes the **Mamba state-space modeling** paradigm for linear sequence scalability and superior memory efficiency.

* **Probabilistic Forecasting:** Moves beyond point estimates by producing multiple **conditional quantiles** to estimate predictive uncertainty.

* **Privacy-Preserving:** Supports **federated pre-training** on decentralized, sensitive data across multiple clients without direct data sharing.

* **Advanced Data Augmentation:** Employs **TSMixup** to interpolate existing samples, expanding temporal manifold coverage and improving zero-shot generalization.

* **Multivariate & Covariate Support:** Designed to jointly model interdependent time series and incorporate external factors like calendar effects or promotions.



---

## Architecture

The model incorporates several specialized layers to optimize time-series performance:

1. **Inverted Sequence Embedding:** Linear projection across the entire historical window (default 100 steps) to capture global temporal dynamics.


2. **Bidirectional Mamba Decoders:** Stacked layers (default 6) using forward and backward state-space blocks to capture context in both temporal directions.


3. **Instance-wise Normalization:** Centers and scales each batch to improve numerical stability and accelerate convergence.


4. **Quantile Projection Head:** Outputs probability levels (0.1, 0.25, 0.5, 0.75, 0.9) to define conditional distribution boundaries.



---

## ## Project Info

**Authors:** Shah Nawaz Haider & Steve Austin.
**University:** University of Science and Technology, Chittagong (USTC).
**Supervisors:** Dr. Hadaate Ullah & Sarowar Morshed Shawon.
**Infrastructure:** Experiments conducted on **AWS g5.4xlarge** instances with NVIDIA A10G GPUs.


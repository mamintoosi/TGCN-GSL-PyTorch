"""
Explicit configuration for DAGMA and experiments.
Every parameter is documented — no undocumented defaults.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DAGMAConfig:
    """
    All DAGMA parameters, explicitly specified.
    Source: paper Section 3 (Table 1) + dagma library defaults.
    """
    lambda1: float = 0.01          # L1 regularization strength
    loss_type: str = 'l2'          # Score function: 'l2' loss
    w_threshold: float = 0.3       # Weight threshold for zeroing small weights
    max_iter: int = 180000         # Maximum optimization iterations (dagma default)
    warm_iter: int = 50000         # Warm-up iterations (dagma default)
    mu_init: float = 1.0           # Initial augmented Lagrangian penalty
    mu_factor: float = 10          # Penalty increase factor
    lr: float = 0.002              # Learning rate for dagma (dagma default)
    s: int = 1                     # Exponent in DAG constraint h(W) = tr(e^{s*(W∘W)}) - d
    check_dag: bool = True         # Whether to check acyclicity at convergence
    verbose: bool = True           # Print optimization progress

    def __post_init__(self):
        assert self.lambda1 > 0, "lambda1 must be positive"
        assert self.loss_type in ('l2', 'l1'), f"loss_type must be 'l2' or 'l1', got {self.loss_type}"
        assert self.w_threshold >= 0, "w_threshold must be non-negative"
        assert self.max_iter > 0, "max_iter must be positive"


@dataclass
class ExperimentConfig:
    """
    Configuration for a single controlled experiment.
    """
    # Dataset
    dataset_name: str = "shenzhen"  # "shenzhen" or "losloop"
    
    # Data parameters
    seq_len: int = 12               # Historical window (number of past steps)
    pre_len: int = 1                # Prediction horizon (1-4)
    split_ratio: float = 0.8        # Train/test split
    normalize: bool = True          # Min-max normalization
    
    # Graph type
    # 0: physical (predefined adjacency)
    # 1: GSL (directed acyclic, from DAGMA)
    # 2: cGSL (symmetrized cyclic)
    # 3: random sparse (physical edges randomly removed)
    # 4: correlation-based
    graph_type: int = 0
    
    # Model
    model_name: str = "GCN"         # "GCN" or "TGCN"
    hidden_dim: int = 100           # Hidden dimension
    loss: str = "mse"              # "mse" for GCN, "mse_with_regularizer" for TGCN
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 50
    batch_size: int = 64
    
    # Seeds
    seed: int = 42
    
    # DAGMA config (used when graph_type in {1, 2})
    dagma: DAGMAConfig = field(default_factory=DAGMAConfig)

    def __post_init__(self):
        assert self.dataset_name in ("shenzhen", "losloop"), f"Unknown dataset: {self.dataset_name}"
        assert 1 <= self.pre_len <= 4, f"pre_len must be 1-4, got {self.pre_len}"
        assert self.model_name in ("GCN", "TGCN"), f"model must be GCN or TGCN, got {self.model_name}"
        assert self.graph_type in (0, 1, 2, 3, 4), f"graph_type must be 0-4, got {self.graph_type}"


# Dataset-specific DAGMA configurations (from paper)
DATASET_DAGMA_CONFIGS = {
    "shenzhen": {"lambda1": 0.01},
    "losloop": {"lambda1": 0.02},
}

# Data paths
DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}

# W_est file paths (existing pre-computed DAGMA results)
W_EST_PATHS = {
    "shenzhen": "data/W_est_shenzhen_pre_len{pre_len}.npy",
    "losloop": "data/W_est_losloop_pre_len{pre_len}.npy",
}

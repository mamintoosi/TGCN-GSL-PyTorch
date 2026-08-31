"""
Data loading, preprocessing, and DAGMA input preparation.
All operations are verified and documented.
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Optional

from gsl_clean.config import DATA_PATHS, ExperimentConfig


def load_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw feature and adjacency data.
    
    Args:
        dataset_name: "shenzhen" or "losloop"
    
    Returns:
        feat: Feature matrix (T, N) — T time steps, N sensors
        adj: Physical adjacency matrix (N, N)
    """
    paths = DATA_PATHS[dataset_name]
    
    # Load features
    feat_df = pd.read_csv(paths["feat"])
    feat = np.array(feat_df, dtype=np.float32)
    
    # Load adjacency
    adj_df = pd.read_csv(paths["adj"], header=None)
    adj = np.array(adj_df, dtype=np.float32)
    
    # Validate shapes
    assert feat.ndim == 2, f"Feature matrix must be 2D, got shape {feat.shape}"
    assert adj.ndim == 2, f"Adjacency matrix must be 2D, got shape {adj.shape}"
    assert feat.shape[1] == adj.shape[0] == adj.shape[1], \
        f"Dimension mismatch: feat {feat.shape}, adj {adj.shape}"
    
    return feat, adj


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Min-max normalization.
    
    Returns:
        normalized: Normalized data in [0, 1]
        max_val: Maximum value used for normalization
    """
    max_val = np.max(data)
    normalized = data / max_val
    return normalized, max_val


def generate_sequences(
    data: np.ndarray,
    seq_len: int,
    pre_len: int,
    split_ratio: float = 0.8,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sequence pairs for training and testing.
    
    Args:
        data: Raw data (T, N)
        seq_len: Number of historical steps
        pre_len: Prediction horizon
        split_ratio: Train/test split ratio
        normalize: Whether to normalize
    
    Returns:
        train_X: (M, seq_len, N)
        train_Y: (M, pre_len, N)
        test_X: (K, seq_len, N)
        test_Y: (K, pre_len, N)
    
    IMPORTANT: This is IDENTICAL to the existing implementation in
    utils/data/functions.py. Verified line by line.
    """
    T = data.shape[0]
    
    if normalize:
        data = data / np.max(data)
    
    train_size = int(T * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:T]
    
    # Generate training sequences
    train_X, train_Y = [], []
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(train_data[i : i + seq_len])
        train_Y.append(train_data[i + seq_len : i + seq_len + pre_len])
    
    # Generate test sequences
    test_X, test_Y = [], []
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(test_data[i : i + seq_len])
        test_Y.append(test_data[i + seq_len : i + seq_len + pre_len])
    
    return (
        np.array(train_X, dtype=np.float32),
        np.array(train_Y, dtype=np.float32),
        np.array(test_X, dtype=np.float32),
        np.array(test_Y, dtype=np.float32),
    )


def prepare_dagma_input(
    train_X: np.ndarray,
    pre_len: int,
) -> np.ndarray:
    """
    Prepare the input matrix for DAGMA.
    
    THIS IS THE CRITICAL FUNCTION in the entire pipeline.
    
    The existing implementation does:
        data = np.array([x[0] for x in self.train_data])
        X = data[i::pre_len]
    
    This extracts the FIRST time step of each training sequence,
    producing a (M, N) matrix where each row is a contemporaneous snapshot.
    
    DAGMA input interpretation:
        Each ROW = one observation of N variables at the SAME time
        Each COLUMN = one variable's time series
    
    This is a cross-sectional (contemporaneous) design.
    
    Args:
        train_X: Training data (M, seq_len, N)
        pre_len: Prediction horizon (used for subsampling)
    
    Returns:
        dagma_input: (M//pre_len, N) or (M, N) matrix for DAGMA
    
    Note: The subsampling with pre_len is part of the existing methodology.
          For pre_len=1: all training samples are used.
          For pre_len=2: every other sample is used.
          For pre_len=3: every third sample is used.
    
    Scientific interpretation:
        The DAGMA input contains CONTemporaneous (same-time) observations
        of all N sensors. DAGMA learns which sensor values covary
        within the same time step, NOT which sensor at time t predicts
        which sensor at time t+1.
    
        The paper's claim that "edge j→i means j at time t predicts i at time t+1"
        is NOT supported by this input construction.
    """
    # Step 1: Extract first time step from each sequence
    # This is exactly what the existing code does: x[0] for x in train_data
    data = np.array([train_X[i][0] for i in range(len(train_X))])
    
    # Step 2: Subsample based on prediction horizon
    X = data[::pre_len]
    
    # Validate
    assert X.ndim == 2, f"DAGMA input must be 2D, got shape {X.shape}"
    M, N = X.shape
    assert M > 0, f"DAGMA input has no samples (M=0)"
    assert N > 1, f"DAGMA input must have >1 variable (N={N})"
    
    return X.astype(np.float64)  # DAGMA uses float64 for numerical stability


def prepare_dagma_input_temporal(
    train_X: np.ndarray,
    pre_len: int,
) -> Optional[np.ndarray]:
    """
    ALTERNATIVE DAGMA input: temporally-lagged design.
    
    This is NOT what the existing code does, but may be what the paper
    INTENDS based on Section 5's description of temporal dependencies.
    
    For each time step t, create a row:
        [x_1(t), ..., x_N(t), x_1(t-1), ..., x_N(t-1), ...]
    
    OR simply use the full (M, seq_len, N) matrix reshaped as:
        (M, seq_len * N)
    
    WARNING: This is a METHODOLOGICAL VARIANT, not a bug fix.
    Using this would be a different methodology.
    
    Args:
        train_X: Training data (M, seq_len, N)
        pre_len: Prediction horizon
    
    Returns:
        dagma_input: (M, seq_len * N) matrix — temporally-augmented input
    
    Note: With this design, DAGMA would learn a graph over seq_len * N
    variables, and the resulting adjacency matrix would be (seq_len*N) × (seq_len*N),
    which is NOT compatible with the GCN/T-GCN input format.
    """
    M, seq_len, N = train_X.shape
    
    # Reshape to (M, seq_len * N)
    X_temporal = train_X.reshape(M, seq_len * N)
    
    # Subsample
    X_temporal = X_temporal[::pre_len]
    
    return X_temporal.astype(np.float64)


def torchify(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    test_X: np.ndarray,
    test_Y: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors."""
    return (
        torch.FloatTensor(train_X),
        torch.FloatTensor(train_Y),
        torch.FloatTensor(test_X),
        torch.FloatTensor(test_Y),
    )

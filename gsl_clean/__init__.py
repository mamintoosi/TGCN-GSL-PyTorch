"""
Clean-room reimplementation of GSL (Graph Structure Learning) for Traffic Prediction.

This module provides a verified, well-documented implementation of:
1. Graph construction from DAGMA output
2. GCN normalization verification
3. DAGMA input preparation
4. Controlled experiment framework
"""
from gsl_clean.config import DAGMAConfig, ExperimentConfig
from gsl_clean.graph_utils import (
    build_gsl_adjacency,
    build_cgsl_adjacency,
    calculate_laplacian_with_self_loop,
    graph_statistics,
)
from gsl_clean.data_pipeline import load_data, prepare_dagma_input, generate_sequences

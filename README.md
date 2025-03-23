# Graph Structure Learning for Traffic Prediction

This repository introduces a novel approach to traffic prediction by incorporating Graph Structure Learning (GSL) techniques. We enhance two fundamental architectures - the Graph Convolutional Network (GCN) proposed by [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907) and its temporal extension T-GCN introduced in [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320). The key innovation lies in automatically learning the optimal graph adjacency matrix from the data, rather than relying on predefined graph structures, which significantly improves the traffic prediction performance of both models.

## Overview
Building upon the foundational Graph Convolutional Network (GCN) introduced by Kipf & Welling (2017), we present GCN-GSL and its temporal extension TGCN-GSL. These models enhance the original GCN architecture by incorporating Graph Structure Learning (GSL) techniques. This innovative approach automatically learns and optimizes graph structures from traffic data, eliminating the need for predefined adjacency matrices. The models dynamically adapt to capture complex spatiotemporal dependencies and evolving traffic patterns in the network.

Key features and innovations:
- Advanced implementation of GCN/T-GCN with Graph Structure Learning
- Flexible prediction capabilities supporting multiple time horizons (1-4 steps ahead)
- Comprehensive evaluation on diverse real-world traffic datasets
- Extensive comparative analysis with baseline models

## Requirements

* numpy
* pandas
* torch
* torchmetrics>=0.11
* dagma (for Graph Structure Learning)

For a complete list of dependencies with specific versions, see `requirements.txt`.

## Environment Setup

To ensure compatibility across different systems, we recommend creating a dedicated conda environment:

```bash
# Create a new conda environment
conda create -n tgcn-gsl python=3.10

# Activate the environment
conda activate tgcn-gsl

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

For GPU support, make sure you have compatible CUDA drivers installed.

## Model Architecture

TGCN-GSL combines the temporal modeling capability of Gated Recurrent Units (GRUs) with the spatial modeling capability of Graph Convolutional Networks (GCNs). The GSL component uses the DAGMA (Directed Acyclic Graph Matrix Estimation) algorithm to learn the optimal graph structure from the data.

## Datasets

The repository includes support for two traffic datasets:
- **Shenzhen (sz)**: Traffic speed data from Shenzhen, China
- **Los Angeles (los)**: Traffic speed data from the Los Angeles highway system

## Model Training

The repository supports training different models (GCN, TGCN) with various configurations, prediction horizons, and with or without Graph Structure Learning.

### Running with Configuration Files

Training is controlled through YAML configuration files. To run a model with a specific configuration:

```bash
python main.py --config configs/<configuration-file>.yaml
```

Examples:
```bash
# Run GCN on Shenzhen dataset with prediction length 1, no GSL
python main.py --config configs/gcn-sz-pre_len1.yaml

# Run GCN on Shenzhen dataset with prediction length 1, with GSL
python main.py --config configs/gcn-sz-gsl-pre_len1.yaml

# Run TGCN on Los Angeles dataset with prediction length 3, with GSL
python main.py --config configs/tgcn-los-gsl-pre_len3.yaml
```

### Configuration File Format

Below is a sample configuration file (`gcn-sz-gsl-pre_len1.yaml`) that trains a GCN model on the Shenzhen dataset with GSL enabled and prediction horizon of 1:

```yaml
fit:
  trainer:
    max_epochs: 50
    accelerator: cuda
    devices: 1
  data:
    dataset_name: shenzhen
    batch_size: 64
    seq_len: 12
    pre_len: 1
  model:
    model:
      class_path: models.GCN
      init_args:
        hidden_dim: 100
        use_gsl: 1
    learning_rate: 0.001
    weight_decay: 0
    loss: mse
```

### Configuration Naming Convention

The configuration files follow this naming convention:
* `<model>-<dataset>-[gsl]-[adj]-pre_len<prediction_length>.yaml`

Where:
- `<model>`: Model type (gcn, tgcn)
- `<dataset>`: Dataset name (sz for Shenzhen, los for Los Angeles loop)
- `[gsl]`: Optional, include if using Graph Structure Learning
- `[adj]`: Optional, include if using the adjacency matrix provided with the dataset
- `<prediction_length>`: Prediction horizon length (1, 2, 3, or 4)

### Reproducing Paper Results with Jupyter Notebooks

For convenience, we provide two Jupyter notebooks that can be used to reproduce the results presented in our paper:
1. **main-GCN-GSL-Colab.ipynb**: Runs all GCN experiments (with and without GSL) on both datasets with different prediction horizons.
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi/TGCN-GSL-PyTorch/blob/main/main-GCN-GSL-Colab.ipynb)

2. **main-TGCN-GSL-Colab.ipynb**: Runs all TGCN experiments (with and without GSL) on both datasets with different prediction horizons.
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi/TGCN-GSL-PyTorch/blob/main/main-TGCN-GSL-Colab.ipynb)

These notebooks automatically execute the training process for all configurations and collect the results. They can be run in Google Colab or any Jupyter environment with the required dependencies installed.

Example of what these notebooks do:
```python
# Loop through datasets and prediction lengths to run all experiments
datasets = ['sz', 'los']  # sz=shenzhen, los=losloop
pred_list = [1, 2, 3, 4]

for dataset in datasets:
    for pre_len in pred_list:
        # Run baseline model
        %run main.py --config configs/gcn-{dataset}-pre_len{pre_len}.yaml
        # Run model with GSL
        %run main.py --config configs/gcn-{dataset}-gsl-pre_len{pre_len}.yaml
```

## Graph Structure Learning

The model incorporates Graph Structure Learning (GSL) using the DAGMA algorithm to estimate the adjacency matrix. This process follows two approaches:

1. **Loading an Existing Matrix**:  
   If an estimated adjacency matrix file (e.g., `W_est_losloop_pre_len1.npy`) exists in the `data/` folder, the program loads it directly. This ensures efficiency by utilizing precomputed matrices.

2. **Estimating a New Matrix**:  
   If the required adjacency matrix file isn't found, the program estimates it using GSL. Once computed, the adjacency matrix is saved for future use. The program provides a log to indicate this process:

   Example output:
   ```
   100%|██████████| 180000/180000.0 [10:03<00:00, 298.33it/s]  
   [2025-03-05 13:39:45,765 INFO]Using device: cuda
   File data/W_est_losloop_pre_len1.npy did not exist. The adjacency matrix estimated by GSL is computed and saved.
   ```

## Evaluation Metrics

The models are evaluated using several metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Accuracy
- R² Score
- Explained Variance

## Results

Our experimental results demonstrate that incorporating Graph Structure Learning (GSL) for adjacency matrix estimation significantly enhances prediction performance compared to traditional GCN-based methods (GCN & TGCN) across various prediction horizons and datasets. The GSL approach enables the discovery of optimized graph structures that better capture the underlying traffic patterns, resulting in improved prediction accuracy. Detailed quantitative results and comparisons are presented in our paper, showing consistent performance gains across different evaluation metrics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

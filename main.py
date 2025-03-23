import argparse
import yaml
import logging
import random  
import numpy as np  
import torch
import os
import csv  # Add this import
import pandas as pd

import models  # noqa: F401 just to ensure we import GCN, GRU, TGCN
from tasks.supervised import SupervisedForecastTask
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVData
from utils.logging import format_logger, output_logger_to_file
from utils.visualization import plot_result 

# Set random seed for reproducibility  
random.seed(42)          # Set Python random seed  
np.random.seed(42)      # Set NumPy seed  
torch.manual_seed(42)   # Set PyTorch seed  

def train_and_validate(
    model_task: SupervisedForecastTask,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_epochs=50,
    batch_size=32,
    device="cuda",
    logger=logging.getLogger(__name__),
    metrics_file="metrics.csv",  # Add this parameter
):
    """
    Manual training + validation loop (pure PyTorch).
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model_task.model.to(device)
    if model_task.regressor is not None:
        model_task.regressor.to(device)

    optimizer = model_task.configure_optimizer()

    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Open the CSV file to save metrics
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "RMSE", "MAE", "Accuracy", "R2", "Expl.Var"])

        for epoch in range(num_epochs):
            model_task.model.train()
            if model_task.regressor is not None:
                model_task.regressor.train()

            total_train_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model_task.training_step((x, y))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # ---- Validation ----
            model_task.model.eval()
            if model_task.regressor is not None:
                model_task.regressor.eval()

            val_metrics = model_task.validation_epoch(val_loader, device)
            # Log metrics every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_metrics['val_loss']:.6f}, "
                    f"RMSE: {val_metrics['RMSE']:.6f}, "
                    f"MAE: {val_metrics['MAE']:.6f}, "
                    f"Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"R2: {val_metrics['R2']:.4f}, "
                    f"Expl.Var: {val_metrics['ExplainedVar']:.4f}"
                )

            # Write metrics to CSV file
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                val_metrics['val_loss'],
                val_metrics['RMSE'],
                val_metrics['MAE'],
                val_metrics['accuracy'],
                val_metrics['R2'],
                val_metrics['ExplainedVar']
            ])
            
            # # Save validation results and plot at the last epoch
            # if epoch == num_epochs - 1:
            #     # Get predictions for plotting
            #     with torch.no_grad():
            #         x, y = next(iter(val_loader))
            #         x, y = x.to(device), y.to(device)
            #         y_pred = model_task.model(x)
                    
            #         # Scale predictions and true values
            #         y_pred = y_pred * model_task.feat_max_val
            #         y = y * model_task.feat_max_val
                    
            #         # Convert to numpy for plotting
            #         y_pred = y_pred.cpu().numpy()
            #         y_true = y.cpu().numpy()
                    
            #         # Reshape for plotting (batch_size, pre_len, num_nodes) -> (batch_size * pre_len, num_nodes)
            #         y_pred = y_pred.transpose(0, 2, 1).reshape(-1, y_pred.shape[-1])
            #         y_true = y_true.reshape(-1, y_true.shape[1])
                                        
            #         # Plot and save results
            #         plot_result(y_pred, y_true, metrics_file.replace('.csv', '.pdf'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gcn.yaml", help="Path to config YAML.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use (e.g. cuda or cpu)")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    format_logger(logger)
    if args.log_path:
        output_logger_to_file(logger, args.log_path)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {args.config}: {config}")

    num_epochs = config["fit"]["trainer"]["max_epochs"]
    batch_size = config["fit"]["data"]["batch_size"]
    seq_len = config["fit"]["data"]["seq_len"]
    pre_len = config["fit"]["data"]["pre_len"]
    learning_rate = config["fit"]["model"]["learning_rate"]
    weight_decay = config["fit"]["model"]["weight_decay"]
    loss_name = config["fit"]["model"]["loss"]
    use_gsl = config["fit"]["model"]["model"]["init_args"].get("use_gsl", 0)  # Get use_gsl from config

    model_class_path = config["fit"]["model"]["model"]["class_path"]  # e.g. "models.GCN"
    model_init_args = config["fit"]["model"]["model"].get("init_args", {})

    dataset_name = config["fit"]["data"]["dataset_name"]
    data_module = SpatioTemporalCSVData(
        dataset_name=dataset_name,
        seq_len=seq_len,
        pre_len=pre_len,
        split_ratio=0.8,
        normalize=True,
        use_gsl=use_gsl,
    )

    # Force required arguments depending on the model
    model_cls_name = model_class_path.split(".")[-1]  # e.g. "GCN"
    ModelClass = getattr(models, model_cls_name)

    train_dataset, val_dataset = data_module.get_datasets()
    if data_module.use_gsl > 0:
        data_module.compute_adjacency_matrix()
    else:
        print(model_cls_name)

    # Create 'results' and 'models' folders if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{model_cls_name}", exist_ok=True)
    os.makedirs("trained-models", exist_ok=True)

    # Generate a unique metrics file name based on config parameters
    metrics_file = f"results/{model_cls_name}/metrics_{dataset_name}_seq{seq_len}_pre{pre_len}_gsl{use_gsl}.csv"

    if model_cls_name in ("GCN", "TGCN"):
        model_init_args["adj"] = data_module.adj
        model_init_args["seq_len"] = seq_len
    elif model_cls_name == "GRU":
        model_init_args["num_nodes"] = data_module.num_nodes

    model = ModelClass(**model_init_args)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    logger.info(f"Using device: {device}")
    model = model.to(device)

    model_task = SupervisedForecastTask(
        model=model,
        loss=loss_name,
        pre_len=pre_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        feat_max_val=data_module.feat_max_val,
    )

    model_filename = f"trained-models/{dataset_name}_{model_cls_name}_seq{seq_len}_pre{pre_len}_gsl{use_gsl}.pt"

    # Train and validate
    train_and_validate(
        model_task=model_task,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        logger=logger,
        metrics_file=metrics_file
    )

    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    logger.info(f"Model saved to {model_filename}")

    logger.info("Finished training!\n")

if __name__ == "__main__":
    main()
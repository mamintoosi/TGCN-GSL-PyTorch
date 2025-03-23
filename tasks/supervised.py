import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from utils import metrics
from utils.losses import mse_with_regularizer_loss


class SupervisedForecastTask:
    """
    A pure PyTorch "task" class that wraps:
      - A model (nn.Module)
      - A regressor (e.g., final linear layer)
      - A chosen loss function
      - Some logic for forward pass, training step, validation step

    We manually handle the optimizer and loops outside.
    """

    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse_with_regularizer",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        self.model = model
        self.pre_len = pre_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feat_max_val = feat_max_val

        # The model might define "hidden_dim" or "output_dim" in .hyperparameters
        # Use whichever is present
        model_hp = getattr(self.model, "hyperparameters", {})
        hidden_dim = model_hp.get("hidden_dim", model_hp.get("output_dim", 64))

        if regressor == "linear":
            self.regressor = nn.Linear(hidden_dim, self.pre_len)
        elif isinstance(regressor, nn.Module):
            self.regressor = regressor
        else:
            self.regressor = None

        self._loss_name = loss

    def forward(self, x):
        # x: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.size()
        hidden = self.model(x)  # shape (batch_size, num_nodes, hidden_dim)
        # flatten to (batch_size*num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(-1)))
        if self.regressor is not None:
            predictions = self.regressor(hidden)  # (batch_size*num_nodes, pre_len)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def compute_loss(self, inputs, targets):
        if self._loss_name == "mse":
            return F.mse_loss(inputs, targets)
        elif self._loss_name == "mse_with_regularizer":
            # We pass `self` so that it can sum up parameters for the reg penalty
            return mse_with_regularizer_loss(inputs, targets, self)
        else:
            raise ValueError(f"Unsupported loss: {self._loss_name}")

    def training_step(self, batch):
        x, y = batch
        predictions = self.forward(x)
        # reshape predictions: (batch_size, num_nodes, pre_len) -> (batch_size*pre_len, num_nodes)
        # or whichever shape matches y exactly
        # In original code: y shape is (batch_size, pre_len, num_nodes)
        # so let's unify them
        predictions = predictions.transpose(1, 2).reshape((-1, x.size(2)))
        y = y.reshape((-1, y.size(2)))
        loss = self.compute_loss(predictions, y)
        return loss

    def validation_epoch(self, val_loader, device):
        # We'll accumulate for final metrics
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0
        total_acc = 0.0
        total_r2 = 0.0
        total_ev = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                predictions = self.forward(x)
                # shape adjustments
                predictions = predictions * self.feat_max_val
                y = y * self.feat_max_val
                preds_2d = predictions.transpose(1, 2).reshape((-1, x.size(2)))
                y_2d = y.reshape((-1, y.size(2)))

                loss = self.compute_loss(preds_2d, y_2d)
                rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(preds_2d, y_2d))
                mae = torchmetrics.functional.mean_absolute_error(preds_2d, y_2d)
                acc = metrics.accuracy(preds_2d, y_2d)
                r2 = metrics.r2(preds_2d, y_2d)
                ev = metrics.explained_variance(preds_2d, y_2d)

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_rmse += rmse.item() * batch_size
                total_mae += mae.item() * batch_size
                total_acc += acc.item() * batch_size
                total_r2 += r2.item() * batch_size
                total_ev += ev.item() * batch_size
                total_samples += batch_size

        avg_metrics = {
            "val_loss": total_loss / total_samples,
            "RMSE": total_rmse / total_samples,
            "MAE": total_mae / total_samples,
            "accuracy": total_acc / total_samples,
            "R2": total_r2 / total_samples,
            "ExplainedVar": total_ev / total_samples,
        }
        return avg_metrics

    def parameters(self):
        # Return all parameters from model + regressor
        if self.regressor is not None:
            return list(self.model.parameters()) + list(self.regressor.parameters())
        else:
            return self.model.parameters()

    def configure_optimizer(self):
        # Return a simple Adam
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

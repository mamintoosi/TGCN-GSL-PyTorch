import copy
import numpy as np
import torch

class BestEpochCallback:
    def __init__(self, monitor="", mode="min"):
        """
        Example callback to track best epoch. You must manually call its methods in pure PyTorch.
        """
        self.monitor = monitor
        self.mode = mode
        if self.mode == "min":
            self.best_value = torch.tensor(np.Inf)
        else:
            self.best_value = torch.tensor(-np.Inf)
        self.best_epoch = 0

    def check_and_update(self, current_epoch, current_value):
        if self.mode == "min" and current_value < self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch
        elif self.mode == "max" and current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch

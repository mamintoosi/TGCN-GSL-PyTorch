import numpy as np
import os

class SaveLastEpochValResults:
    def __init__(self, save_path=None):
        super().__init__()
        self.save_path = save_path
        self.ground_truths = []
        self.predictions = []

    def on_validation_batch_end(self, predictions, targets):
        """
        You can call this at each validation batch if you want to store them.
        """
        self.predictions.append(predictions.detach().cpu().numpy())
        self.ground_truths.append(targets.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        """
        Finally call this once at the end of validation epoch to save the arrays.
        """
        if not self.save_path:
            self.save_path = "test_results.npz"
        ground_truths = np.concatenate(self.ground_truths, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        np.savez(self.save_path, outputs=predictions, targets=ground_truths)
        print(f"Saved validation results to {self.save_path}")

from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVData
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVDataModule  # optional if you want to keep old name

SupervisedDataModule = SpatioTemporalCSVDataModule

__all__ = [
    "SupervisedDataModule",
    "SpatioTemporalCSVDataModule",
    "SpatioTemporalCSVData",  # new pure PyTorch version
]

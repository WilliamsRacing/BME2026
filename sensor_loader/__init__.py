"""Framework-independent, disk-bounded sensor ZIP loading (Python 3.8+)."""
from .catalog import Catalog, build_catalog, split_healthids
from .dataset import LoaderConfig, ZipWindowDataset, SensorDataLoader, managed_dataloader

__all__ = ["Catalog", "build_catalog", "split_healthids", "LoaderConfig",
           "ZipWindowDataset", "SensorDataLoader", "managed_dataloader"]

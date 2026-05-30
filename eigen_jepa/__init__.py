from .data import MarketConfig, MarketWindowDataset, build_datasets, generate_market_series
from .memory import SpectralMemory, MemoryState
from .model import EigenJEPA, EigenJEPAConfig
from .spectral import (
    rolling_covariance,
    topk_spectrum,
    projector_from_vecs,
    principal_angles,
    spectral_metrics,
    subspace_distance,
    eigengaps,
    effective_rank,
    spectral_entropy,
)

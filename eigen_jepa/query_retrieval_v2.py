"""Opt-in query-gradient model; legacy training and frozen policies are unchanged.

The forward read is the legacy cosine/top-k/softmax map. Only its derivative
through the query is enabled. Stored content is an owned, detached read snapshot;
discrete neighbor membership is not differentiated. A nonzero gradient is not
evidence of forecasting utility. Checkpoints restore model/memory, not an exact
optimizer, random stream or data-loader continuation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import io
import math
from pathlib import Path
import pickle

import torch

from .memory import MemoryState, SpectralMemory
from .model import EigenJEPA, EigenJEPAConfig

MODEL_SCHEMA = "eigen_jepa.query_gradient_model.v2"
RETRIEVAL_POLICY = "selected_topk_cosine_query_gradient.v2"
CHECKPOINT_SCHEMA = "eigen_jepa.query_gradient_checkpoint.v2"
FLOAT_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, *, minimum=0.0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        valid = math.isfinite(value) and value >= minimum and (not strictly_positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} is outside its finite supported range")
    return value


def _tensor(value, name, *, shape=None, dtype=None, device=None):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype not in (torch.float32, torch.float64, torch.int64)
        or (shape is not None and tuple(value.shape) != tuple(shape))
        or (dtype is not None and value.dtype != dtype)
        or (device is not None and value.device != torch.device(device))
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite dense tensor with the declared shape/dtype/device")
    return value


@dataclass
class QueryGradientConfigV2(EigenJEPAConfig):
    model_schema: str = field(default=MODEL_SCHEMA, init=False)
    retrieval_policy: str = field(default=RETRIEVAL_POLICY, init=False)


def query_config_v2(mapping: dict) -> QueryGradientConfigV2:
    """Parse the complete explicit v2 config; do not promote a legacy config."""
    if not isinstance(mapping, dict) or set(mapping) != {item.name for item in fields(QueryGradientConfigV2)}:
        raise ValueError("Expected the complete QueryGradientConfigV2 fields")
    if mapping["model_schema"] != MODEL_SCHEMA or mapping["retrieval_policy"] != RETRIEVAL_POLICY:
        raise ValueError("Model/retrieval identity is not query-gradient v2")
    numeric = {name: value for name, value in mapping.items() if name not in {"model_schema", "retrieval_policy"}}
    for name, value in numeric.items():
        if name == "dropout":
            if _number(value, name) > 1:
                raise ValueError("dropout must be in [0, 1]")
        else:
            _integer(value, name)
    if min(numeric["d_model"], numeric["latent_dim"]) < 2 or numeric["k"] > numeric["num_assets"]:
        raise ValueError("Hidden widths must be >= 2 and k cannot exceed num_assets")
    return QueryGradientConfigV2(**numeric)


class QueryGradientMemoryV2(SpectralMemory):
    """Legacy prototype writes with a query-differentiable selected-neighbor read."""
    retrieval_policy = RETRIEVAL_POLICY

    def __init__(self, key_dim, value_dim, max_items=128, top_k=4, temperature=0.2,
                 merge_radius=0.18, min_salience=0.25, device="cpu", *, dtype=None):
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        if self.dtype not in FLOAT_DTYPES.values():
            raise ValueError("Query-gradient v2 supports float32 and float64")
        super().__init__(key_dim, value_dim, max_items, top_k, temperature, merge_radius, min_salience, device)
        self.clear()
        self._validate_state()

    def clear(self):
        self.state = MemoryState(
            torch.empty(0, self.key_dim, dtype=self.dtype, device=self.device),
            torch.empty(0, self.value_dim, dtype=self.dtype, device=self.device),
            torch.empty(0, dtype=self.dtype, device=self.device),
            torch.empty(0, dtype=torch.int64, device=self.device),
        )

    def _validate_state(self):
        if self.retrieval_policy != RETRIEVAL_POLICY:
            raise ValueError("Memory retrieval identity is not query-gradient v2")
        for name in ("key_dim", "value_dim", "max_items", "top_k"):
            _integer(getattr(self, name), name)
        _number(self.temperature, "temperature", strictly_positive=True)
        _number(self.merge_radius, "merge_radius")
        _number(self.min_salience, "min_salience")
        keys = _tensor(self.state.keys, "memory.keys", dtype=self.dtype, device=self.device)
        if keys.ndim != 2 or keys.shape[1] != self.key_dim or keys.shape[0] > self.max_items:
            raise ValueError("Memory keys disagree with dimensions/capacity")
        size = keys.shape[0]
        _tensor(self.state.values, "memory.values", shape=(size, self.value_dim), dtype=self.dtype, device=self.device)
        scores = _tensor(self.state.salience, "memory.salience", shape=(size,), device=self.device)
        if not scores.is_floating_point() or (scores < 0).any():
            raise ValueError("Memory salience must be floating and nonnegative")
        regimes = _tensor(self.state.regimes, "memory.regimes", shape=(size,), dtype=torch.int64, device=self.device)
        if ((regimes < 0) | (regimes > 2)).any():
            raise ValueError("Memory regimes must be 0, 1 or 2")

    def retrieve(self, query: torch.Tensor):
        self._validate_state()
        _tensor(query, "query", dtype=self.dtype, device=self.device)
        if query.ndim != 2 or query.shape[1] != self.key_dim:
            raise ValueError("Expected query [B, key_dim]")
        if self.state.size == 0:
            return torch.zeros(query.shape[0], self.value_dim, dtype=query.dtype, device=query.device), {
                "weights": torch.empty(query.shape[0], 0, dtype=query.dtype, device=query.device),
            }
        # Detaching alone would still alias live state, which a later write may
        # change before backward. Each read owns both tensors used by its graph.
        keys = self.state.keys.detach().clone()
        values = self.state.values.detach().clone()
        if not torch.isfinite(query.norm(dim=-1)).all() or not torch.isfinite(keys.norm(dim=-1)).all():
            raise ValueError("Cosine normalization exceeds the finite tensor dtype")
        q = self._normalize(query)
        k = self._normalize(keys)
        sim = torch.matmul(q, k.T)
        vals, indices = torch.topk(sim, k=min(self.top_k, sim.shape[-1]), dim=-1)
        weights = torch.softmax(vals / max(self.temperature, 1e-6), dim=-1)
        result = torch.sum(weights.unsqueeze(-1) * values[indices], dim=1)
        if not all(torch.isfinite(tensor).all() for tensor in (q, k, sim, weights, result)):
            raise ValueError("Query-gradient retrieval exceeds the finite tensor dtype")
        return result, {"weights": weights, "indices": indices, "similarity": vals}


class QueryGradientEigenJEPAV2(EigenJEPA):
    """Explicit successor model with unchanged parameter allocation and forward map."""

    def __init__(self, cfg: QueryGradientConfigV2, target_dim: int):
        if type(cfg) is not QueryGradientConfigV2:
            raise ValueError("The v2 model requires an explicit QueryGradientConfigV2")
        cfg = query_config_v2(asdict(cfg))
        self.target_dim_v2 = _integer(target_dim, "target_dim")
        if target_dim != cfg.k + cfg.num_assets ** 2 + 4:
            raise ValueError("target_dim disagrees with the declared spectral target format")
        super().__init__(cfg, target_dim)

    def forward(self, x, memory=None, gate_override=None, memory_scale=1.0):
        if memory is not None and not isinstance(memory, QueryGradientMemoryV2):
            raise ValueError("The v2 model requires v2 memory or explicit memory=None")
        return super().forward(x, memory, gate_override, memory_scale)


@dataclass(frozen=True)
class RestoredQueryCheckpointV2:
    model: QueryGradientEigenJEPAV2
    memory: QueryGradientMemoryV2
    checkpoint_sha256: str


def _memory_configuration(memory):
    memory._validate_state()
    return {**{name: getattr(memory, name) for name in (
        "key_dim", "value_dim", "max_items", "top_k", "temperature", "merge_radius", "min_salience",
    )}, "dtype": str(memory.dtype).removeprefix("torch."), "retrieval_policy": RETRIEVAL_POLICY}


def _restore_payload(payload, *, device="cpu"):
    expected = {"schema", "model_schema", "retrieval_policy", "model_config", "target_dim",
                "model_dtype", "model_state", "memory_config", "memory_state"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Expected the complete query-gradient v2 checkpoint")
    if (payload["schema"], payload["model_schema"], payload["retrieval_policy"]) != (
        CHECKPOINT_SCHEMA, MODEL_SCHEMA, RETRIEVAL_POLICY,
    ):
        raise ValueError("Checkpoint model/retrieval identity is not query-gradient v2")
    cfg = query_config_v2(payload["model_config"])
    dtype_name = payload["model_dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in FLOAT_DTYPES:
        raise ValueError("Checkpoint model dtype must be float32 or float64")
    dtype = FLOAT_DTYPES[dtype_name]
    # Loading weights replaces initialization, so restore without consuming the
    # caller's RNG or depending on its default device/dtype for the final tensors.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = QueryGradientEigenJEPAV2(cfg, payload["target_dim"]).to(dtype=dtype)
    state = payload["model_state"]
    template = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(template):
        raise ValueError("Checkpoint model tensor names do not match its config")
    for name, tensor in state.items():
        _tensor(tensor, f"model.{name}", shape=template[name].shape, dtype=dtype)
    model.load_state_dict(state, strict=True)
    policy = payload["memory_config"]
    policy_names = {"key_dim", "value_dim", "max_items", "top_k", "temperature", "merge_radius", "min_salience", "dtype", "retrieval_policy"}
    if not isinstance(policy, dict) or set(policy) != policy_names:
        raise ValueError("Checkpoint must retain all memory policy settings")
    if (policy["retrieval_policy"] != RETRIEVAL_POLICY or policy["dtype"] != dtype_name
            or policy["key_dim"] != cfg.memory_dim or policy["value_dim"] != cfg.latent_dim):
        raise ValueError("Checkpoint memory policy disagrees with its model")
    memory = QueryGradientMemoryV2(**{name: value for name, value in policy.items()
                                     if name not in {"dtype", "retrieval_policy"}}, dtype=dtype, device="cpu")
    saved_memory = payload["memory_state"]
    if not isinstance(saved_memory, dict) or set(saved_memory) != {"keys", "values", "salience", "regimes"}:
        raise ValueError("Checkpoint memory state has missing or unknown tensors")
    memory.state = MemoryState(**{name: _tensor(tensor, f"memory.{name}").detach().cpu().clone()
                                  for name, tensor in saved_memory.items()})
    memory._validate_state()
    return model.to(device).eval(), memory.to(device)


def save_query_checkpoint_v2(path, model: QueryGradientEigenJEPAV2,
                             memory: QueryGradientMemoryV2) -> str:
    """Save an owned v2 model/bank to a fresh file and return its byte SHA-256.

    This records the default forward call (gate_override=None, memory_scale=1).
    Per-call ablation overrides, optimizer state and RNG/loader state are outside
    this format. A fresh interpreter and unchanged source checkout are required
    for reproducible execution; hashes are not secure execution attestation.
    """
    if type(model) is not QueryGradientEigenJEPAV2 or type(memory) is not QueryGradientMemoryV2:
        raise ValueError("Only the explicit query-gradient v2 model and bank can be saved")
    payload = {
        "schema": CHECKPOINT_SCHEMA, "model_schema": MODEL_SCHEMA,
        "retrieval_policy": RETRIEVAL_POLICY, "model_config": asdict(model.cfg),
        "target_dim": model.target_dim_v2,
        "model_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "memory_config": _memory_configuration(memory),
        "memory_state": {name: getattr(memory.state, name).detach().cpu().clone()
                         for name in ("keys", "values", "salience", "regimes")},
    }
    # Refuse internally inconsistent metadata/tensors before creating an artifact.
    _restore_payload(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_query_checkpoint_v2(path, *, device="cpu") -> RestoredQueryCheckpointV2:
    """Read/hash/decode one byte snapshot; a legacy checkpoint is never upgraded."""
    raw = Path(path).read_bytes()
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, ValueError, IndexError) as exc:
        raise ValueError("Unsupported weights-only query-gradient v2 checkpoint") from exc
    model, memory = _restore_payload(payload, device=device)
    return RestoredQueryCheckpointV2(model, memory, hashlib.sha256(raw).hexdigest())

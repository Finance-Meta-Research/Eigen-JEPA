"""Replay a declared checkpoint policy on an explicit development-window snapshot.

This opt-in entry point does not regenerate data, open a checkpoint's CSV path,
or authorize a scientific run. The legacy evaluator and source bindings remain
unchanged. Only the repository's dense float32 checkpoint format is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .covariance_geometry_v3 import evaluate_model_geometry_v3
from .data import batch_collate
from .memory import SpectralMemory
from .model import EigenJEPA, EigenJEPAConfig

SCHEMA = "eigen_jepa.checkpoint_evaluation.v3"
SNAPSHOT_SCHEMA = "eigen_jepa.development_windows.v1"
VARIANTS = frozenset({
    "full", "no_memory", "no_gate", "no_regime", "no_spectral",
    "no_subspace", "no_mask", "no_tailweight",
})
SOURCE_MODULES = (
    "checkpoint_evaluation_v3.py", "covariance_geometry_v3.py", "evaluation_v2.py",
    "data.py", "model.py", "memory.py", "spectral.py",
)


def _mapping(value, name):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _required(mapping, key, name):
    if key not in mapping:
        raise ValueError(f"missing {name}.{key}")
    return mapping[key]


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _tensor(value, name, *, shape=None, dtype=None):
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.dtype not in {torch.float32, torch.float64, torch.int64, torch.bool}
    ):
        raise ValueError(f"{name} must be a dense float32/float64/int64/bool tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _read_tensor_artifact(path, name):
    # Hash and deserialize the same owned byte string, with no unrestricted retry.
    raw = Path(path).read_bytes()
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} is not a supported weights-only tensor artifact") from exc
    return _mapping(payload, name), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RestoredCheckpoint:
    model: EigenJEPA
    memory: SpectralMemory
    checkpoint_sha256: str
    target_dim: int
    data_configuration: dict
    inference_policy: dict


def restore_checkpoint_v3(path, *, device="cpu") -> RestoredCheckpoint:
    """Restore weights, all memory settings and the saved variant without guessing.

    Unversioned training checkpoints are accepted only when their existing
    ``cfg``, ``args``, ``variant`` and ``memory_state`` identify the full policy.
    Other saved arguments, including external data paths, are never executed.
    """
    checkpoint, checkpoint_hash = _read_tensor_artifact(path, "checkpoint")
    config = _mapping(_required(checkpoint, "cfg", "checkpoint"), "checkpoint.cfg")
    expected_fields = {field.name for field in fields(EigenJEPAConfig)}
    if set(config) != expected_fields:
        raise ValueError("checkpoint.cfg must contain exactly the saved EigenJEPAConfig fields")
    for name in expected_fields - {"dropout"}:
        _integer(config[name], f"checkpoint.cfg.{name}")
    dropout = _number(config["dropout"], "checkpoint.cfg.dropout")
    if not 0 <= dropout <= 1 or min(config["d_model"], config["latent_dim"]) < 2:
        raise ValueError("checkpoint.cfg requires dropout in [0, 1] and hidden widths >= 2")
    if config["k"] > config["num_assets"]:
        raise ValueError("checkpoint.cfg.k must not exceed num_assets")
    cfg = EigenJEPAConfig(**config)
    target_dim = _integer(_required(checkpoint, "target_dim", "checkpoint"), "target_dim")
    if target_dim != cfg.k + cfg.num_assets**2 + 4:
        raise ValueError("target_dim disagrees with the declared spectral target format")
    data_config = _mapping(
        _required(checkpoint, "data_cfg", "checkpoint"), "checkpoint.data_cfg",
    )
    try:
        data_config = json.loads(json.dumps(data_config, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint.data_cfg must be finite JSON-compatible metadata") from exc

    args = _mapping(_required(checkpoint, "args", "checkpoint"), "checkpoint.args")
    variant = _required(checkpoint, "variant", "checkpoint")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("checkpoint.variant is not a supported training variant")
    if _required(args, "variant", "checkpoint.args") != variant:
        raise ValueError("checkpoint.variant and checkpoint.args.variant disagree")
    saved = {
        "max_items": _integer(_required(args, "memory_size", "checkpoint.args"), "memory_size"),
        "top_k": _integer(_required(args, "memory_top_k", "checkpoint.args"), "memory_top_k"),
        "temperature": _number(_required(args, "memory_temperature", "checkpoint.args"), "memory_temperature"),
        "merge_radius": _number(_required(args, "merge_radius", "checkpoint.args"), "merge_radius"),
        "min_salience": _number(_required(args, "min_salience", "checkpoint.args"), "min_salience"),
    }
    if saved["temperature"] <= 0 or saved["merge_radius"] < 0:
        raise ValueError("memory_temperature must be positive and merge_radius nonnegative")
    memory_state = _mapping(
        _required(checkpoint, "memory_state", "checkpoint"), "checkpoint.memory_state",
    )
    if set(memory_state) != {"keys", "values", "salience", "regimes", "meta"}:
        raise ValueError("checkpoint.memory_state has missing or unknown fields")
    meta = _tensor(memory_state["meta"], "memory.meta", shape=(4,), dtype=torch.int64)
    if meta.tolist() != [cfg.memory_dim, cfg.latent_dim, saved["max_items"], saved["top_k"]]:
        raise ValueError("memory.meta disagrees with saved dimensions/capacity/top_k")
    keys = _tensor(memory_state["keys"], "memory.keys", dtype=torch.float32)
    if keys.ndim != 2 or keys.shape[1] != cfg.memory_dim or keys.shape[0] > saved["max_items"]:
        raise ValueError("memory.keys must fit the declared key dimension and capacity")
    size = keys.shape[0]
    _tensor(memory_state["values"], "memory.values", shape=(size, cfg.latent_dim), dtype=torch.float32)
    _tensor(memory_state["salience"], "memory.salience", shape=(size,), dtype=torch.float32)
    regimes = _tensor(memory_state["regimes"], "memory.regimes", shape=(size,), dtype=torch.int64)
    if not bool(((regimes >= 0) & (regimes <= 2)).all()):
        raise ValueError("memory.regimes must contain labels 0, 1 or 2")

    # Loaded weights replace initialization; restoration must not consume the
    # caller's random stream. Construction is on CPU before the requested move.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = EigenJEPA(cfg, target_dim=target_dim).float()
    state = _mapping(_required(checkpoint, "model", "checkpoint"), "checkpoint.model")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint.model keys do not match the declared model")
    for name, template in expected.items():
        _tensor(state[name], f"model.{name}", shape=template.shape, dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    effective_memory = {**saved, "min_salience": 10.0 if variant == "no_memory" else saved["min_salience"]}
    memory = SpectralMemory(
        key_dim=cfg.memory_dim, value_dim=cfg.latent_dim, device=device, **effective_memory,
    )
    memory.load_state_dict(memory_state)
    policy = {
        "variant": variant,
        "memory_enabled": variant != "no_memory",
        "gate_override": 1.0 if variant == "no_gate" else None,
        "memory_scale": 1.0,
        "saved_memory_configuration": saved,
        "effective_memory_configuration": effective_memory,
        "memory_items": size,
    }
    return RestoredCheckpoint(model, memory, checkpoint_hash, target_dim, data_config, policy)


def save_development_snapshot(path, windows: Iterable[dict], *, source_reference: str) -> str:
    """Materialize already selected windows once, preserving an existing file.

    Use an independently authorized development population. The source reference
    is a supplied label, not independent provenance. Full shape/semantic checks
    occur against the checkpoint when the snapshot is evaluated.
    """
    if not isinstance(source_reference, str) or not source_reference.strip():
        raise ValueError("source_reference must be a nonempty string")
    owned = []
    for index, row in enumerate(windows):
        row = _mapping(row, f"window[{index}]")
        owned.append({key: _tensor(value, f"window[{index}].{key}").detach().cpu().clone()
                      for key, value in row.items()})
    if not owned:
        raise ValueError("snapshot windows must be nonempty")
    payload = {"schema": SNAPSHOT_SCHEMA, "evaluation_mode": "development",
               "source_reference": source_reference, "windows": owned}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _load_snapshot(path, restored):
    snapshot, snapshot_hash = _read_tensor_artifact(path, "snapshot")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("evaluation_mode") != "development":
        raise ValueError("snapshot must use the explicit development-window schema and mode")
    reference = snapshot.get("source_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("snapshot.source_reference must be a nonempty string")
    windows = snapshot.get("windows")
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("snapshot.windows must be a nonempty list of tensor mappings")
    cfg = restored.model.cfg
    shapes = {
        "x": (cfg.context_len, cfg.input_dim + 1),
        "returns_ctx": (cfg.context_len, cfg.num_assets),
        "target_vec": (restored.target_dim,), "evals_true": (cfg.k,),
        "cov_true": (cfg.num_assets, cfg.num_assets),
        "proj_true": (cfg.num_assets, cfg.num_assets),
        **{name: () for name in ("drift_true", "risk_true", "entropy_true", "rank_true", "event_true")},
        "window_index": (), "regime_true": (),
    }
    identities, layouts = set(), None
    for index, row in enumerate(windows):
        row = _mapping(row, f"window[{index}]")
        for name, shape in shapes.items():
            dtype = torch.int64 if name in {"window_index", "regime_true"} else torch.float32
            _tensor(_required(row, name, f"window[{index}]"), f"window[{index}].{name}", shape=shape, dtype=dtype)
        for name, value in row.items():
            _tensor(value, f"window[{index}].{name}")
        layout = {name: (tuple(value.shape), value.dtype) for name, value in row.items()}
        if layouts is not None and layout != layouts:
            raise ValueError("snapshot windows must share field names, shapes and dtypes")
        layouts = layout
        identity = row["window_index"].item()
        if identity < 0 or identity in identities:
            raise ValueError("snapshot window_index values must be nonnegative and unique")
        identities.add(identity)
        if row["regime_true"].item() not in {0, 1, 2} or row["event_true"].item() not in {0, 1}:
            raise ValueError("snapshot requires binary event and 0/1/2 regime labels")
    return list(windows), snapshot_hash, reference


def _digest(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def evaluate_checkpoint_v3(
    checkpoint_path, snapshot_path, *, batch_size=32, device="cpu",
    event_threshold=0.45, rank_rtol=None,
) -> dict:
    """Return a digest-bound full-cohort report using the saved inference policy."""
    _integer(batch_size, "batch_size")
    threshold = _number(event_threshold, "event_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("event_threshold must be in [0, 1]")
    restored = restore_checkpoint_v3(checkpoint_path, device=device)
    windows, snapshot_hash, reference = _load_snapshot(snapshot_path, restored)
    policy = restored.inference_policy
    # DataLoader creates its iterator seed tensor during iteration. Bind those
    # factories as well as construction; caller defaults must not create meta
    # tensors. The model and its input moves still use the requested device.
    with torch.device("cpu"):
        loader = DataLoader(
            windows, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=batch_collate, generator=torch.Generator().manual_seed(0),
        )
        report = evaluate_model_geometry_v3(
            restored.model, loader, device=device,
            memory=restored.memory if policy["memory_enabled"] else None,
            gate_override=policy["gate_override"], memory_scale=policy["memory_scale"],
            event_threshold=threshold, rank_rtol=rank_rtol,
        )
    source_dir = Path(__file__).resolve().parent
    payload = {
        "schema": SCHEMA, "evaluation_mode": "development",
        "checkpoint_sha256": restored.checkpoint_sha256,
        "snapshot_sha256": snapshot_hash,
        "input_provenance": {"supplied_source_reference": reference,
                             "independently_verified": False, "protected_run_authorized": False},
        "model_configuration": asdict(restored.model.cfg),
        "target_dim": restored.target_dim, "data_configuration": restored.data_configuration,
        "inference_policy": policy,
        "execution": {"batch_size": batch_size, "shuffle": False, "num_workers": 0,
                      "device": str(next(restored.model.parameters()).device)},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "platform": platform.platform(),
                        "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in SOURCE_MODULES},
        "evaluation": report,
    }
    return {**payload, "result_sha256": _digest(payload)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--event-threshold", type=float, default=0.45)
    parser.add_argument("--rank-rtol", type=float)
    args = parser.parse_args(argv)
    try:
        report = evaluate_checkpoint_v3(
            args.checkpoint, args.snapshot, batch_size=args.batch_size, device=args.device,
            event_threshold=args.event_threshold, rank_rtol=args.rank_rtol,
        )
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(2, f"checkpoint evaluation failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "result_sha256": report["result_sha256"]}))


if __name__ == "__main__":
    main()

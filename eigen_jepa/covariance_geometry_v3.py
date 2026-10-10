"""Opt-in evaluation of the covariance actually represented by legacy factors.

This inference-only API preserves the model's covariance and leaves historical
model, training, and evaluation definitions unchanged. A successor must select
and source-bind this geometry and its numerical-rank policy before outcome use.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real

import torch

from .evaluation_v2 import evaluation_report, window_records

SCHEMA = "eigen_jepa.covariance_geometry_evaluation.v3"


@dataclass(frozen=True)
class CovarianceGeometry:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    support_projector: torch.Tensor
    covariance: torch.Tensor
    support_rank: torch.Tensor
    reconstruction_relative_max_error: torch.Tensor
    rank_rtol: float


def _tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.is_complex()
        or value.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a dense real floating tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().to(dtype=torch.float64)


@torch.no_grad()
def covariance_factor_geometry(
    weights: torch.Tensor,
    vectors: torch.Tensor,
    *,
    rank_rtol: float | None = None,
) -> CovarianceGeometry:
    """Compute the eigensystem and support of ``U diag(w) U.T`` in float64.

    ``w`` has shape [batch, k], ``U`` has shape [batch, assets, k], and k cannot
    exceed assets. No orthogonality, unit norm, ordering, or full rank is assumed.
    Eigenvalues retain every singular value; only the support projector uses
    ``s > rank_rtol * s_max``. The zero covariance has zero support.

    This operator deliberately provides no training gradient through the SVD.
    Rank-deficient and repeated spectra need no arbitrary differentiable basis.
    """
    w, u = _tensor(weights, "weights"), _tensor(vectors, "vectors")
    if (
        w.ndim != 2
        or u.ndim != 3
        or w.shape[0] == 0
        or w.shape[0] != u.shape[0]
        or w.shape[1] != u.shape[2]
        or not 1 <= w.shape[1] <= u.shape[1]
        or w.device != u.device
    ):
        raise ValueError(
            "weights [batch, k] and vectors [batch, assets, k] must share a "
            "nonempty batch, k <= assets, and device"
        )
    if (w < 0).any():
        raise ValueError("covariance factor weights must be nonnegative")
    assets, k = u.shape[-2:]
    if rank_rtol is None:
        tolerance = max(assets, k) * torch.finfo(torch.float64).eps
    else:
        if isinstance(rank_rtol, bool) or not isinstance(rank_rtol, Real):
            raise ValueError("rank_rtol must be a finite real number in [0, 1)")
        try:
            tolerance = float(rank_rtol)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "rank_rtol must be a finite real number in [0, 1)"
            ) from exc
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("rank_rtol must be a finite real number in [0, 1)")

    factor = u * torch.sqrt(w).unsqueeze(-2)
    if not torch.isfinite(factor).all():
        raise ValueError("weighted covariance factors exceed the float64 domain")
    covariance = factor @ factor.transpose(-1, -2)
    if not torch.isfinite(covariance).all():
        raise ValueError("factor covariance exceeds the finite float64 domain")
    scale = factor.abs().amax(dim=(-2, -1), keepdim=True)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    # Thin SVD preserves the covariance; QR with unchanged weights would not.
    try:
        basis, singular, _ = torch.linalg.svd(factor / safe_scale, full_matrices=False)
    except RuntimeError as exc:
        raise ValueError(
            "covariance-factor SVD failed; no fallback was substituted"
        ) from exc
    eigenvalues = (singular * scale.squeeze(-1)).square()
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(basis).all():
        raise ValueError("covariance eigensystem exceeds the finite float64 domain")
    supported = singular > tolerance * singular[..., :1]
    if (supported & (eigenvalues == 0)).any():
        raise ValueError(
            "supported covariance eigenvalues underflow the float64 domain"
        )
    support = (basis * supported.unsqueeze(-2)) @ basis.transpose(-1, -2)

    reconstructed = (basis * eigenvalues.unsqueeze(-2)) @ basis.transpose(-1, -2)
    if not torch.isfinite(reconstructed).all():
        raise ValueError("reconstructed covariance exceeds the finite float64 domain")
    cov_scale = covariance.abs().amax(dim=(-2, -1), keepdim=True)
    safe_cov_scale = torch.where(cov_scale > 0, cov_scale, torch.ones_like(cov_scale))
    error = (
        (reconstructed / safe_cov_scale - covariance / safe_cov_scale)
        .abs()
        .amax(dim=(-2, -1))
    )
    if (
        not torch.isfinite(error).all()
        or (error > 64 * assets * torch.finfo(torch.float64).eps).any()
    ):
        raise ValueError("SVD reconstruction failed the declared float64 tolerance")
    return CovarianceGeometry(
        eigenvalues=eigenvalues,
        eigenvectors=basis,
        support_projector=support,
        covariance=covariance,
        support_rank=supported.sum(dim=-1),
        reconstruction_relative_max_error=error,
        rank_rtol=tolerance,
    )


def _tensor_digest(named_tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named_tensors):
        tensor = named_tensors[name].detach().cpu().contiguous()
        header = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_model_geometry_v3(
    model,
    loader: Iterable,
    device="cpu",
    memory=None,
    *,
    rank_rtol: float | None = None,
    gate_override=None,
    memory_scale=1.0,
    event_threshold=0.45,
) -> dict:
    """Evaluate the unchanged model's covariance spectrum and numerical support.

    Reuse frozen input windows across comparisons. Other learned heads and v2
    cohort reductions keep their original meanings. The report binds the raw
    factors and their derived covariance, but callers must retain checkpoints,
    inputs, data/split provenance, source and environment in their run manifest.
    """
    model.to(device).eval()
    records, geometry_records = [], []
    cohort_policy = None
    for batch in loader:
        raw = model(
            batch["x"].to(device),
            memory=memory,
            gate_override=gate_override,
            memory_scale=memory_scale,
        )
        geometry = covariance_factor_geometry(
            raw["eig"], raw["sub"], rank_rtol=rank_rtol
        )
        policy = {
            "assets": raw["sub"].shape[-2],
            "factor_columns": raw["sub"].shape[-1],
            "rank_rtol": geometry.rank_rtol,
        }
        if cohort_policy is not None and policy != cohort_policy:
            raise ValueError(
                "all windows must share the same covariance geometry policy"
            )
        cohort_policy = policy
        prediction = {
            **raw,
            "eig": geometry.eigenvalues,
            "sub": geometry.eigenvectors,
            "proj": geometry.support_projector,
        }
        target = model.target_latent(batch["target_vec"].to(device))
        rows = window_records(batch, prediction, latent_target=target)
        records.extend(rows)
        for i, row in enumerate(rows):
            geometry_records.append(
                {
                    "window_index": row["window_index"],
                    "factor_sha256": _tensor_digest(
                        {"weights": raw["eig"][i], "vectors": raw["sub"][i]}
                    ),
                    "covariance_sha256": _tensor_digest(
                        {"covariance": geometry.covariance[i]}
                    ),
                    "support_rank": int(geometry.support_rank[i].item()),
                    "covariance_eigenvalues": geometry.eigenvalues[i].cpu().tolist(),
                    "reconstruction_relative_max_error": float(
                        geometry.reconstruction_relative_max_error[i].item()
                    ),
                }
            )
    # The existing reducer validates the complete cohort, including unique IDs.
    reduced = evaluation_report(records, event_threshold=event_threshold)
    payload = {
        **{
            key: value
            for key, value in reduced.items()
            if key not in {"schema", "result_sha256"}
        },
        "schema": SCHEMA,
        "geometry": {
            **cohort_policy,
            "covariance": "U diag(w) U^T; original legacy factor covariance",
            "eigenvalues": "descending squared singular values of U sqrt(diag(w))",
            "projector": "numerical support; s > rank_rtol * largest_singular_value",
            "arithmetic": "float64; maximum-absolute-entry scaled thin SVD",
            "unchanged_heads": [
                "drift",
                "risk",
                "entropy",
                "rank",
                "gate",
                "regime_logits",
            ],
            "cohort_reducer": "eigen_jepa.evaluation.v2; original EPS=1e-12 policies",
        },
        "geometry_records": sorted(
            geometry_records, key=lambda row: row["window_index"]
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {**payload, "result_sha256": digest}

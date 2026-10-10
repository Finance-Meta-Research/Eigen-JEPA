"""Generated checkpoint controls; no retained scientific outcomes are loaded."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from eigen_jepa.covariance_geometry_v3 import evaluate_model_geometry_v3
from eigen_jepa.data import MarketConfig, batch_collate, build_datasets
from eigen_jepa.memory import SpectralMemory
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig


def checkpoint_fixture(variant="full"):
    """Construct an untrained, occupied-memory model with known inference policy."""
    torch.manual_seed(212)
    data_cfg = MarketConfig(
        num_assets=3, total_steps=60, context_len=8, horizon=3,
        num_train=8, num_val=5, num_test=5, seed=212,
        mask_ratio=0.0 if variant == "no_mask" else 0.25,
    )
    cfg = EigenJEPAConfig(
        input_dim=7, num_assets=3, context_len=8, k=2,
        d_model=16, latent_dim=16, memory_dim=16,
    )
    model = EigenJEPA(cfg, target_dim=15).eval()
    with torch.no_grad():
        model.query_head[-1].weight.zero_()
        model.query_head[-1].bias.zero_()
        model.query_head[-1].bias[0] = 1
        model.gate_head[-1].weight.zero_()
        model.gate_head[-1].bias.fill_(-2)
        model.memory_projector.weight.copy_(torch.eye(16))
    memory = SpectralMemory(
        key_dim=16, value_dim=16, max_items=7, top_k=2,
        temperature=0.7, merge_radius=0.01, min_salience=0.6,
    )
    memory.write(
        torch.eye(16)[:2], torch.stack([torch.full((16,), 8.), torch.full((16,), -8.)]),
        torch.tensor([0.9, 0.9]), torch.tensor([0, 2]),
    )
    if variant == "no_memory":
        memory.min_salience = 10.0
        memory.clear()
    checkpoint = {
        "cfg": asdict(cfg), "target_dim": 15, "data_cfg": asdict(data_cfg),
        "model": model.state_dict(), "memory_state": memory.state_dict(),
        "variant": variant, "history": [],
        "args": {
            "variant": variant, "memory_size": 7, "memory_top_k": 2,
            "memory_temperature": 0.7, "merge_radius": 0.01,
            "min_salience": 0.6, "device": torch.device("cpu"),
        },
    }
    return checkpoint, model, memory, data_cfg


def training_windows(data_cfg):
    dataset = build_datasets(data_cfg, k=2)["train"]
    return [{key: value.clone() for key, value in dataset[i].items()} for i in range(5)]


def legacy_factory_witness(directory: Path, variant: str):
    """Intercept the *actual* legacy CLI's first evaluation, before artifact writes."""
    from eigen_jepa import eval as legacy_eval

    checkpoint, original, original_memory, _ = checkpoint_fixture(variant)
    checkpoint_path = directory / (variant + ".pt")
    torch.save(checkpoint, checkpoint_path)
    captured = {}

    class StopAfterFactory(Exception):
        pass

    def intercept(loaded, loader, device, memory=None, gate_override=None, **kwargs):
        batch = next(iter(loader))
        with torch.no_grad():
            actual = loaded(batch["x"], memory=memory, gate_override=gate_override)
            expected = original(
                batch["x"], memory=None if variant == "no_memory" else original_memory,
                gate_override=1.0 if variant == "no_gate" else None,
            )
        captured.update({
            "variant": variant,
            "saved_temperature": checkpoint["args"]["memory_temperature"],
            "loaded_temperature": memory.temperature,
            "saved_merge_radius": checkpoint["args"]["merge_radius"],
            "loaded_merge_radius": memory.merge_radius,
            "expected_memory_is_none": variant == "no_memory",
            "loaded_memory_is_none": memory is None,
            "expected_gate_mean": expected["gate"].mean().item(),
            "loaded_gate_mean": actual["gate"].mean().item(),
            "max_latent_difference": (actual["z_hat"] - expected["z_hat"]).abs().max().item(),
        })
        raise StopAfterFactory

    output = directory / "legacy-output-must-not-be-created"
    with patch.object(legacy_eval, "evaluate_model", side_effect=intercept), patch.object(
        sys, "argv", ["eigen_jepa.eval", "--checkpoint", str(checkpoint_path), "--out_dir", str(output)],
    ), pytest.raises(StopAfterFactory):
        legacy_eval.main()
    assert not output.exists()
    return captured


@pytest.mark.parametrize("variant", ["full", "no_gate", "no_memory"])
def test_preserved_legacy_factory_is_a_discriminating_failure_control(tmp_path, variant):
    witness = legacy_factory_witness(tmp_path, variant)
    assert witness["loaded_temperature"] == 0.2
    assert witness["saved_temperature"] == 0.7
    assert witness["loaded_merge_radius"] == 0.18
    if variant == "no_gate":
        assert witness["expected_gate_mean"] == 1.0
        assert witness["loaded_gate_mean"] < 0.2
    if variant == "no_memory":
        assert witness["expected_memory_is_none"]
        assert not witness["loaded_memory_is_none"]
        assert witness["max_latent_difference"] == 0  # Valid training checkpoints have empty memory.
    else:
        assert witness["max_latent_difference"] > 0.1


def serialized_fixture(tmp_path, variant="full"):
    from eigen_jepa.checkpoint_evaluation_v3 import save_development_snapshot

    checkpoint, model, memory, data_cfg = checkpoint_fixture(variant)
    windows = training_windows(data_cfg)
    checkpoint_path, snapshot_path = tmp_path / "checkpoint.pt", tmp_path / "windows.pt"
    torch.save(checkpoint, checkpoint_path)
    save_development_snapshot(snapshot_path, windows, source_reference="generated-training-fixture:212")
    return checkpoint_path, snapshot_path, model, memory, windows


@pytest.mark.parametrize("variant", [
    "full", "no_gate", "no_memory", "no_regime", "no_spectral", "no_subspace", "no_mask", "no_tailweight",
])
def test_restored_raw_predictions_and_full_cohort_match_original_model(tmp_path, variant):
    from eigen_jepa.checkpoint_evaluation_v3 import (
        evaluate_checkpoint_v3,
        restore_checkpoint_v3,
    )

    checkpoint_path, snapshot_path, original, original_memory, windows = serialized_fixture(tmp_path, variant)
    frozen = copy.deepcopy(windows)
    rng_before = torch.get_rng_state().clone()
    restored = restore_checkpoint_v3(checkpoint_path)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert not restored.model.training
    for name, value in original.state_dict().items():
        assert torch.equal(restored.model.state_dict()[name], value)
    for name, value in original_memory.state_dict().items():
        assert torch.equal(restored.memory.state_dict()[name], value)
    assert restored.memory.temperature == 0.7
    assert restored.memory.merge_radius == 0.01
    assert restored.memory.min_salience == (10.0 if variant == "no_memory" else 0.6)
    active = variant != "no_memory"
    gate_override = 1.0 if variant == "no_gate" else None
    batch = batch_collate(windows)
    with torch.no_grad():
        expected = original(batch["x"], memory=original_memory if active else None, gate_override=gate_override)
        actual = restored.model(
            batch["x"], memory=restored.memory if restored.inference_policy["memory_enabled"] else None,
            gate_override=restored.inference_policy["gate_override"],
        )
    for name in expected:
        if isinstance(expected[name], torch.Tensor):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    # This oracle calls the existing geometry evaluator with the original
    # in-memory model and explicitly prescribed policy, bypassing restoration.
    batches = [batch_collate(windows[start:start + 2]) for start in range(0, len(windows), 2)]
    direct = evaluate_model_geometry_v3(
        original, batches, memory=original_memory if active else None, gate_override=gate_override,
    )
    report = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    assert report["evaluation"] == direct
    assert report["evaluation"]["metrics"]["window_count"] == 5
    assert report["inference_policy"]["variant"] == variant
    assert report["execution"]["batch_size"] == 2  # Includes the final one-window batch.
    assert report["input_provenance"]["independently_verified"] is False
    assert report["input_provenance"]["protected_run_authorized"] is False
    assert torch.equal(rng_before, torch.get_rng_state())
    for before, after in zip(frozen, windows):
        for name in before:
            assert torch.equal(before[name], after[name])


def test_factory_honors_checkpoint_precision_without_changing_global_defaults(tmp_path):
    from eigen_jepa.checkpoint_evaluation_v3 import (
        evaluate_checkpoint_v3,
        restore_checkpoint_v3,
    )

    path, snapshot_path, original, memory, windows = serialized_fixture(tmp_path)
    batch = batch_collate(windows)
    ordinary_report = evaluate_checkpoint_v3(path, snapshot_path, batch_size=2)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        # A caller's unrelated default device must not turn saved CPU weights
        # into unmaterialized tensors or consume another device's random stream.
        with torch.device("meta"):
            restored = restore_checkpoint_v3(path, device="cpu")
            ambient_report = evaluate_checkpoint_v3(path, snapshot_path, device="cpu", batch_size=2)
            assert torch.get_default_dtype() == torch.float64
            assert torch.empty(0).device.type == "meta"
    finally:
        torch.set_default_dtype(previous_dtype)
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cpu"
               for parameter in restored.model.parameters())
    assert ambient_report == ordinary_report
    with torch.no_grad():
        expected = original(batch["x"], memory=memory)
        actual = restored.model(batch["x"], memory=restored.memory)
    torch.testing.assert_close(actual["z_hat"], expected["z_hat"], rtol=0, atol=0)


def test_snapshot_cli_binds_exact_files_source_and_effective_policy(tmp_path):
    from eigen_jepa import checkpoint_evaluation_v3 as replay

    checkpoint_path, snapshot_path, _, _, _ = serialized_fixture(tmp_path, "no_gate")
    output_path = tmp_path / "evaluation.json"
    result = subprocess.run(
        [sys.executable, "-m", "eigen_jepa.checkpoint_evaluation_v3",
         "--checkpoint", str(checkpoint_path), "--snapshot", str(snapshot_path),
         "--output", str(output_path), "--batch-size", "2", "--rank-rtol", "0.000001"],
        check=True, text=True, capture_output=True, env=os.environ.copy(), timeout=40,
    )
    report = json.loads(output_path.read_text())
    assert report["schema"] == replay.SCHEMA
    assert report["evaluation"]["metrics"]["window_count"] == 5
    assert report["inference_policy"]["gate_override"] == 1.0
    assert report["inference_policy"]["effective_memory_configuration"]["temperature"] == 0.7
    assert report["evaluation"]["geometry"]["rank_rtol"] == 0.000001
    assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert report["snapshot_sha256"] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    for name, digest in report["source_sha256"].items():
        assert digest == hashlib.sha256((Path(replay.__file__).parent / name).read_bytes()).hexdigest()
    claimed = report.pop("result_sha256")
    independent = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    assert claimed == independent == json.loads(result.stdout)["result_sha256"]


def test_existing_snapshot_and_report_are_preserved(tmp_path, capsys):
    from eigen_jepa.checkpoint_evaluation_v3 import main, save_development_snapshot

    checkpoint_path, snapshot_path, _, _, windows = serialized_fixture(tmp_path)
    snapshot_bytes = snapshot_path.read_bytes()
    with pytest.raises(FileExistsError):
        save_development_snapshot(snapshot_path, windows, source_reference="a second population")
    assert snapshot_path.read_bytes() == snapshot_bytes
    output = tmp_path / "evaluation.json"
    output.write_bytes(b"retained-result\n")
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(checkpoint_path), "--snapshot", str(snapshot_path), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"retained-result\n"
    assert "checkpoint evaluation failed" in capsys.readouterr().err


def test_replay_never_regenerates_masks_or_opens_checkpoint_data_path(tmp_path, monkeypatch):
    from eigen_jepa import data
    from eigen_jepa.checkpoint_evaluation_v3 import evaluate_checkpoint_v3

    checkpoint_path, snapshot_path, _, _, _ = serialized_fixture(tmp_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["data_cfg"].update(data_source="csv", csv_path="/missing/must-not-be-opened.csv")
    torch.save(checkpoint, checkpoint_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Data regeneration is outside the frozen-snapshot path")

    monkeypatch.setattr(data, "build_datasets", forbidden)
    monkeypatch.setattr(data, "_load_csv_market", forbidden)
    first = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    second = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    assert first == second
    assert first["data_configuration"]["csv_path"] == "/missing/must-not-be-opened.csv"


def test_policy_or_snapshot_change_produces_a_distinct_receipt(tmp_path):
    from eigen_jepa.checkpoint_evaluation_v3 import evaluate_checkpoint_v3

    checkpoint_path, snapshot_path, _, _, _ = serialized_fixture(tmp_path)
    first = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["args"]["memory_temperature"] = 0.2
    torch.save(checkpoint, checkpoint_path)
    changed = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    assert first["checkpoint_sha256"] != changed["checkpoint_sha256"]
    assert first["evaluation"]["result_sha256"] != changed["evaluation"]["result_sha256"]
    snapshot = torch.load(snapshot_path, weights_only=True)
    snapshot["windows"][0]["x"][0, 0] += 0.25
    torch.save(snapshot, snapshot_path)
    changed_input = evaluate_checkpoint_v3(checkpoint_path, snapshot_path, batch_size=2)
    assert changed["snapshot_sha256"] != changed_input["snapshot_sha256"]
    assert changed["evaluation"]["records"][0]["input_sha256"] != changed_input["evaluation"]["records"][0]["input_sha256"]
    assert len({first["result_sha256"], changed["result_sha256"], changed_input["result_sha256"]}) == 3


@pytest.mark.parametrize("fault", [
    "missing-temperature", "missing-variant", "variant-disagreement", "unknown-variant",
    "capacity-disagreement", "key-shape", "value-shape", "memory-nan", "memory-regime",
    "model-nan", "model-float64", "model-missing-key", "missing-config-field", "target-dimension",
    "zero-temperature", "nan-temperature", "boolean-top-k",
])
def test_ambiguous_or_invalid_checkpoint_is_rejected(tmp_path, fault):
    from eigen_jepa.checkpoint_evaluation_v3 import restore_checkpoint_v3

    checkpoint, _, _, _ = checkpoint_fixture()
    if fault == "missing-temperature":
        del checkpoint["args"]["memory_temperature"]
    elif fault == "missing-variant":
        del checkpoint["variant"]
    elif fault == "variant-disagreement":
        checkpoint["args"]["variant"] = "no_gate"
    elif fault == "unknown-variant":
        checkpoint["variant"] = checkpoint["args"]["variant"] = "unrecorded"
    elif fault == "capacity-disagreement":
        checkpoint["memory_state"]["meta"][2] = 8
    elif fault == "key-shape":
        checkpoint["memory_state"]["keys"] = torch.zeros(2, 15)
    elif fault == "value-shape":
        checkpoint["memory_state"]["values"] = torch.zeros(3, 16)
    elif fault == "memory-nan":
        checkpoint["memory_state"]["values"][0, 0] = float("nan")
    elif fault == "memory-regime":
        checkpoint["memory_state"]["regimes"][0] = 3
    elif fault == "model-nan":
        checkpoint["model"]["context.pos"][0, 0, 0] = float("nan")
    elif fault == "model-float64":
        checkpoint["model"]["context.pos"] = checkpoint["model"]["context.pos"].double()
    elif fault == "model-missing-key":
        del checkpoint["model"]["context.pos"]
    elif fault == "missing-config-field":
        del checkpoint["cfg"]["dropout"]
    elif fault == "target-dimension":
        checkpoint["target_dim"] = 16
    elif fault == "zero-temperature":
        checkpoint["args"]["memory_temperature"] = 0
    elif fault == "nan-temperature":
        checkpoint["args"]["memory_temperature"] = float("nan")
    elif fault == "boolean-top-k":
        checkpoint["args"]["memory_top_k"] = True
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        restore_checkpoint_v3(path)


@pytest.mark.parametrize("fault", [
    "missing-field", "wrong-x-shape", "wrong-target-shape", "float64-input", "nonfinite-target",
    "duplicate-id", "negative-id", "fractional-id", "event-label", "regime-label",
    "different-extra-shape", "empty", "protected-mode", "empty-source",
])
def test_invalid_snapshot_fails_before_any_output(tmp_path, fault, capsys):
    from eigen_jepa.checkpoint_evaluation_v3 import main

    checkpoint_path, snapshot_path, _, _, _ = serialized_fixture(tmp_path)
    snapshot = torch.load(snapshot_path, weights_only=True)
    rows = snapshot["windows"]
    if fault == "missing-field":
        del rows[0]["returns_ctx"]
    elif fault == "wrong-x-shape":
        rows[0]["x"] = rows[0]["x"][:-1]
    elif fault == "wrong-target-shape":
        rows[0]["target_vec"] = rows[0]["target_vec"][:-1]
    elif fault == "float64-input":
        rows[0]["x"] = rows[0]["x"].double()
    elif fault == "nonfinite-target":
        rows[0]["cov_true"][0, 0] = float("inf")
    elif fault == "duplicate-id":
        rows[1]["window_index"] = rows[0]["window_index"].clone()
    elif fault == "negative-id":
        rows[0]["window_index"] = torch.tensor(-1)
    elif fault == "fractional-id":
        rows[0]["window_index"] = torch.tensor(1.5)
    elif fault == "event-label":
        rows[0]["event_true"] = torch.tensor(0.5)
    elif fault == "regime-label":
        rows[0]["regime_true"] = torch.tensor(4)
    elif fault == "different-extra-shape":
        rows[0]["returns_fut"] = rows[0]["returns_fut"][:-1]
    elif fault == "empty":
        snapshot["windows"] = []
    elif fault == "protected-mode":
        snapshot["evaluation_mode"] = "protected"
    elif fault == "empty-source":
        snapshot["source_reference"] = "  "
    torch.save(snapshot, snapshot_path)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(checkpoint_path), "--snapshot", str(snapshot_path), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
    assert "checkpoint evaluation failed" in capsys.readouterr().err


def test_weights_only_rejection_never_retries_unrestricted_loading(tmp_path, monkeypatch):
    from eigen_jepa.checkpoint_evaluation_v3 import restore_checkpoint_v3

    path = tmp_path / "unsupported.pt"
    torch.save({"unsupported": Path("not-a-supported-checkpoint-value")}, path)
    original_load = torch.load
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe)
    with pytest.raises(ValueError, match="weights-only"):
        restore_checkpoint_v3(path)
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps([legacy_factory_witness(Path(directory), variant)
                          for variant in ("full", "no_gate", "no_memory")], indent=2))

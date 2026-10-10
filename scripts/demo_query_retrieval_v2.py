"""Fixed paired two-update gradient fixture; no financial performance evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from eigen_jepa.losses import spectral_jepa_loss
from eigen_jepa.memory import MemoryState, SpectralMemory
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig
from eigen_jepa.query_retrieval_v2 import (
    QueryGradientConfigV2, QueryGradientEigenJEPAV2, QueryGradientMemoryV2,
    load_query_checkpoint_v2, save_query_checkpoint_v2,
)


def tensor_identity(mapping):
    digest = hashlib.sha256()
    for name, tensor in sorted(mapping.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = QueryGradientConfigV2(input_dim=7, num_assets=3, context_len=6, k=2,
                                d_model=8, n_heads=2, n_layers=1, dropout=0.0,
                                latent_dim=8, memory_dim=8)
    old_cfg = EigenJEPAConfig(**{name: value for name, value in asdict(cfg).items()
                                if name not in {"model_schema", "retrieval_policy"}})
    torch.manual_seed(94501)
    old = EigenJEPA(old_cfg, 15)
    torch.manual_seed(94501)
    new = QueryGradientEigenJEPAV2(cfg, 15)
    initial_old, initial_new = tensor_identity(old.state_dict()), tensor_identity(new.state_dict())
    if initial_old != initial_new:
        raise ValueError("Paired model initial parameters differ")
    torch.save({name: value.detach().clone() for name, value in new.state_dict().items()},
               args.output / "shared_initial_model.pt")
    old_memory = SpectralMemory(8, 8, max_items=5, top_k=2, temperature=0.7,
                                merge_radius=0.0, min_salience=0.0)
    new_memory = QueryGradientMemoryV2(8, 8, max_items=5, top_k=2, temperature=0.7,
                                       merge_radius=0.0, min_salience=0.0)
    initial_memory = {"keys": torch.eye(8)[:3].clone(),
                      "values": torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10,
                      "salience": torch.ones(3), "regimes": torch.arange(3)}
    for memory in (old_memory, new_memory):
        memory.state = MemoryState(**{name: value.clone() for name, value in initial_memory.items()})
    inputs = {"x": torch.linspace(-1., 1., 96).reshape(2, 6, 8),
              "target": torch.linspace(-0.5, 0.5, 30).reshape(2, 15),
              "eigenvalues": torch.tensor([[1.2, 0.4], [1.0, 0.3]]),
              "projector": torch.eye(3).repeat(2, 1, 1), "drift": torch.tensor([0.2, 0.8]),
              "regime": torch.tensor([0, 2]), "event": torch.tensor([0., 1.]),
              "risk": torch.tensor([0.3, 0.7]), "entropy": torch.tensor([0.8, 0.6]),
              "rank": torch.tensor([2.0, 1.5])}
    torch.save(inputs, args.output / "shared_generated_inputs.pt")
    torch.save(initial_memory, args.output / "shared_initial_memory.pt")
    (args.output / "model_configs.json").write_text(json.dumps(
        {"legacy": asdict(old_cfg), "query_v2": asdict(cfg)}, indent=2) + "\n")
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (old, new)]
    steps = []
    for step in range(2):
        records, initial_predictions = {}, []
        for arm, model, memory, optimizer in zip(("legacy", "query_v2"), (old, new),
                                                (old_memory, new_memory), optimizers):
            # Pair the target encoder's existing dropout draws within each step.
            torch.manual_seed(94520 + step)
            rng_hash = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
            before_query = tensor_identity(model.query_head.state_dict())
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs["x"], memory)
            target = model.target_latent(inputs["target"])
            loss, _ = spectral_jepa_loss(prediction, target, inputs["eigenvalues"], inputs["projector"],
                                        inputs["drift"], inputs["regime"], inputs["event"], inputs["risk"],
                                        inputs["entropy"], inputs["rank"])
            initial_predictions.append(prediction["z_hat"].detach().clone())
            if not torch.isfinite(loss):
                raise ValueError("Generated spectral loss is nonfinite")
            loss.backward()
            gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                         if parameter.grad is not None}
            if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
                raise ValueError("Generated optimizer gradients are missing/nonfinite")
            torch.save(gradients, args.output / f"{arm}_gradients_step_{step}.pt")
            query_gradients = {name: None if parameter.grad is None else parameter.grad.norm().item()
                               for name, parameter in model.query_head.named_parameters()}
            if arm == "legacy" and any(value is not None for value in query_gradients.values()):
                raise ValueError("Legacy query gradient unexpectedly exists")
            if arm == "query_v2" and any(value is None or value <= 0 for value in query_gradients.values()):
                raise ValueError("Expected nonzero v2 query gradients are absent")
            optimizer.step()
            memory.write(prediction["query"].detach(), target.detach(), torch.ones(2), inputs["regime"])
            records[arm] = {"loss": loss.item(), "paired_rng_sha256": rng_hash,
                            "query_gradient_norms": query_gradients,
                            "query_before_sha256": before_query,
                            "query_after_sha256": tensor_identity(model.query_head.state_dict()),
                            "gradient_tensor_sha256": tensor_identity(gradients),
                            "memory_items": memory.state.size}
        if step == 0 and not torch.equal(*initial_predictions):
            raise ValueError("The paired first forward was not exactly equal")
        if records["legacy"]["query_before_sha256"] != records["legacy"]["query_after_sha256"]:
            raise ValueError("Legacy query weights changed without gradients")
        if records["query_v2"]["query_before_sha256"] == records["query_v2"]["query_after_sha256"]:
            raise ValueError("The v2 optimizer did not update the query weights")
        steps.append({"step": step, "arms": records})
    torch.save({"model": old.state_dict(), "memory": old_memory.state_dict()},
               args.output / "legacy_final_tensors.pt")
    checkpoint_hash = save_query_checkpoint_v2(args.output / "query_v2_checkpoint.pt", new, new_memory)
    restored = load_query_checkpoint_v2(args.output / "query_v2_checkpoint.pt")
    new.eval()
    with torch.no_grad():
        if not torch.equal(new(inputs["x"], new_memory)["z_hat"],
                           restored.model(inputs["x"], restored.memory)["z_hat"]):
            raise ValueError("The saved v2 checkpoint did not reproduce its forward")
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(args.output.iterdir()) if path.is_file()}
    report = {"schema": "eigen_jepa.generated_query_gradient_demo.v2",
              "scope": "Paired algorithmic gradient witness; no forecasting utility or predecessor reclassification",
              "torch": str(torch.__version__), "initialization_seed": 94501,
              "per_step_dropout_seeds": [94520, 94521], "threads": torch.get_num_threads(),
              "examples": 2, "context_length": 6, "latent_dim": 8, "updates_per_arm": 2,
              "shared_initial_model_tensor_sha256": initial_new,
              "shared_initial_memory_tensor_sha256": tensor_identity(initial_memory),
              "shared_input_tensor_sha256": tensor_identity(inputs), "first_forward_exact": True,
              "steps": steps, "v2_checkpoint_sha256": checkpoint_hash,
              "v2_checkpoint_replay_exact": True, "artifact_sha256": artifacts}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

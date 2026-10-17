"""Generated mathematical and actual-model tests for the explicit v2 successor."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from eigen_jepa.losses import spectral_jepa_loss
from eigen_jepa.memory import MemoryState, SpectralMemory
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig
from eigen_jepa.query_retrieval_v2 import (
    CHECKPOINT_SCHEMA, MODEL_SCHEMA, RETRIEVAL_POLICY,
    QueryGradientConfigV2, QueryGradientEigenJEPAV2, QueryGradientMemoryV2,
    load_query_checkpoint_v2, query_config_v2, save_query_checkpoint_v2,
)


def small_memory(*, dtype=torch.float64, top_k=2):
    memory = QueryGradientMemoryV2(2, 1, max_items=4, top_k=top_k,
                                   temperature=0.7, merge_radius=0.0,
                                   min_salience=0.0, dtype=dtype)
    memory.state = MemoryState(torch.eye(2, dtype=dtype),
                               torch.tensor([[2.0], [5.0]], dtype=dtype),
                               torch.ones(2, dtype=dtype), torch.zeros(2, dtype=torch.long))
    return memory


def tiny_config():
    return QueryGradientConfigV2(input_dim=7, num_assets=3, context_len=6, k=2,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0,
                                 latent_dim=8, memory_dim=8)


def model_memory(*, dtype=torch.float32):
    memory = QueryGradientMemoryV2(8, 8, max_items=5, top_k=2, temperature=0.7,
                                   merge_radius=0.0, min_salience=0.0, dtype=dtype)
    memory.state = MemoryState(torch.eye(8, dtype=dtype)[:3].clone(),
                               torch.arange(24, dtype=dtype).reshape(3, 8) / 10,
                               torch.ones(3, dtype=dtype), torch.arange(3))
    return memory


def actual_loss(model, memory):
    x = torch.linspace(-1.0, 1.0, 96).reshape(2, 6, 8)
    prediction = model(x, memory)
    target = model.target_latent(torch.linspace(-0.5, 0.5, 30).reshape(2, 15))
    loss, _ = spectral_jepa_loss(
        prediction, target, torch.tensor([[1.2, 0.4], [1.0, 0.3]]),
        torch.eye(3).repeat(2, 1, 1), torch.tensor([0.2, 0.8]),
        torch.tensor([0, 2]), torch.tensor([0.0, 1.0]),
        torch.tensor([0.3, 0.7]), torch.tensor([0.8, 0.6]), torch.tensor([2.0, 1.5]),
    )
    return loss, prediction


def test_two_neighbor_query_derivative_matches_hand_derived_cosine_softmax():
    memory = small_memory()
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    result, info = memory.retrieve(query)
    result.sum().backward()
    eps, radius = 1e-8, query.detach().norm().item()
    q = query.detach().squeeze(0)
    probs = torch.softmax(q / ((radius + eps) * (1 + eps) * 0.7), dim=0)
    d_difference = (torch.tensor([1.0, -1.0], dtype=q.dtype) / (radius + eps)
                    - (q[0] - q[1]) * q / (radius * (radius + eps) ** 2)) / (1 + eps)
    expected = (2.0 - 5.0) * probs.prod() / 0.7 * d_difference
    torch.testing.assert_close(query.grad[0], expected, rtol=1e-12, atol=1e-14)
    assert query.grad.norm().item() > 0.1
    assert not info["indices"].requires_grad


def test_float64_gradcheck_away_from_neighbor_selection_boundaries():
    memory = small_memory()
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q: memory.retrieve(q)[0], (query,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_selected_topk_forward_is_bit_identical_to_legacy_read():
    memory = small_memory()
    memory.state = MemoryState(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64),
                               torch.tensor([[2.], [5.], [11.]], dtype=torch.float64),
                               torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.long))
    legacy = SpectralMemory(2, 1, max_items=4, top_k=2, temperature=0.7)
    legacy.state = memory.state
    query = torch.tensor([[0.6, 0.8], [-0.8, 0.6]], dtype=torch.float64, requires_grad=True)
    actual, info = memory.retrieve(query)
    old, old_info = legacy.retrieve(query)
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    for name in info:
        torch.testing.assert_close(info[name], old_info[name], rtol=0, atol=0)
    assert actual.requires_grad and not old.requires_grad


def test_stored_content_is_detached_and_write_cannot_change_an_existing_read_graph():
    memory = small_memory()
    keys = memory.state.keys.requires_grad_()
    values = memory.state.values.requires_grad_()
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    control_query = query.detach().clone().requires_grad_()
    expected, _ = memory.retrieve(control_query)
    expected_gradient = torch.autograd.grad(expected.sum(), control_query)[0]
    result, _ = memory.retrieve(query)
    memory.merge_radius = 0.2
    memory.write(torch.tensor([[1., 0.]], dtype=torch.float64),
                 torch.tensor([[100.]], dtype=torch.float64), torch.tensor([2.]), torch.tensor([2]))
    assert memory.state.values[0].item() > 2
    result.sum().backward()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(query.grad, expected_gradient, rtol=0, atol=0)
    assert keys.grad is None and values.grad is None


@pytest.mark.parametrize("limit", ["one_neighbor", "equal_values"])
def test_declared_constant_read_limits_have_zero_query_gradient(limit):
    memory = small_memory(top_k=1 if limit == "one_neighbor" else 2)
    if limit == "equal_values":
        memory.state.values.fill_(5.0)
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    result, _ = memory.retrieve(query)
    result.sum().backward()
    torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=1e-14)


def test_empty_memory_has_no_query_graph_and_clear_retains_explicit_double_dtype():
    memory = small_memory()
    memory.clear()
    query = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    result, info = memory.retrieve(query)
    assert result.shape == (2, 1) and info["weights"].shape == (2, 0)
    assert result.dtype == info["weights"].dtype == torch.float64
    assert not result.requires_grad and torch.count_nonzero(result).item() == 0


def test_ties_keep_runtime_topk_rule_without_claiming_membership_derivative():
    memory = small_memory(top_k=1)
    query = torch.tensor([[1., 1.]], dtype=torch.float64, requires_grad=True)
    _, info = memory.retrieve(query)
    direct = torch.topk(memory._normalize(query) @ memory._normalize(memory.state.keys).T, k=1, dim=-1)
    assert torch.equal(info["indices"], direct.indices)
    lower, _ = memory.retrieve(torch.tensor([[1. - 1e-6, 1.]], dtype=torch.float64))
    upper, _ = memory.retrieve(torch.tensor([[1. + 1e-6, 1.]], dtype=torch.float64))
    assert lower.item() == 5.0 and upper.item() == 2.0


@pytest.mark.parametrize("case", ["query_shape", "query_nan", "query_integer", "query_float32",
                                    "query_meta", "key_nan", "value_inf", "value_shape", "regime",
                                    "negative_salience", "top_k", "temperature", "norm_overflow"])
def test_invalid_query_or_bank_is_refused_without_mutation(case):
    memory = small_memory()
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    if case == "query_shape": query = query[:, :1]
    elif case == "query_nan": query[0, 1] = float("nan")
    elif case == "query_integer": query = query.long()
    elif case == "query_float32": query = query.float()
    elif case == "query_meta": query = query.to("meta")
    elif case == "key_nan": memory.state.keys[0, 0] = float("nan")
    elif case == "value_inf": memory.state.values[0, 0] = float("inf")
    elif case == "value_shape": memory.state.values = torch.zeros(2, 2, dtype=torch.float64)
    elif case == "regime": memory.state.regimes[0] = 3
    elif case == "negative_salience": memory.state.salience[0] = -1
    elif case == "top_k": memory.top_k = 0
    elif case == "temperature": memory.temperature = float("nan")
    elif case == "norm_overflow": query.fill_(1e300)
    before = {name: getattr(memory.state, name).clone() for name in ("keys", "values", "salience", "regimes")}
    with pytest.raises(ValueError):
        memory.retrieve(query)
    for name, original in before.items():
        torch.testing.assert_close(getattr(memory.state, name), original, rtol=0, atol=0, equal_nan=True)


def test_model_initialization_outputs_and_query_gradients_distinguish_versions():
    config = tiny_config()
    legacy_config = EigenJEPAConfig(**{name: value for name, value in asdict(config).items()
                                      if name not in {"model_schema", "retrieval_policy"}})
    torch.manual_seed(94501)
    old = EigenJEPA(legacy_config, 15).eval()
    torch.manual_seed(94501)
    new = QueryGradientEigenJEPAV2(config, 15).eval()
    assert set(old.state_dict()) == set(new.state_dict())
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name])
    memory = model_memory()
    old_memory = SpectralMemory(8, 8, max_items=5, top_k=2, temperature=0.7)
    old_memory.state = memory.state
    old_loss, old_output = actual_loss(old, old_memory)
    new_loss, new_output = actual_loss(new, memory)
    for name in old_output.keys() - {"mem_info"}:
        torch.testing.assert_close(new_output[name], old_output[name], rtol=0, atol=0)
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    old_loss.backward()
    new_loss.backward()
    assert all(parameter.grad is None for parameter in old.query_head.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.norm() > 0 for parameter in new.query_head.parameters())
    old_query = {name: tensor.clone() for name, tensor in new.query_head.state_dict().items()}
    torch.optim.SGD(new.parameters(), lr=0.01).step()
    assert any(not torch.equal(old_query[name], tensor) for name, tensor in new.query_head.state_dict().items())


def test_v2_model_requires_explicit_config_and_memory_identity():
    config = tiny_config()
    with pytest.raises(ValueError, match="explicit"):
        QueryGradientEigenJEPAV2(EigenJEPAConfig(7, 3, 6, k=2), 15)
    model = QueryGradientEigenJEPAV2(config, 15)
    with pytest.raises(ValueError, match="v2 memory"):
        model(torch.ones(2, 6, 8), SpectralMemory(8, 8))
    assert model(torch.ones(2, 6, 8), None)["mem_info"] is None
    with pytest.raises(ValueError):
        query_config_v2({name: value for name, value in asdict(config).items() if name != "model_schema"})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_records_config_policy_dtype_and_owned_tensors_and_roundtrips(tmp_path, dtype):
    model = QueryGradientEigenJEPAV2(tiny_config(), 15).to(dtype=dtype).eval()
    memory = model_memory(dtype=dtype)
    memory.temperature, memory.merge_radius, memory.min_salience = 0.37, 0.12, 0.04
    path = tmp_path / "query-v2.pt"
    rng = torch.random.get_rng_state().clone()
    digest = save_query_checkpoint_v2(path, model, memory)
    assert torch.equal(rng, torch.random.get_rng_state())
    payload = torch.load(path, weights_only=True)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["model_schema"] == payload["model_config"]["model_schema"] == MODEL_SCHEMA
    assert payload["retrieval_policy"] == payload["model_config"]["retrieval_policy"] == RETRIEVAL_POLICY
    assert json.loads(json.dumps(payload["model_config"])) == asdict(tiny_config())
    assert payload["memory_config"]["temperature"] == 0.37
    assert payload["memory_config"]["merge_radius"] == 0.12
    assert payload["memory_config"]["min_salience"] == 0.04
    restored = load_query_checkpoint_v2(path)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert restored.checkpoint_sha256 == digest == hashlib.sha256(path.read_bytes()).hexdigest()
    x = torch.linspace(-1.0, 1.0, 96, dtype=dtype).reshape(2, 6, 8)
    torch.testing.assert_close(restored.model(x, restored.memory)["z_hat"], model(x, memory)["z_hat"], rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.model.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        next(model.parameters()).add_(1)
        memory.state.values.add_(5)
    assert not torch.equal(memory.state.values, restored.memory.state.values)
    with pytest.raises(FileExistsError):
        save_query_checkpoint_v2(path, model, memory)


@pytest.mark.parametrize("case", ["legacy", "schema", "config_identity", "retrieval_identity",
                                    "memory_identity", "policy_missing", "wrong_dtype", "state_shape",
                                    "state_nan", "memory_dtype", "memory_dimensions", "invalid_regime"])
def test_checkpoint_loader_refuses_incompatible_identity_or_tensor_state(tmp_path, case):
    path = tmp_path / "query-v2.pt"
    save_query_checkpoint_v2(path, QueryGradientEigenJEPAV2(tiny_config(), 15), model_memory())
    payload = torch.load(path, weights_only=True)
    if case == "legacy": payload = {"cfg": asdict(EigenJEPAConfig(7, 3, 6)), "model": payload["model_state"]}
    elif case == "schema": payload["schema"] = "eigen_jepa.legacy"
    elif case == "config_identity": payload["model_config"]["model_schema"] = "legacy"
    elif case == "retrieval_identity": payload["retrieval_policy"] = "no_grad"
    elif case == "memory_identity": payload["memory_config"]["retrieval_policy"] = "no_grad"
    elif case == "policy_missing": del payload["memory_config"]["temperature"]
    elif case == "wrong_dtype": payload["model_dtype"] = "float64"
    elif case == "state_shape": payload["model_state"]["context.pos"] = torch.zeros(1)
    elif case == "state_nan": payload["model_state"]["context.pos"].fill_(float("nan"))
    elif case == "memory_dtype": payload["memory_state"]["keys"] = payload["memory_state"]["keys"].double()
    elif case == "memory_dimensions": payload["memory_config"]["key_dim"] = 3
    elif case == "invalid_regime": payload["memory_state"]["regimes"][0] = 3
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError):
        load_query_checkpoint_v2(bad)

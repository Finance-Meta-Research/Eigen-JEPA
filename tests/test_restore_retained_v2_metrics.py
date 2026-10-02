import hashlib
import importlib.util
from pathlib import Path
import tempfile
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("restore_v2", Path(__file__).resolve().parents[1] / "scripts/restore_retained_v2_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_bundle(root, monkeypatch, missing=False, duplicate=False):
    archive = root / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in module.METRIC_PATHS[1:] if missing else module.METRIC_PATHS:
            z.writestr(name, '{"fixture": true}')
        z.writestr("../unselected.txt", "must never extract")
        if duplicate:
            z.writestr("metrics.json", "duplicate")
    # Only this unit fixture patches the expected digest; production stays pinned.
    monkeypatch.setattr(module, "EXPECTED_DIGEST", hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive


def test_restores_only_named_members_and_is_idempotent(tmp_path, monkeypatch):
    archive = fixture_bundle(tmp_path, monkeypatch)
    target = tmp_path / "out"
    assert module.restore(archive, target) == 21
    assert module.restore(archive, target) == 21
    assert len(list(target.rglob("*.json"))) == 21
    assert not (tmp_path / "unselected.txt").exists()


def test_digest_mismatch_writes_nothing(tmp_path, monkeypatch):
    archive = fixture_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "EXPECTED_DIGEST", "0" * 64)
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.restore(archive, target)
    assert not target.exists()


def test_conflicting_late_file_prevents_all_writes(tmp_path, monkeypatch):
    archive = fixture_bundle(tmp_path, monkeypatch)
    target = tmp_path / "out"
    existing = target / module.METRIC_PATHS[-1]
    existing.parent.mkdir(parents=True)
    existing.write_text("keep")
    with pytest.raises(ValueError, match="differs"):
        module.restore(archive, target)
    assert existing.read_text() == "keep"
    assert not (target / "metrics.json").exists()


@pytest.mark.parametrize("kwargs", [{"missing": True}, {"duplicate": True}])
def test_missing_or_duplicate_member_fails(tmp_path, monkeypatch, kwargs):
    archive = fixture_bundle(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        module.restore(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_symlink_destination_rejected(tmp_path, monkeypatch):
    archive = fixture_bundle(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    target = tmp_path / "out"
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.restore(archive, target)
    assert not list(external.iterdir())

from argparse import Namespace

import pytest

from eigen_jepa.benchmark import _resolve_seed_list


def test_explicit_seed_list_preserves_frozen_order():
    args = Namespace(seeds=[7, 19, 31, 43, 59], seed=7, num_seeds=5, seed_stride=999)
    assert _resolve_seed_list(args) == [7, 19, 31, 43, 59]


def test_generated_seed_list_remains_backward_compatible():
    args = Namespace(seeds=None, seed=7, num_seeds=3, seed_stride=11)
    assert _resolve_seed_list(args) == [7, 18, 29]


def test_explicit_seed_list_rejects_duplicates():
    args = Namespace(seeds=[7, 19, 19], seed=7, num_seeds=3, seed_stride=11)
    with pytest.raises(ValueError, match='unique seeds'):
        _resolve_seed_list(args)

from pathlib import Path

from scripts.build_final_rigor_v2_paired_table import build_rows, load_source, render_tex


TABLE_PATH = Path("paper/final_rigor_v2_paired_table.tex")
MANUSCRIPT_PATH = Path("paper/conference_v2.tex")


def _by_key():
    return {(row["comparison"], row["metric"]): row for row in build_rows(load_source())}


def test_committed_paired_table_is_generated_from_retained_seed_evidence():
    rows = build_rows(load_source())
    assert len(rows) == 15
    assert render_tex(rows) == TABLE_PATH.read_text(encoding="utf-8")


def test_paired_seed_counts_preserve_null_and_directional_disagreement():
    rows = _by_key()

    memory_eig = rows[("no_memory", "eig_nmse")]
    assert (memory_eig["full_better_seeds"], memory_eig["ablation_better_seeds"], memory_eig["ties"]) == (4, 1, 0)
    assert memory_eig["mean_full_minus_ablation"] < 0.0

    memory_gate = rows[("no_memory", "gate_cal")]
    assert (memory_gate["full_better_seeds"], memory_gate["ablation_better_seeds"], memory_gate["ties"]) == (5, 0, 0)

    memory_tail = rows[("no_memory", "tail_f1")]
    assert (memory_tail["full_better_seeds"], memory_tail["ablation_better_seeds"], memory_tail["ties"]) == (0, 0, 5)

    no_regime_eig = rows[("no_regime", "eig_nmse")]
    assert no_regime_eig["mean_full_minus_ablation"] > 0.0
    assert (no_regime_eig["full_better_seeds"], no_regime_eig["ablation_better_seeds"]) == (3, 2)

    no_gate_cal = rows[("no_gate", "gate_cal")]
    assert no_gate_cal["mean_full_minus_ablation"] < 0.0
    assert (no_gate_cal["full_better_seeds"], no_gate_cal["ablation_better_seeds"]) == (2, 3)


def test_manuscript_keeps_paired_table_descriptive_only():
    text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert "final_rigor_v2_paired_table.tex" in text
    assert "Paired seed signs add an important qualification" in text
    assert "not an inferential test" in text
    assert "no preregistered significance test" in text

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def test_main_tex_delegates_to_evidence_bound_conference_manuscript():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "\\input{conference_v2.tex}" in main
    assert "\\documentclass" not in main
    assert "equity, crypto, and rates" not in main


def test_conference_manuscript_retains_frozen_v2_claim_boundary():
    text = (PAPER / "conference_v2.tex").read_text(encoding="utf-8")

    required_fragments = (
        "seeds $\\{7,19,31,43,59\\}$",
        "all 20 seed-by-variant runs",
        "Tail F1 was exactly identical",
        "did not preregister a significance test or practical-effect threshold",
        "synthetic and does not claim trading alpha or real-market validation",
        "33988159305",
        "9975833698",
        "5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4",
    )
    for fragment in required_fragments:
        assert fragment in text

    forbidden_fragments = (
        "We train and evaluate the system on a chronological regime-switching benchmark spanning equity, crypto, and rates",
        "demonstrates trading alpha",
        "statistically significant improvement",
    )
    for fragment in forbidden_fragments:
        assert fragment not in text

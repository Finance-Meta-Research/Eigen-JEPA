from pathlib import Path


PAPER = Path("paper/conference_v2.tex")


def test_related_work_is_present_and_bounded():
    text = PAPER.read_text(encoding="utf-8")
    assert "\\section{Related work and scope}" in text
    assert "scientific context, not experimental baselines" in text
    assert "does not establish superiority over DCC" in text
    assert "does not include matched DCC, shrinkage, or random-matrix forecasting baselines" in text


def test_verified_citation_metadata_replaces_mistitled_pafka_entry():
    text = PAPER.read_text(encoding="utf-8")
    assert "Estimated Correlation Matrices and Financial Markets" not in text
    assert "Estimated Correlation Matrices and Portfolio Optimization" in text
    assert "doi:10.1016/j.physa.2004.05.079" in text
    assert "doi:10.1198/073500102288618487" in text
    assert "doi:10.1016/S0927-5398(03)00007-0" in text
    assert "doi:10.1016/S0047-259X(03)00096-4" in text
    assert "doi:10.1109/CVPR52729.2023.01499" in text

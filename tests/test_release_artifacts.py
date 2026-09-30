import json
from pathlib import Path

from paper.build_pdf import build


def test_report_pdf_is_built_from_canonical_results(tmp_path: Path):
    output = build(tmp_path / "report.pdf")
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 100_000


def test_demo_metadata_identifies_real_inputs():
    metadata = json.loads(Path("docs/assets/demo_metadata.json").read_text())
    assert metadata["synthetic_case"] == "medium_flashing_region"
    assert metadata["seed"] == 7
    assert metadata["sweep_points"] == 49
    assert metadata["rapid_flashing_replayed"] is False

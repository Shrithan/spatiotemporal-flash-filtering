from pathlib import Path

from experiments.generate_v1_figures import architecture_figure


def test_architecture_figure_writes_raster_and_vector_outputs(tmp_path: Path):
    outputs=architecture_figure(tmp_path)
    assert {path.suffix for path in outputs}=={".png",".svg"}
    assert all(path.stat().st_size>0 for path in outputs)

from pathlib import Path

from flashfilter import AdaptiveFilterConfig, AnalysisReport, FilterReport, analyze_video, filter_video
from flashfilter.synthetic import generate_cases
from flashfilter.video import write_video


def test_public_analysis_api_returns_structured_nonclinical_report(tmp_path: Path):
    case = generate_cases(length=8, height=16, width=16)["small_flashing_square"]
    source = tmp_path / "input.npz"
    write_video(source, case.frames, case.fps)
    report = analyze_video(source)
    assert isinstance(report, AnalysisReport)
    assert report.frame_count == 8
    assert (report.width, report.height) == (16, 16)
    assert report.temporal["peak_activity"] > 0
    assert report.medical_safety_claimed is False
    assert report.to_dict()["evidence"]["standards_compliance_claimed"] is False


def test_public_filter_api_preserves_lossless_dimensions_and_fps(tmp_path: Path):
    case = generate_cases(length=8, height=16, width=16)["small_flashing_square"]
    source, output = tmp_path / "input.npz", tmp_path / "output.npz"
    write_video(source, case.frames, case.fps)
    report = filter_video(
        source, output,
        adaptive_config=AdaptiveFilterConfig(max_luminance_step=.05),
    )
    assert isinstance(report, FilterReport)
    assert report.frames_processed == 8
    assert report.resolution_preserved and report.fps_preserved
    assert report.output_analysis is not None
    assert report.medical_safety_claimed is False

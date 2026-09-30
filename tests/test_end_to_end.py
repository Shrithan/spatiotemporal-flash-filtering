from pathlib import Path
import pandas as pd
from flashfilter.benchmark import evaluate
from flashfilter.cli import main
from flashfilter.synthetic import generate_cases
from flashfilter.video import read_video, write_video
from flashfilter.streaming import analyze_video_stream, filter_video_stream, inspect_video_stream


def test_missing_video_has_actionable_error(tmp_path: Path):
    missing = tmp_path / "not_created.mp4"
    try:
        read_video(missing)
    except FileNotFoundError as error:
        assert "does not exist" in str(error)
        assert "before analysis" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")

def test_tiny_video_round_trip_and_filter(tmp_path: Path):
    case=generate_cases(length=8,height=16,width=16)["small_flashing_square"]
    source=tmp_path/"input.npz"; output=tmp_path/"output.npz"; write_video(source,case.frames,case.fps)
    main(["filter",str(source),"--output",str(output),"--block-size","4","--temporal-window","1"])
    decoded=read_video(output); assert decoded.frames.shape==case.frames.shape and decoded.fps==case.fps

def test_tiny_benchmark_writes_structured_results(tmp_path: Path):
    table=evaluate(tmp_path,length=8,height=16,width=16)
    assert len(table)==72
    assert {"none","global","localized","adaptive","enhanced_exploratory","framegen_exploratory"}==set(table.method)
    assert (tmp_path/"benchmark.csv").exists() and len(pd.read_csv(tmp_path/"summary.csv"))==6


def test_streaming_video_pipeline(tmp_path: Path):
    case=generate_cases(length=8,height=16,width=16)["small_flashing_square"]
    source=tmp_path/"stream_input.mp4"; output=tmp_path/"stream_output.mp4"; write_video(source,case.frames,case.fps)
    count=filter_video_stream(source,output,"localized")
    metrics=analyze_video_stream(output)
    assert count==8 and output.stat().st_size>0 and metrics["peak_activity"]>=0


def test_streaming_multianalyzer_pipeline(tmp_path: Path):
    case=generate_cases(length=8,height=32,width=32)["medium_flashing_region"]
    source=tmp_path/"multi_input.mp4"; output=tmp_path/"multi_output.mp4"; write_video(source,case.frames,case.fps)
    count=filter_video_stream(source,output,"multianalyzer")
    report=inspect_video_stream(source)
    assert count==8 and output.stat().st_size>0
    assert report["luminance_flash_events"]>0 and report["standards_compliance_claimed"] is False

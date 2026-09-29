"""Reproducible evaluation shared by scripts and CLI."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from .filtering import global_filter, localized_filter, no_filter
from .localization import LocalizationConfig
from .metrics import distortion_metrics, localization_metrics, modified_area_ratio, residual_inside_regions, temporal_metrics
from .synthetic import generate_cases, generate_analyzer_cases
from .analyzers import AnalyzerConfig, analyze_frames
from .enhanced import EnhancedConfig, enhanced_filter
from .framegen import FrameGenerationConfig, framegen_filter

DEFAULT_CONFIG=LocalizationConfig(threshold=.12,block_size=8,temporal_window=2,cleanup_kernel=3,feather_sigma=2.)

def evaluate(output_dir: str|Path, *, length: int=24, height: int=64, width: int=96, seed: int=7, config: LocalizationConfig=DEFAULT_CONFIG) -> pd.DataFrame:
    """Evaluate all methods on identical deterministic inputs and save CSV/JSON."""
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True); rows=[]; traces=[]
    for case in generate_cases(length,height,width,seed).values():
        methods={"none":lambda:no_filter(case.frames),"global":lambda:global_filter(case.frames,config.threshold),"localized":lambda:localized_filter(case.frames,config),"enhanced":lambda:enhanced_filter(case.frames,EnhancedConfig()),"framegen":lambda:framegen_filter(case.frames,FrameGenerationConfig())}
        for method,run in methods.items():
            start=perf_counter(); result=run(); elapsed=perf_counter()-start
            temporal=temporal_metrics(result.frames,config.threshold,config.threshold); distortion=distortion_metrics(case.frames,result.frames); localization=localization_metrics(result.masks,case.masks)
            row={"scenario":case.name,"category":case.category,"method":method,**temporal,**distortion,**localization,"modified_area_ratio":modified_area_ratio(case.frames,result.frames),"residual_inside_regions":residual_inside_regions(case.frames,result.frames,case.masks),"seconds":elapsed,"fps":length/elapsed if elapsed else np.inf}
            rows.append(row)
            from .luminance import brightness
            from .temporal import global_change_scores
            traces.extend({"scenario":case.name,"method":method,"frame":i,"activity":float(value)} for i,value in enumerate(global_change_scores(brightness(result.frames))))
    table=pd.DataFrame(rows); table.to_csv(destination/"benchmark.csv",index=False); pd.DataFrame(traces).to_csv(destination/"temporal_traces.csv",index=False)
    summary=table.groupby("method",as_index=False).mean(numeric_only=True); summary.to_csv(destination/"summary.csv",index=False)
    (destination/"config.json").write_text(json.dumps({"seed":seed,"length":length,"height":height,"width":width,"localization":config.__dict__},indent=2)+"\n")
    return table

def run_ablations(output_dir: str|Path, *, length: int=24, height: int=64, width: int=96, seed: int=7) -> pd.DataFrame:
    """Measure cleanup, mask softness, block size, threshold, and time-window choices."""
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True); cases=generate_cases(length,height,width,seed); rows=[]
    variants=[("no_cleanup",LocalizationConfig(.12,8,2,3,2),False,True),("cleanup",LocalizationConfig(.12,8,2,3,2),True,True),("hard_mask",LocalizationConfig(.12,8,2,3,0),True,False)]
    variants += [(f"block_{b}",LocalizationConfig(.12,b,2,3,2),True,True) for b in (4,16)]
    variants += [(f"threshold_{v}",LocalizationConfig(v,8,2,3,2),True,True) for v in (.08,.18)]
    variants += [(f"window_{w}",LocalizationConfig(.12,8,w,3,2),True,True) for w in (1,4)]
    for scenario,case in cases.items():
        for name,config,cleanup,soft in variants:
            result=localized_filter(case.frames,config,cleanup=cleanup,soft=soft); row={"scenario":scenario,"category":case.category,"variant":name,**temporal_metrics(result.frames),**distortion_metrics(case.frames,result.frames),**localization_metrics(result.masks,case.masks),"modified_area_ratio":modified_area_ratio(case.frames,result.frames)}; rows.append(row)
    table=pd.DataFrame(rows); table.to_csv(destination/"ablations.csv",index=False); table.groupby("variant",as_index=False).mean(numeric_only=True).to_csv(destination/"ablation_summary.csv",index=False); return table


def evaluate_analyzers(output_dir: str | Path) -> pd.DataFrame:
    """Evaluate channel-level decisions on deterministic boundary cases."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = AnalyzerConfig(min_area_ratio=.20, pattern_tile_size=64)
    rows = []
    for case in generate_analyzer_cases().values():
        result = analyze_frames(case.frames, case.fps, config)
        predicted = {
            "luminance": bool(result.summary["luminance_limit_exceeded"]),
            "red": bool(result.summary["red_limit_exceeded"]),
            "pattern": bool(result.summary["persistent_pattern_detected"]),
        }
        expected = {
            "luminance": case.expect_luminance,
            "red": case.expect_red,
            "pattern": case.expect_pattern,
        }
        rows.append({
            "scenario": case.name,
            "description": case.description,
            **{f"expected_{key}": value for key, value in expected.items()},
            **{f"predicted_{key}": value for key, value in predicted.items()},
            "all_channels_correct": predicted == expected,
            **result.summary,
        })
    table = pd.DataFrame(rows)
    table.to_csv(destination / "analyzer_benchmark.csv", index=False)
    summary = {
        "scenarios": int(len(table)),
        "exact_match_accuracy": float(table["all_channels_correct"].mean()),
        "standards_compliance_claimed": False,
    }
    (destination / "analyzer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return table

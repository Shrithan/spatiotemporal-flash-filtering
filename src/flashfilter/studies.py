"""Deterministic v1.0 parameter and component studies.

The objective values are computational temporal-activity and image-distortion
metrics. They are not estimates of clinical risk or standards conformance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyzers import AnalyzerConfig
from .filtering import (
    AdaptiveFilterConfig,
    FilterResult,
    adaptive_event_filter,
    global_filter,
    localized_filter,
    no_filter,
)
from .localization import LocalizationConfig
from .metrics import distortion_metrics, localization_metrics, modified_area_ratio, temporal_metrics
from .synthetic import SyntheticCase, generate_cases


@dataclass(frozen=True)
class SweepDefinition:
    """Compact, scientifically interpretable parameter grid."""

    global_thresholds: tuple[float, ...] = (.04, .08, .12, .18)
    global_blends: tuple[float, ...] = (.35, .65, .85)
    localized_thresholds: tuple[float, ...] = (.06, .12, .18)
    localized_blends: tuple[float, ...] = (.35, .65, .85)
    localized_block_sizes: tuple[int, ...] = (4, 8)
    adaptive_luminance_thresholds: tuple[float, ...] = (.05, .10, .15)
    adaptive_max_luminance_steps: tuple[float, ...] = (.03, .08, .15)
    adaptive_profiles: tuple[str, ...] = ("moderate", "strong")


def pareto_efficient(values: np.ndarray) -> np.ndarray:
    """Return points not dominated when every supplied objective is minimized."""
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("values must have shape (N, K) with K >= 2")
    efficient = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominated[index] = False
        efficient[index] = not np.any(dominated)
    return efficient


def _measure(case: SyntheticCase, method: str, config_id: str, result: FilterResult, seconds: float, parameters: dict) -> dict:
    return {
        "scenario": case.name,
        "category": case.category,
        "method": method,
        "config_id": config_id,
        **parameters,
        **temporal_metrics(result.frames),
        **distortion_metrics(case.frames, result.frames),
        **localization_metrics(result.masks, case.masks),
        "modified_area_ratio": modified_area_ratio(case.frames, result.frames),
        "seconds": seconds,
        "fps": len(case.frames) / seconds if seconds else np.inf,
    }


def _sweep_configurations(definition: SweepDefinition) -> list[tuple[str, str, dict, Callable[[SyntheticCase], FilterResult]]]:
    configurations: list[tuple[str, str, dict, Callable[[SyntheticCase], FilterResult]]] = [
        ("none", "none", {}, lambda case: no_filter(case.frames))
    ]
    for threshold in definition.global_thresholds:
        for blend in definition.global_blends:
            config_id=f"global_t{threshold:.2f}_a{blend:.2f}"
            parameters={"threshold":threshold,"blend":blend}
            configurations.append(("global",config_id,parameters,lambda case,t=threshold,a=blend:global_filter(case.frames,t,a)))
    for threshold in definition.localized_thresholds:
        for blend in definition.localized_blends:
            for block_size in definition.localized_block_sizes:
                config_id=f"localized_t{threshold:.2f}_a{blend:.2f}_b{block_size}"
                parameters={"threshold":threshold,"blend":blend,"block_size":block_size}
                config=LocalizationConfig(threshold,block_size,2,3,2.)
                configurations.append(("localized",config_id,parameters,lambda case,c=config,a=blend:localized_filter(case.frames,c,a)))
    profiles={
        "moderate": {"desaturation":.5,"pattern_contrast_reduction":.25},
        "strong": {"desaturation":1.,"pattern_contrast_reduction":.5},
    }
    for threshold in definition.adaptive_luminance_thresholds:
        for step in definition.adaptive_max_luminance_steps:
            for profile in definition.adaptive_profiles:
                strengths=profiles[profile]
                config_id=f"adaptive_t{threshold:.2f}_s{step:.2f}_{profile}"
                parameters={"threshold":threshold,"max_luminance_step":step,"profile":profile,**strengths}
                analyzer=AnalyzerConfig(luminance_transition=threshold)
                correction=AdaptiveFilterConfig(max_luminance_step=step,**strengths)
                configurations.append(("adaptive",config_id,parameters,lambda case,a=analyzer,c=correction:adaptive_event_filter(case.frames,case.fps,analyzer_config=a,filter_config=c)))
    return configurations


def run_parameter_sweep(
    output_dir: str | Path,
    *,
    length: int = 24,
    height: int = 64,
    width: int = 96,
    seed: int = 7,
    definition: SweepDefinition = SweepDefinition(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the suppression-distortion study and write data and Pareto plot."""
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    cases=generate_cases(length,height,width,seed); rows=[]
    for method,config_id,parameters,run in _sweep_configurations(definition):
        for case in cases.values():
            start=perf_counter(); result=run(case); seconds=perf_counter()-start
            rows.append(_measure(case,method,config_id,result,seconds,parameters))
    detail=pd.DataFrame(rows); detail.to_csv(destination/"parameter_sweep.csv",index=False)
    summary=detail.groupby(["method","config_id"],as_index=False).mean(numeric_only=True)
    summary["pareto_efficient"]=pareto_efficient(summary[["mae","mean_activity"]].to_numpy())
    summary.to_csv(destination/"parameter_sweep_summary.csv",index=False)
    (destination/"sweep_definition.json").write_text(json.dumps({"seed":seed,"length":length,"height":height,"width":width,"grid":asdict(definition),"objectives":{"distortion":"mae","residual_activity":"mean_activity"},"medical_risk_function":False},indent=2)+"\n")

    fig,axis=plt.subplots(figsize=(7.2,4.8))
    colors={"none":"#6b7280","global":"#d97706","localized":"#2563eb","adaptive":"#059669"}
    for method,group in summary.groupby("method"):
        axis.scatter(group.mae,group.mean_activity,label=method,color=colors[method],alpha=.72,s=34)
    frontier=summary[summary.pareto_efficient].sort_values("mae")
    axis.plot(frontier.mae,frontier.mean_activity,color="#111827",linewidth=1.4,marker="o",markersize=3,label="non-dominated frontier")
    axis.set(xlabel="Mean absolute error from source (lower is better)",ylabel="Residual mean temporal activity (lower is better)",title="Suppression-distortion parameter study")
    axis.grid(alpha=.25); axis.legend(ncol=2); fig.tight_layout(); fig.savefig(destination/"suppression_distortion_pareto.png",dpi=220); plt.close(fig)
    return detail,summary


ABLATION_CHANNELS: tuple[tuple[str, bool, bool, bool], ...] = (
    ("luminance_only", True, False, False),
    ("red_only", False, True, False),
    ("pattern_only", False, False, True),
    ("luminance_red", True, True, False),
    ("luminance_pattern", True, False, True),
    ("red_pattern", False, True, True),
    ("all_channels", True, True, True),
)


def run_adaptive_ablation(
    output_dir: str | Path,
    *,
    length: int = 24,
    height: int = 64,
    width: int = 96,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure contributions of all meaningful adaptive channel subsets."""
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in generate_cases(length,height,width,seed).values():
        for name,enable_luminance,enable_red,enable_pattern in ABLATION_CHANNELS:
            config=AdaptiveFilterConfig(enable_luminance=enable_luminance,enable_red=enable_red,enable_pattern=enable_pattern)
            start=perf_counter(); result=adaptive_event_filter(case.frames,case.fps,filter_config=config); seconds=perf_counter()-start
            rows.append(_measure(case,"adaptive",name,result,seconds,{"enable_luminance":enable_luminance,"enable_red":enable_red,"enable_pattern":enable_pattern}))
    detail=pd.DataFrame(rows); detail.to_csv(destination/"adaptive_ablation.csv",index=False)
    summary=detail.groupby("config_id",as_index=False).mean(numeric_only=True); summary.to_csv(destination/"adaptive_ablation_summary.csv",index=False)
    return detail,summary

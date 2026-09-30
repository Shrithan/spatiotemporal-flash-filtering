"""Command-line interface for analysis, filtering, and reproducible experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .benchmark import evaluate, evaluate_analyzers, run_ablations
from .filtering import AdaptiveFilterConfig, adaptive_event_filter, global_filter, localized_filter, multianalyzer_filter
from .localization import LocalizationConfig
from .metrics import temporal_metrics
from .streaming import analyze_video_stream, filter_video_stream, inspect_video_stream
from .synthetic import generate_cases
from .video import read_video, write_video

def _config(args: argparse.Namespace) -> LocalizationConfig:
    return LocalizationConfig(args.threshold,args.block_size,args.temporal_window,args.cleanup_kernel,args.feather_sigma)

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="flashfilter",description="Measure and reduce computational temporal-activity proxies in video.")
    sub=parser.add_subparsers(dest="command",required=True)
    analyze=sub.add_parser("analyze",help="print temporal-activity metrics for a video"); analyze.add_argument("input")
    inspect=sub.add_parser("inspect",help="report separate luminance-flash, red-flash, and regular-pattern evidence"); inspect.add_argument("input")
    filtering=sub.add_parser("filter",help="filter video with a selected research method"); filtering.add_argument("input"); filtering.add_argument("--method",choices=("global","localized","enhanced","framegen","multianalyzer","adaptive"),default="localized"); filtering.add_argument("--output",required=True); filtering.add_argument("--blend",type=float,default=.65)
    filtering.add_argument("--desaturation",type=float,default=.90,help="localized red-event desaturation strength [0,1]")
    filtering.add_argument("--max-luminance-step",type=float,default=.08,help="largest allowed relative-luminance step in detected general flashes")
    filtering.add_argument("--pattern-contrast-reduction",type=float,default=.35,help="localized regular-pattern contrast reduction [0,1]")
    generate=sub.add_parser("generate-synthetic",help="write deterministic synthetic videos and ground truth"); generate.add_argument("--output-dir",default="experiments/generated"); generate.add_argument("--format",choices=("npz","mp4"),default="npz"); generate.add_argument("--seed",type=int,default=7)
    benchmark=sub.add_parser("benchmark",help="run all methods and generate metrics and plots"); benchmark.add_argument("--output-dir",default="experiments/results"); benchmark.add_argument("--seed",type=int,default=7)
    analyzer_benchmark=sub.add_parser("benchmark-analyzers",help="run deterministic luminance/red/pattern boundary cases"); analyzer_benchmark.add_argument("--output-dir",default="experiments/results/analyzers")
    ablate=sub.add_parser("ablate",help="run localized-method component ablations"); ablate.add_argument("--output-dir",default="experiments/results"); ablate.add_argument("--seed",type=int,default=7)
    sweep=sub.add_parser("sweep",help="run the deterministic suppression-distortion parameter study"); sweep.add_argument("--output-dir",default="experiments/results/sweep"); sweep.add_argument("--seed",type=int,default=7)
    adaptive_ablation=sub.add_parser("ablate-adaptive",help="ablate adaptive luminance, red, and pattern channels"); adaptive_ablation.add_argument("--output-dir",default="experiments/results/adaptive_ablation"); adaptive_ablation.add_argument("--seed",type=int,default=7)
    for child in (filtering,benchmark):
        child.add_argument("--threshold",type=float,default=.12,help="normalized activity decision threshold")
        child.add_argument("--block-size",type=int,default=8,help="spatial block side length in pixels")
        child.add_argument("--temporal-window",type=int,default=2,help="number of recent difference maps to average")
        child.add_argument("--cleanup-kernel",type=int,default=3,help="morphological closing kernel side length")
        child.add_argument("--feather-sigma",type=float,default=2.,help="Gaussian mask feathering sigma")
    return parser

def main(argv: list[str]|None=None) -> None:
    args=build_parser().parse_args(argv)
    if args.command=="analyze":
        if Path(args.input).suffix.lower()==".npz":
            report=temporal_metrics(read_video(args.input).frames)
        else:
            report=analyze_video_stream(args.input)
        print(json.dumps(report,indent=2)); return
    if args.command=="inspect":
        if Path(args.input).suffix.lower()==".npz":
            from .analyzers import analyze_frames
            video=read_video(args.input); report=analyze_frames(video.frames,video.fps).summary
        else: report=inspect_video_stream(args.input)
        print(json.dumps(report,indent=2))
        return
    if args.command=="filter":
        adaptive_config=AdaptiveFilterConfig(args.desaturation,args.max_luminance_step,args.pattern_contrast_reduction,args.feather_sigma)
        if Path(args.input).suffix.lower() == ".npz":
            video=read_video(args.input)
            if args.method=="global": result=global_filter(video.frames,args.threshold,args.blend)
            elif args.method=="localized": result=localized_filter(video.frames,_config(args),args.blend)
            elif args.method=="enhanced":
                from .enhanced import EnhancedConfig, enhanced_filter
                result=enhanced_filter(video.frames,EnhancedConfig(transition_threshold=args.threshold))
            elif args.method=="framegen":
                from .framegen import FrameGenerationConfig, framegen_filter
                result=framegen_filter(video.frames,FrameGenerationConfig(transition_threshold=args.threshold))
            elif args.method=="multianalyzer": result=multianalyzer_filter(video.frames,video.fps,blend=args.blend)
            else: result=adaptive_event_filter(video.frames,video.fps,filter_config=adaptive_config)
            write_video(args.output,result.frames,video.fps)
        else:
            filter_video_stream(args.input,args.output,args.method,_config(args),args.blend,adaptive_config)
        return
    if args.command=="generate-synthetic":
        destination=Path(args.output_dir); destination.mkdir(parents=True,exist_ok=True)
        for case in generate_cases(seed=args.seed).values():
            write_video(destination/f"{case.name}.{args.format}",case.frames,case.fps); npz=destination/f"{case.name}_ground_truth.npz"; __import__("numpy").savez_compressed(npz,masks=case.masks,fps=case.fps)
        return
    if args.command=="benchmark":
        from .visualization import generate_plots
        evaluate(args.output_dir,seed=args.seed,config=_config(args)); generate_plots(args.output_dir); return
    if args.command=="benchmark-analyzers": evaluate_analyzers(args.output_dir); return
    if args.command=="ablate": run_ablations(args.output_dir,seed=args.seed); return
    if args.command=="sweep":
        from .studies import run_parameter_sweep
        run_parameter_sweep(args.output_dir,seed=args.seed); return
    if args.command=="ablate-adaptive":
        from .studies import run_adaptive_ablation
        run_adaptive_ablation(args.output_dir,seed=args.seed); return

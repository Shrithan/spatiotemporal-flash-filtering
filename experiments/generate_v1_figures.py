"""Generate v1.0 paper/README figures from code and saved result tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd


COLORS={"input":"#374151","analysis":"#dbeafe","luminance":"#fef3c7","red":"#fee2e2","pattern":"#ede9fe","output":"#d1fae5"}


def _box(axis,xy,width,height,text,color,fontsize=9):
    x,y=xy
    patch=FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.012,rounding_size=0.018",facecolor=color,edgecolor="#374151",linewidth=1.1)
    axis.add_patch(patch); axis.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=fontsize)
    return patch


def _arrow(axis,start,end):
    axis.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,color="#4b5563",linewidth=1.2))


def architecture_figure(output_dir: Path) -> list[Path]:
    """Draw only stages implemented by the event-specific adaptive method."""
    fig,axis=plt.subplots(figsize=(11,6.4)); axis.set_xlim(0,1); axis.set_ylim(0,1); axis.axis("off")
    _box(axis,(.39,.88),.22,.075,"Decoded sRGB video",COLORS["input"],10)
    _box(axis,(.35,.75),.30,.075,"sRGB inverse transfer function\n+ relative linear luminance Y",COLORS["analysis"],9)
    _arrow(axis,(.5,.88),(.5,.83)); _arrow(axis,(.5,.75),(.5,.69))
    _box(axis,(.39,.62),.22,.07,"Evidence analysis",COLORS["analysis"],10)
    centers=(.18,.5,.82)
    evidence=("Opposing luminance\ntransitions","Saturated-red\ndominance proxy","Regular stripe\ntile profiles")
    correction=("Luminance-step\nlimiting","Equal-luminance\nlocal desaturation","Local contrast\nreduction")
    colors=(COLORS["luminance"],COLORS["red"],COLORS["pattern"])
    for center,label,operation,color in zip(centers,evidence,correction,colors):
        _arrow(axis,(.5,.62),(center,.54)); _box(axis,(center-.12,.45),.24,.09,label,color)
        _arrow(axis,(center,.45),(center,.36)); _box(axis,(center-.12,.27),.24,.09,operation,color)
        _arrow(axis,(center,.27),(.5,.19))
    _box(axis,(.33,.11),.34,.08,"Gaussian-feathered masks\n+ event-specific compositing",COLORS["analysis"],9)
    _arrow(axis,(.5,.11),(.5,.07)); _box(axis,(.39,.005),.22,.055,"Filtered output video",COLORS["output"],10)
    axis.set_title("Event-Specific Adaptive Filtering",fontsize=15,weight="bold",pad=12)
    fig.tight_layout(); outputs=[]
    for suffix,dpi in (("png",220),("svg",None)):
        path=output_dir/f"adaptive_architecture.{suffix}"; fig.savefig(path,dpi=dpi,bbox_inches="tight"); outputs.append(path)
    plt.close(fig); return outputs


def result_figures(results_dir: Path, output_dir: Path) -> list[Path]:
    outputs=[]
    ablation=pd.read_csv(results_dir/"adaptive_ablation"/"adaptive_ablation_summary.csv")
    order=["luminance_only","red_only","pattern_only","luminance_red","luminance_pattern","red_pattern","all_channels"]
    labels=["L","R","P","L+R","L+P","R+P","L+R+P"]
    table=ablation.set_index("config_id").loc[order]
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8))
    axes[0].bar(labels,table.mean_activity,color="#2563eb"); axes[0].set(ylabel="Residual mean temporal activity",title="Suppression")
    axes[1].bar(labels,table.mae,color="#d97706"); axes[1].set(ylabel="Mean absolute error",title="Distortion")
    for axis in axes: axis.grid(axis="y",alpha=.22); axis.set_xlabel("Enabled evidence channels")
    fig.suptitle("Adaptive component ablation (L=luminance, R=red, P=pattern)",fontsize=12); fig.tight_layout()
    path=output_dir/"adaptive_ablation.png"; fig.savefig(path,dpi=220); plt.close(fig); outputs.append(path)

    benchmark=pd.read_csv(results_dir/"benchmark.csv")
    scenarios=["scene_cut","moving_bright_object","global_brightness_change"]
    scenario_labels=["Scene cut","Moving object","Global exposure\nchange"]
    methods=["global","localized","adaptive"]
    method_colors={"global":"#d97706","localized":"#2563eb","adaptive":"#059669"}
    subset=benchmark[benchmark.scenario.isin(scenarios)&benchmark.method.isin(methods)].copy()
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    pivot=subset.pivot(index="scenario",columns="method",values="modified_area_ratio").loc[scenarios,methods]
    pivot.plot.bar(ax=axes[0],color=[method_colors[name] for name in methods]); axes[0].set(ylabel="Modified area ratio",xlabel="",title="Unnecessary modification on confounds")
    pivot=subset.pivot(index="scenario",columns="method",values="mean_activity").loc[scenarios,methods]
    pivot.plot.bar(ax=axes[1],color=[method_colors[name] for name in methods]); axes[1].set(ylabel="Residual mean temporal activity",xlabel="",title="Residual activity on confounds")
    for axis in axes:
        axis.set_xticklabels(scenario_labels,rotation=15,ha="right")
        axis.grid(axis="y",alpha=.22)
        axis.legend(title="Method")
    fig.tight_layout(); path=output_dir/"failure_cases.png"; fig.savefig(path,dpi=220); plt.close(fig); outputs.append(path)
    return outputs


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description="Generate v1 figures from saved experiment tables.")
    parser.add_argument("--results-dir",type=Path,default=root/"experiments"/"results"/"canonical")
    parser.add_argument("--output-dir",type=Path,default=root/"paper"/"figures")
    args=parser.parse_args()
    output=args.output_dir; output.mkdir(parents=True,exist_ok=True)
    outputs=architecture_figure(output)+result_figures(args.results_dir,output)
    print("\n".join(str(path) for path in outputs))


if __name__=="__main__": main()

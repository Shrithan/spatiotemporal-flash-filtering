"""Plots generated exclusively from saved experimental data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def generate_plots(results_dir: str|Path, figures_dir: str|Path|None=None) -> list[Path]:
    results=Path(results_dir); figures=Path(figures_dir or results/"figures"); figures.mkdir(parents=True,exist_ok=True); outputs=[]
    table=pd.read_csv(results/"benchmark.csv"); summary=pd.read_csv(results/"summary.csv"); traces=pd.read_csv(results/"temporal_traces.csv")
    fig,ax=plt.subplots(figsize=(7,4)); subset=traces[traces.scenario=="small_flashing_square"]
    for name,group in subset.groupby("method"): ax.plot(group.frame,group.activity,label=name)
    ax.set(xlabel="Frame",ylabel="Global mean temporal change",title="Small flashing square: global dilution"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); p=figures/"temporal_activity.png"; fig.savefig(p,dpi=180); plt.close(fig); outputs.append(p)
    fig,ax=plt.subplots(figsize=(6,4));
    for name,group in table.groupby("method"): ax.scatter(group.mae,group.mean_activity,label=name,alpha=.75)
    ax.set(xlabel="MAE from source",ylabel="Mean temporal activity",title="Suppression-distortion tradeoff"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); p=figures/"tradeoff.png"; fig.savefig(p,dpi=180); plt.close(fig); outputs.append(p)
    fig,axes=plt.subplots(1,2,figsize=(9,4)); summary.plot.bar(x="method",y="modified_area_ratio",ax=axes[0],legend=False); summary.plot.bar(x="method",y="fps",ax=axes[1],legend=False); axes[0].set_ylabel("Modified area ratio"); axes[1].set_ylabel("Processing FPS"); fig.tight_layout(); p=figures/"area_runtime.png"; fig.savefig(p,dpi=180); plt.close(fig); outputs.append(p)
    targets=table[(table.category=="target") & (table.method=="localized")]; fig,ax=plt.subplots(figsize=(7,4)); targets.plot.bar(x="scenario",y=["iou","f1"],ax=ax); ax.set_ylabel("Score"); ax.set_ylim(0,1.05); fig.tight_layout(); p=figures/"localization.png"; fig.savefig(p,dpi=180); plt.close(fig); outputs.append(p)
    return outputs

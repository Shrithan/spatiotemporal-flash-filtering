import json

import numpy as np
import pandas as pd

from flashfilter.studies import (
    ABLATION_CHANNELS,
    SweepDefinition,
    pareto_efficient,
    run_adaptive_ablation,
    run_parameter_sweep,
)


def test_pareto_efficient_marks_only_non_dominated_points():
    values=np.array([[0.,1.],[.5,.5],[1.,0.],[.7,.8]])
    assert pareto_efficient(values).tolist()==[True,True,True,False]


def test_small_sweep_is_deterministic_and_serialized(tmp_path):
    definition=SweepDefinition(
        global_thresholds=(.1,),global_blends=(.5,),
        localized_thresholds=(.1,),localized_blends=(.5,),localized_block_sizes=(4,),
        adaptive_luminance_thresholds=(.1,),adaptive_max_luminance_steps=(.08,),adaptive_profiles=("moderate",),
    )
    first_detail,first_summary=run_parameter_sweep(tmp_path/"first",length=8,height=16,width=16,definition=definition)
    second_detail,second_summary=run_parameter_sweep(tmp_path/"second",length=8,height=16,width=16,definition=definition)
    columns=[column for column in first_detail.columns if column not in {"seconds","fps"}]
    pd.testing.assert_frame_equal(first_detail[columns],second_detail[columns])
    assert set(first_summary.method)=={"none","global","localized","adaptive"}
    assert (tmp_path/"first"/"suppression_distortion_pareto.png").exists()
    config=json.loads((tmp_path/"first"/"sweep_definition.json").read_text())
    assert config["seed"]==7 and config["medical_risk_function"] is False


def test_adaptive_ablation_enumerates_channel_combinations(tmp_path):
    detail,summary=run_adaptive_ablation(tmp_path,length=8,height=16,width=16)
    assert set(summary.config_id)=={item[0] for item in ABLATION_CHANNELS}
    assert len(detail)==12*len(ABLATION_CHANNELS)
    assert (tmp_path/"adaptive_ablation.csv").exists()

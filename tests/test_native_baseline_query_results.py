import numpy as np
import pytest
from experiments.native_baseline_repair.query_results import query_order,verify_identity,errors,metrics
from experiments.native_baseline_repair.query_results import verify_terminal

def test_query_order_comes_from_native_loop_log():
    assert query_order("Localize image: seq1/a.png\nINFO Localize image:seq1/b.png\n")==["seq1/a.png","seq1/b.png"]

def test_full_unique_named_gt_coverage():
    eye=np.eye(4).tolist();snapshot=dict(queries=[dict(name="a",pose_w2c=eye),dict(name="b",pose_w2c=eye)])
    rows=[dict(gt_pose_w2c=eye)]*2
    verify_identity(["a","b"],rows,["a","b"],snapshot)
    for names,results in [(["a","a"],rows),(["b","a"],rows),(["a"],rows[:1])]:
        with pytest.raises(ValueError):verify_identity(names,results,["a","b"],snapshot)
    wrong=np.eye(4);wrong[0,3]=1
    with pytest.raises(ValueError):verify_identity(["a","b"],[dict(gt_pose_w2c=wrong),rows[1]],["a","b"],snapshot)

def test_center_error_and_no_failure_dropping():
    gt=np.eye(4);pred=np.eye(4);pred[0,3]=2
    te,re=errors(pred,gt);assert te==200 and re==0
    result=metrics([1,2,200],[0,0,0])
    assert result["query_count"]==3 and result["gt_1m_count"]==1 and result["median_te_cm"]==2

def test_terminal_and_launch_bind_same_contract():
    terminal=dict(terminal=True,exit_code=0,case="cell",job_id="wc_job_actual",
                  evidence_source="WebCodex observe_jobs",contract_sha256="current")
    launch=dict(contract_sha256="current")
    verify_terminal(terminal,launch,"current","cell")
    for changes in [dict(contract_sha256="old"),dict(job_id=""),dict(evidence_source="unchecked"),dict(exit_code=1)]:
        with pytest.raises(ValueError):verify_terminal(dict(terminal,**changes),launch,"current","cell")
    with pytest.raises(ValueError):verify_terminal(terminal,dict(contract_sha256="old"),"current","cell")


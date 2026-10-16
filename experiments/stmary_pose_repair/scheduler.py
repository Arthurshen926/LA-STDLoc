"""Scheduling-only sidecar for an already frozen StMary mechanism experiment.

The original numeric selector, map deployment, evaluator and final verifier are
called unchanged. No placeholder B and no cached trace from a different source.
"""
from __future__ import annotations
import argparse, contextlib, fcntl, inspect, time
from pathlib import Path
import numpy as np
import torch
from experiments.stmary_pose_repair import mechanism as S

DESIGN={"schema":"stmary_scheduler_v1",
 "changes":["compute authentic new-native rows/trace before old-native B exists",
            "publish new maps only after original budget validation",
            "evaluate a ready cell without waiting for unrelated maps"],
 "unchanged":"base protocol, numerical code, parameters, data, selection and evaluation metrics",
 "final_acceptance":"original mechanism.report over all six cells"}
NEW_CONFIGS=("new_native","new_spatial_B","new_spatial_N")

def code_hashes():
    paths=(Path(__file__).resolve(),S.A.ROOT/"tests/test_stmary_pose_repair_scheduler.py")
    return {str(p.relative_to(S.A.ROOT)):S.sha256(p) for p in paths}

@contextlib.contextmanager
def lock(root,name,blocking=False):
    folder=Path(root)/"scheduler/locks";folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+".lock")).open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX if blocking else fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def freeze(root):
    root=Path(root);p=S.protocol(root)
    S.A.write_new(root/"scheduler/protocol.json",{
        "design":DESIGN,"mechanism_protocol_sha256":S.sha256(root/"protocol.json"),
        "mechanism_code_sha256":p["code_sha256"],"scheduler_code_sha256":code_hashes()})

def protocol(root):
    root=Path(root);p=S.protocol(root);q=S.A.read(root/"scheduler/protocol.json")
    if (q["design"]!=DESIGN or q["mechanism_protocol_sha256"]!=S.sha256(root/"protocol.json") or
        q["mechanism_code_sha256"]!=p["code_sha256"] or q["scheduler_code_sha256"]!=code_hashes()):
        raise ValueError("scheduler/base protocol or code changed")
    return p

def inputs(root):
    root=Path(root);summary,full,source=S.bound_full(root,"anygsloc")
    if Path(summary["selector_state"]["path"]).resolve()!=(root/"new_full/anygsloc/selector_state.pt").resolve():
        raise ValueError("selector-state path differs")
    state=S.A.load(S.unchanged(summary["selector_state"],hash_content=True))
    expected={"protocol_sha256":S.sha256(root/"protocol.json"),
              "cache_sha256":source["sha256"],"full_sha256":summary["full"]["sha256"]}
    if state["provenance"].get("mechanism_binding")!=expected or any(
            not torch.equal(state[k],full[k]) for k in ("anchor_xyz","anchor_features")):
        raise ValueError("selector state detached from current full map/cache")
    selector_path=Path(inspect.getsourcefile(S.A.select_compact_rows_v2)).resolve()
    binding={**expected,"selector_state_sha256":summary["selector_state"]["sha256"],
             "scheduler_protocol_sha256":S.sha256(root/"scheduler/protocol.json"),
             "selector_source":{"path":str(selector_path),"sha256":S.sha256(selector_path)}}
    return state,source,binding

def valid_rows(rows,count):
    if (not isinstance(rows,torch.Tensor) or rows.dtype!=torch.long or rows.device.type!="cpu" or
        rows.ndim!=1 or rows.numel()==0 or len(torch.unique(rows))!=rows.numel() or
        int(rows.min())<0 or int(rows.max())>=count):
        raise ValueError("invalid native selector rows")

def precompute(root):
    root=Path(root);protocol(root)
    with lock(root,"new-native-computation"):
        folder=root/"scheduler/new_native"
        if (folder/"selection.pt").exists() or (folder/"summary.json").exists():
            raise FileExistsError("new native precompute already exists")
        state,source,binding=inputs(root)
        cache=S.A.load(S.unchanged(source,hash_content=True))["queries"]
        start=time.perf_counter()
        rows,trace=S.A.select_compact_rows_v2(state,cache)
        elapsed=time.perf_counter()-start
        valid_rows(rows,len(state["anchor_xyz"]))
        protocol(root);S.unchanged(source)
        S.atomic_save_new({"rows":rows,"trace":trace},folder/"selection.pt")
        S.A.write_new(folder/"summary.json",{
            "complete":True,"binding":binding,"N":int(rows.numel()),
            "budget_consumed":False,"seconds":elapsed,"payload":S.evidence(folder/"selection.pt")})

def publish(root):
    root=Path(root);protocol(root)
    with lock(root,"new-native-publication"):
        b=S.budget(root);state,source,binding=inputs(root)
        folder=root/"scheduler/new_native";meta=S.A.read(folder/"summary.json")
        if (meta.get("complete") is not True or meta.get("budget_consumed") is not False or
            meta["binding"]!=binding or Path(meta["payload"]["path"]).resolve()!=(folder/"selection.pt").resolve()):
            raise ValueError("precomputed selector source binding differs")
        saved=S.A.load(S.unchanged(meta["payload"],hash_content=True))
        rows,trace=saved["rows"],saved["trace"];valid_rows(rows,len(state["anchor_xyz"]))
        n=int(rows.numel())
        if meta["N"]!=n or not torch.equal(trace["final_source_ids"],rows):
            raise ValueError("native rows and authentic trace differ")
        for c in NEW_CONFIGS:
            if (root/"maps"/c).exists():raise FileExistsError("new selection already exists: "+c)
        if (root/"new_native_trace.pt").exists() or (root/"new_native_summary.json").exists():
            raise FileExistsError("native trace or summary already exists")
        started=time.perf_counter()
        lineage={"source_full_sha256":binding["full_sha256"],"cache_sha256":source["sha256"]}
        S.deploy(root,"new_native",state,rows.numpy(),{"natural_N":n,"B":b,**lineage})
        S.deploy(root,"new_spatial_B",state,S.spatial_rows(state["anchor_xyz"].numpy(),b),{"B":b,**lineage})
        S.deploy(root,"new_spatial_N",state,S.spatial_rows(state["anchor_xyz"].numpy(),n),{"N":n,**lineage})
        S.A.save_new(root/"new_native_trace.pt",trace)
        S.A.write_new(root/"new_native_summary.json",{
            "N":n,"B":b,"seconds":meta["seconds"]+time.perf_counter()-started,
            "trace":S.evidence(root/"new_native_trace.pt"),
            "scheduler_protocol_sha256":binding["scheduler_protocol_sha256"],
            "precomputed_selection_sha256":meta["payload"]["sha256"],
            "timing_scope":"selector plus publication; scheduling gap excluded"})
        for c in NEW_CONFIGS:S.checked_selection(root,c)

def evaluate(root,config,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root)
    if config not in S.CONFIGS:raise ValueError("invalid configuration")
    if torch.device(device).type!="cuda" or torch.device(device).index is None:
        raise ValueError("explicit indexed CUDA device required")
    with lock(root,"evaluate-"+config):
        S.budget(root);sel=S.checked_selection(root,config);s=p["scenes"][S.SCENE]
        output=root/"evaluations"/config
        if output.exists():raise FileExistsError("evaluation already exists")
        dataset=ColmapDataset(s["dataset"],load_masks=True);queries=dataset.split("test")
        if [c.image_name for c in queries]!=s["test_names"] or any(dataset.valid_mask(c) is None for c in queries):
            raise ValueError("incomplete query-on input")
        with lock(root,"query-snapshot",blocking=True):
            snap=S.M.evaluation_snapshot(root,S.SCENE,dataset,s)
        run(["--dataset",s["dataset"],"--map",sel["path"],"--output",str(output),
             "--image-domain","rectified","--images","processed","--mask-policy","explicit_valid_mask",
             "--rectification-contract","opencv_same_k","--seed","2026","--device",device])
        with lock(root,"query-snapshot",blocking=True):
            if S.M.evaluation_snapshot(root,S.SCENE,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:
                raise ValueError("query input changed")
        protocol(root)
        S.A.write_new(output/"attestation.json",{
            "protocol_sha256":S.sha256(root/"protocol.json"),"query_inputs_sha256":snap,
            "map_sha256":sel["sha256"],"budget_sha256":S.sha256(root/"budget.json"),
            "fresh_evaluation":True,"scene":S.SCENE,"configuration":config,
            "scheduler_protocol_sha256":S.sha256(root/"scheduler/protocol.json"),
            "files":{n:S.A.checked(output/n) for n in ("results.json","manifest.json","summary.json")}})
        S.verify_cell(root,config,p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","precompute-new-native","publish-new-native","early-evaluate"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--config",choices=S.CONFIGS);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=="freeze":freeze(a.root)
    elif a.stage=="precompute-new-native":precompute(a.root)
    elif a.stage=="publish-new-native":publish(a.root)
    else:evaluate(a.root,a.config,a.device)
if __name__=="__main__":main()

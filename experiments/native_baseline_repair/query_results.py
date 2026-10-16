"""Read-only metric/completeness verification, publishing only a fresh cell verification receipt."""
import argparse
import json
import math
from pathlib import Path
import re
import cv2
import numpy as np
from experiments.native_baseline_repair.query_plan import ref, verify, validate_contract, write_new

def query_order(log):
    return re.findall(r"Localize image:\s*(\S+)",log)

def verify_identity(names,results,expected,snapshot):
    if names!=expected or len(set(names))!=len(names) or len(results)!=len(expected):
        raise ValueError("query coverage/order/uniqueness mismatch")
    gt={q["name"]:np.asarray(q["pose_w2c"],dtype=np.float32) for q in snapshot["queries"]}
    for name,row in zip(names,results):
        if not np.array_equal(np.asarray(row["gt_pose_w2c"],dtype=np.float32),gt[name]):
            raise ValueError("result paired with wrong frozen GT")
    return gt

def errors(pred,gt):
    pred=np.asarray(pred,dtype=np.float32);gt=np.asarray(gt,dtype=np.float32)
    if pred.shape!=(4,4) or not np.isfinite(pred).all():raise ValueError("invalid predicted matrix")
    te=float(np.linalg.norm(np.linalg.inv(pred)[:3,3]-np.linalg.inv(gt)[:3,3])*100)
    re_deg=float(np.linalg.norm(cv2.Rodrigues(gt[:3,:3]@pred[:3,:3].T)[0])*180/math.pi)
    return te,re_deg

def metrics(te,re_deg):
    te=np.asarray(te);re_deg=np.asarray(re_deg)
    return dict(median_te_cm=float(np.median(te)),median_re_deg=float(np.median(re_deg)),
                r5_percent=float(np.mean((te<5)&(re_deg<5))*100),
                r15_percent=float(np.mean((te<15)&(re_deg<5))*100),
                p90_te_cm=float(np.percentile(te,90)),gt_1m_count=int((te>100).sum()),query_count=len(te))

def verify_terminal(terminal, launch, contract_sha, case):
    if not(terminal.get("terminal") is True and terminal.get("exit_code")==0
           and terminal.get("case")==case and str(terminal.get("job_id","")).startswith("wc_job_")
           and terminal.get("evidence_source")=="WebCodex observe_jobs"
           and terminal.get("contract_sha256")==contract_sha
           and launch.get("contract_sha256")==contract_sha):
        raise ValueError("native query job not proven complete for this exact launched contract")

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--contract",type=Path,required=True)
    ap.add_argument("--terminal-evidence",type=Path,required=True);a=ap.parse_args()
    c=json.loads(a.contract.read_text());p=json.loads(Path(c["protocol"]["path"]).read_text())
    verify(c["protocol"]);validate_contract(c,p,Path(c["protocol"]["path"]))
    terminal=json.loads(a.terminal_evidence.read_text())
    launch_path=a.contract.parent/"launch/launch.json"
    launch=json.loads(launch_path.read_text())
    verify_terminal(terminal,launch,ref(a.contract)["sha256"],c["case"])
    for item in c["model_assets"].values():verify(item)
    receipt_path=Path(c["receipt"]);receipt=json.loads(receipt_path.read_text())
    for k,v in dict(status="complete",method=c["method"],phase="evaluate",mapping_semantics=c["mapping_semantics"],
                    query_semantics=True,seed=2026,warmup_queries=0).items():
        if receipt.get(k)!=v:raise ValueError("native query receipt mismatch: "+k)
    if receipt["native_output"]!=c["native_output"]:raise ValueError("native output redirected")
    out=Path(c["native_output"]);results_path=out/"results.json";results=json.loads(results_path.read_text())
    log=a.contract.parent/"launch/native.log";names=query_order(log.read_text(errors="replace"))
    verify(c["query_snapshot"])
    snapshot=json.loads(Path(c["query_snapshot"]["path"]).read_text())
    gt=verify_identity(names,results,c["query_names"],snapshot)
    if len(receipt["timings"])!=len(names):raise ValueError("native localize calls did not cover all queries")
    stats={};deltas={}
    for stage,key in [("sparse","sparse"),("dense","dense")]:
        te=[];ae=[];dt=[];da=[]
        for name,row in zip(names,results):
            pose=row[key]["pose_w2c"] if stage=="sparse" else row[key][-1]["pose_w2c"]
            t,r=errors(pose,gt[name]);te.append(t);ae.append(r)
            dt.append(abs(t-row[stage+"_TE"]));da.append(abs(r-row[stage+"_AE"]))
        # Serialization/dtype roundoff is diagnostic, never a dropped-query criterion.
        stats[stage]=metrics(te,ae)
        deltas[stage]=dict(max_te_cm=max(dt),max_re_deg=max(da))
        if not np.allclose(te,[row[stage+"_TE"] for row in results],rtol=2e-6,atol=1e-4):
            raise ValueError("independent translation metric differs from native output")
        if not np.allclose(ae,[row[stage+"_AE"] for row in results],rtol=2e-6,atol=1e-4):
            raise ValueError("independent rotation metric differs from native output")
    value=dict(complete=True,case=c["case"],metrics=stats,metric_recompute_deltas=deltas,
               contract=ref(a.contract),terminal_evidence=ref(a.terminal_evidence),native_receipt=ref(receipt_path),
               results=ref(results_path),native_log=ref(log),launch=ref(launch_path),query_names=names,query_order_complete_unique=True,
               telemetry_not_formal_benchmark=True,verifier=ref(__file__))
    write_new(a.contract.parent/"verification.json",value)
    print(json.dumps(dict(case=c["case"],metrics=stats,path=str(a.contract.parent/"verification.json"))))
if __name__=="__main__":main()

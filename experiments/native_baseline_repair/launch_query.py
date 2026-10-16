"""Finite native query-on evaluator launcher with immutable inputs and an explicit GPU lease."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from experiments.native_baseline_repair.launch_ulf import sha, read, check_ref, nofile_pair
from experiments.native_baseline_repair.query_plan import validate_contract

def check_lease(v, contract_sha, now=None):
    now=time.time() if now is None else now
    if (v.get("approved") is not True or v.get("scope")!="native_baseline_query_semantic_on"
        or v.get("contract_sha256")!=contract_sha or not v.get("approval_reference")
        or not v.get("owner_handover") or not v.get("gpu_uuid","").startswith("GPU-")
        or not v.get("valid_from_unix",float("inf"))<=now
        or v.get("valid_until_unix",-1)<now+1860
        or v["valid_until_unix"]-v["valid_from_unix"]>7200):
        raise ValueError("invalid or expiring query GPU lease")
    return v["gpu_uuid"]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract",type=Path,required=True);ap.add_argument("--lease",type=Path,required=True)
    a=ap.parse_args();c=read(a.contract);ch=sha(a.contract)
    if c["schema"]!="native_baseline_query_semantic_on_cell_v1" or not c["query_semantics"]:
        raise ValueError("invalid query contract")
    check_ref(c["protocol"]);p=read(c["protocol"]["path"])
    validate_contract(c,p,Path(c["protocol"]["path"]))
    review=read(Path(c["protocol"]["path"]).parent/"independent_review.json")
    if not (review.get("complete") is True and review.get("verdict")=="PASS"
            and review.get("protocol_sha256")==c["protocol"]["sha256"]):
        raise ValueError("query adapter independent review is absent")
    lh=sha(a.lease);uuid=check_lease(read(a.lease),ch)
    for path,h in p["code_sha256"].items():
        if sha(path)!=h:raise ValueError("frozen query code changed: "+path)
    for r in [c["wrapper"],c["native_entry"],c["native_config"],c["query_snapshot"],*c["model_assets"].values(),*c["scene_inputs"].values()]:
        check_ref(r)
    if c.get("training_acceptance"):check_ref(c["training_acceptance"])
    # Recheck all frozen query RGB bytes at the last pre-launch boundary.
    snapshot=read(c["query_snapshot"]["path"])
    for q in snapshot["queries"]:
        if sha(Path(c["source"])/"processed"/q["name"])!=q["image_sha256"]:
            raise ValueError("query RGB changed")
    model=Path(c["model_view"])
    for rel,r in c["model_assets"].items():
        if not os.path.samefile(model/rel,r["path"]):raise ValueError("model view retargeted")
    if Path(c["receipt"]).exists() or Path(c["native_output"]).exists():raise FileExistsError("evaluation output exists")
    gpu_rows=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()
    if uuid not in [x.strip() for x in gpu_rows]:raise ValueError("GPU UUID unavailable")
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
    if any(x.split(",")[0].strip()==uuid for x in apps.splitlines()):raise RuntimeError("leased GPU occupied")
    if sha(a.lease)!=lh or check_lease(read(a.lease),ch)!=uuid:raise ValueError("lease changed")
    if sha(a.contract)!=ch:raise ValueError("query contract changed")
    resource.setrlimit(resource.RLIMIT_NOFILE,nofile_pair(*resource.getrlimit(resource.RLIMIT_NOFILE)))
    env=os.environ.copy();env.update(c["env"]);env["CUDA_VISIBLE_DEVICES"]=uuid
    for key in ("ULFLOC_POSTPROCESS_PLY","ULFLOC_POSTPROCESS_INDICES","ANYGSLOC_DELEGATE_EXECUTOR"):env.pop(key,None)
    adapter=Path(__file__).with_name("query_semantic_adapter.py")
    command=[c["executable"],str(adapter),"--contract",str(a.contract)]
    out=a.contract.parent/"launch";out.mkdir(exist_ok=False)
    (out/"launch.json").write_text(json.dumps(dict(contract_sha256=ch,lease_sha256=lh,command=command,
        gpu_uuid=uuid,pid=os.getpid(),unix=time.time(),runner_timeout_seconds=1800),indent=2))
    print(json.dumps(dict(case=c["case"],gpu_uuid=uuid,command=command)),flush=True)
    with (out/"native.log").open("xb",buffering=0) as f:
        os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
        os.chdir(Path(c["native_entry"]["path"]).parent);os.execvpe(command[0],command,env)
if __name__=="__main__":main()

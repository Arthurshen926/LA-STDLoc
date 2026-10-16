"""Small stage2 artifact builder. Requires real completed prefix and separately supplied measured budget."""
import argparse,json,sys,time
from pathlib import Path
V3=Path("/root/STDLoc/experiments/matcha_state_continuation_v3")
sys.path.insert(0,str(V3))
import stage_launcher as launch
R=Path("/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1")
P=R/"matcha_state_continuation_v3_preparation"
def ref(path):return {"path":str(path),"sha256":launch.sha(path)}
def write(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,sort_keys=True)
def main(receipt_path,budget_path):
    first=launch.validate_spec(json.loads((P/"stage1_spec.json").read_text()))
    launch.verify_frozen(first,time.monotonic()+30)
    root=Path(first["output_root"])
    result_path=root/"launcher_result.json"; result=json.loads(result_path.read_text())
    receipt=json.loads(receipt_path.read_text())
    launch.require(receipt["schema"]=="matcha_v3_official_prefix_terminal" and
      receipt["status"]=="completed" and receipt["exit_code"]==0 and
      receipt["launcher_result_sha256"]==launch.sha(result_path),"official successful prefix required")
    launch.require(result["stage"]==1 and result["verdict"]=="PREFIX14999_COMPLETE" and
      result["terminal"]==14999 and not result["floating_trajectory_equivalence_claimed"],"prefix incomplete")
    owner=json.loads((root/"launch.json").read_text())
    launch.require(owner["spec_sha256"]==launch.sha(P/"stage1_spec.json"),"prefix spec changed")
    checkpoint=root/"state/step_14999/manifest.json"; static=root/"state/static/manifest.json"
    for path in (checkpoint,static):
        info=json.loads(path.read_text())
        launch.require(info["schema"]=="matcha_exact_resume_v2" and
          (path.parent/"state.pt").stat().st_size==info["bytes"],"manifest/actual size mismatch")
    perf=[json.loads(line) for line in (root/"state/performance.jsonl").read_text().splitlines()]
    launch.require(perf[-1]["event"]=="stop" and perf[-1]["iteration"]==14999,"truncated prefix")
    setup=[x for x in perf if x["event"]=="fresh_setup"]
    capture=[x for x in perf if x["event"]=="full_capture"]
    launch.require(len(setup)==1 and len(capture)==1 and capture[0]["iteration"]==14999,
                   "unexpected observation frequency")
    measurement={"schema":"matcha_v3_actual_prefix_resources","launcher_result":ref(result_path),
      "official_terminal":ref(receipt_path),"checkpoint_manifest":ref(checkpoint),"static_manifest":ref(static),
      "performance":ref(root/"state/performance.jsonl"),"setup":setup[0],"capture14999":capture[0],
      "last1000_progress":[x for x in perf if x["event"]=="progress" and x["iteration"]>=13900],
      "elapsed_seconds":result["elapsed_seconds"],"dynamic_payload_not_reloaded":True}
    budget=json.loads(budget_path.read_text())
    spec=dict(first);spec["stage"]=2
    spec["output_root"]=str(launch.RUNS/(launch.SERIES+"_stage2"))
    spec["references"]=dict(first["references"])
    spec["references"].update(prefix_result=ref(result_path),prefix_terminal=ref(receipt_path),
      checkpoint_manifest=ref(checkpoint),static_manifest=ref(static),budget=ref(budget_path),
      source_review={"path":str(P/"stage2_source_execution_review.json"),"sha256":"0"*64})
    spec["forecast_seconds"]=budget["forecast_seconds"]
    launch.validate_spec(spec);launch.validate_budget(spec,budget)
    launch.require(not Path(spec["output_root"]).exists(),"stage2 output already exists")
    write(P/"stage2_actual_resources.json",measurement);write(P/"stage2_spec.json",spec)
    print(json.dumps({"stage2_spec_sha256":launch.sha(P/"stage2_spec.json"),
      "stage2_spec_body_sha256":launch.base.spec_digest(spec),
      "actual_resources_sha256":launch.sha(P/"stage2_actual_resources.json"),
      "forecast_seconds":spec["forecast_seconds"],"gpu_launched":False,"needs_independent_review":True}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--official-receipt",required=True,type=Path)
    p.add_argument("--measured-budget",required=True,type=Path);a=p.parse_args()
    main(a.official_receipt,a.measured_budget)

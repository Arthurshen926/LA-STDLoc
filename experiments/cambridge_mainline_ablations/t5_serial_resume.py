"""Fixed serial continuation of the twelve T5 cells after the GPU1 ShopFacade lease."""
from pathlib import Path
import argparse,json,os,subprocess
from experiments.cambridge_mainline_ablations import t5 as T, t7 as C, accuracy_scheduler as S
SOURCE=Path("/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2/T5")
RELEASE=C.REPO/"outputs/anygsloc_mainline_ablations_20261011_v2/T5_GPU1_release.json"
LEASE_JOB="wc_job_QmDQm7icVihNwrMp"
REUSED=[(m,"ShopFacade") for m in T.CONFIGS]
REMAINING=[(m,s) for m in T.CONFIGS for s in C.SCENES if s!="ShopFacade"]
def code_hashes():
 return {n:C.sha256(C.REPO/n) for n in ("experiments/cambridge_mainline_ablations/t5_serial_resume.py","tests/test_cambridge_t5_resume.py","experiments/cambridge_mainline_ablations/accuracy_scheduler.py")}
def identity(att,expected):
 for key,value in expected.items():
  if att.get(key)!=value:raise ValueError("reused cell identity differs: "+key)
 if set(att.get("files",{}))!={"results.json","manifest.json","summary.json"}:raise ValueError("reused file set differs")
def checked_reuse(mode,scene):
 p=T.protocol(SOURCE);ref=T.checked_geometry(SOURCE,scene,mode);folder=SOURCE/"evaluations"/scene/mode
 att=C.read(folder/"attestation.json");snap=C.snapshot(SOURCE,scene,p["scenes"][scene])
 identity(att,dict(protocol_sha256=C.sha256(SOURCE/"protocol.json"),map_sha256=ref["sha256"],
  query_inputs_sha256=snap,scene=scene,configuration=mode))
 C.checked_files(folder,att["files"],("results.json","manifest.json","summary.json"))
 C.manifest_check(folder,ref,p["scenes"][scene],C.read(Path(p["full_references"][scene]["full"]["evaluation_folder"])/"manifest.json"))
 metrics=C.independent_metrics(C.read(folder/"results.json"),p["scenes"][scene]["test_names"])
 if metrics["query_count"]!=103:raise ValueError("ShopFacade query denominator differs")
 return {"attestation":C.evidence(folder/"attestation.json"),"files":att["files"],"map":ref}
def checked_release():
 r=C.read(RELEASE)
 for k,v in dict(released=True,owned_children_waited=True,no_remaining_gpu1_workers=True,terminal=True,exit_code=0,
  borrower="/root/plan_anygsloc_research_revision",borrower_job_id=LEASE_JOB,
  t5_protocol_sha256=C.sha256(SOURCE/"protocol.json"),cells=["ShopFacade/"+m for m in T.CONFIGS]).items():
  if r.get(k)!=v:raise ValueError("lease release differs: "+k)
 return C.evidence(RELEASE)
def checked_plan(root):
 root=Path(root);p=C.read(root/"protocol.json")
 if p["code_sha256"]!=code_hashes() or p["source_root"]!=str(SOURCE) or p["cells"]!=[list(c) for c in REMAINING] or p["reused_cells"]!=[list(c) for c in REUSED]:raise ValueError("fixed continuation plan changed")
 for ref in p["bindings"].values():C.unchanged(ref)
 T.protocol(SOURCE);C.require_review(SOURCE);checked_release()
 for key,old in p["reused"].items():
  mode,scene=key.split("/")
  if checked_reuse(mode,scene)!=old:raise ValueError("reused evidence changed")
 return p
def prepare(root):
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 T.protocol(SOURCE);C.require_review(SOURCE);release=checked_release()
 if any((SOURCE/"evaluations"/s/m).exists() for m,s in REMAINING):raise FileExistsError("unexpected existing remaining cell")
 reuse={m+"/"+s:checked_reuse(m,s) for m,s in REUSED}
 C.write_new(root/"protocol.json",dict(schema="t5_twelve_cell_serial_continuation_v1",code_sha256=code_hashes(),source_root=str(SOURCE),
  cells=REMAINING,reused_cells=REUSED,reused=reuse,bindings={**{n:C.evidence(SOURCE/n) for n in ("protocol.json","preflight.json","independent_pre_gpu_review.json","gpu1_shop_lease.json")},"release":release},
  concurrency=1,physical_gpu=0,cpu_threads=4,seed=2026,formal_timing=False,algorithm_or_parameters_changed=False))
 print("T5_RESUME_PREPARED",C.sha256(root/"protocol.json"),flush=True)
def execute_cell(mode,scene):
 if (mode,scene) not in REMAINING:raise ValueError("outside fixed remaining cells")
 S.current_admission(1)
 subprocess.run(S.query_command("T5",SOURCE,mode,scene),env=S.worker_env(),check=True,timeout=7200)
def run(root):
 root=Path(root);p=checked_plan(root);review=C.read(root/"independent_pre_gpu_review.json")
 if review.get("verdict")!="PASS" or review.get("protocol_sha256")!=C.sha256(root/"protocol.json"):raise ValueError("continuation review missing/stale")
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU0 required")
 S.current_admission(1)
 C.write_new(root/"start.json",dict(pid=os.getpid(),protocol_sha256=C.sha256(root/"protocol.json"),review=C.evidence(root/"independent_pre_gpu_review.json")))
 for mode in T.CONFIGS:
  for m,scene in REMAINING:
   if m!=mode:continue
   execute_cell(mode,scene)
   S.mirror_status("T5",dict(stage="cell_complete",configuration=mode,scene=scene,concurrency=1))
  T.report(SOURCE,mode)
  S.mirror_status("T5",dict(stage="five_scene_complete",configuration=mode,report=C.read(SOURCE/"reports"/(mode+".json")),concurrency=1))
 checked_plan(root)
 C.write_new(SOURCE/"report.json",dict(complete=True,design=T.DESIGN,results={m:C.read(SOURCE/"reports"/(m+".json")) for m in T.CONFIGS},
  execution_protocol=C.evidence(root/"protocol.json"),gpu1_lease=C.evidence(SOURCE/"gpu1_shop_lease.json")))
 print("T5_ALL_FIFTEEN_COMPLETE",flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","run"));p.add_argument("--root",type=Path,required=True);a=p.parse_args()
 (prepare if a.stage=="prepare" else run)(a.root)

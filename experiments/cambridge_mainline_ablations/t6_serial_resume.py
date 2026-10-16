"""CPU-only parent; fixed ten-cell serial continuation of T6 off conditions."""
from pathlib import Path
import argparse,json,os,subprocess,time,torch
from experiments.cambridge_mainline_ablations import t6 as T,t7 as C,accuracy_scheduler as S
SOURCE=Path("/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2/T6")
MODES=("filter_off_fused","filter_off_single")
CELLS=[(m,s) for m in MODES for s in C.SCENES]
def hashes():
 return {n:C.sha256(C.REPO/n) for n in ("experiments/cambridge_mainline_ablations/t6_serial_resume.py","tests/test_cambridge_t6_resume.py","experiments/cambridge_mainline_ablations/accuracy_scheduler.py")}
def identity(att,expected):
 for k,v in expected.items():
  if att.get(k)!=v:raise ValueError("reused identity differs: "+k)
 if set(att.get("files",{}))!={"results.json","manifest.json","summary.json"}:raise ValueError("reused files differ")
def reuse(scene):
 p=T.protocol(SOURCE);mode="filter_on_single";folder=SOURCE/"evaluations"/scene/mode;ref=T.map_for(SOURCE,scene,mode)
 att=C.read(folder/"attestation.json");snap=C.snapshot(SOURCE,scene,p["scenes"][scene])
 identity(att,dict(scene=scene,configuration=mode,protocol_sha256=C.sha256(SOURCE/"protocol.json"),map_sha256=ref["sha256"],query_inputs_sha256=snap))
 C.checked_files(folder,att["files"],("results.json","manifest.json","summary.json"))
 C.manifest_check(folder,ref,p["scenes"][scene],C.read(Path(p["reused"][scene]["full"]["evaluation_folder"])/"manifest.json"))
 C.independent_metrics(C.read(folder/"results.json"),p["scenes"][scene]["test_names"])
 return dict(attestation=C.evidence(folder/"attestation.json"),files=att["files"],map=ref)
def cpu_parent():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("parent must have no visible CUDA device/context")
def wait_empty(timeout=60.):
 start=time.monotonic();observed=[]
 while True:
  pids=sorted(S.gpu_process_ids());observed.append(pids)
  if not pids:return observed
  if time.monotonic()-start>=timeout:raise RuntimeError("GPU0 still occupied; no PID assumed owned: "+str(observed))
  time.sleep(1)
def prepare(root):
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 cpu_parent();T.protocol(SOURCE);C.require_review(SOURCE)
 if any((SOURCE/"evaluations"/s/m).exists() for m,s in CELLS):raise FileExistsError("unexpected remaining output")
 reused={s:reuse(s) for s in C.SCENES}
 C.write_new(root/"protocol.json",dict(schema="t6_ten_cell_serial_cpu_parent_v1",code_sha256=hashes(),source_root=str(SOURCE),cells=CELLS,reused=reused,
  bindings={n:C.evidence(SOURCE/n) for n in ("protocol.json","preflight.json","independent_pre_gpu_review.json","serial_execution.json","cpu_only_report_replay.json","reports/filter_on_single.json")},
  concurrency=1,physical_gpu=0,parent_cuda_visible_devices="",child_cuda_visible_devices="0",gpu_vacancy_timeout_seconds=60,cpu_threads=4,seed=2026,formal_timing=False))
 print("T6_RESUME_PREPARED",C.sha256(root/"protocol.json"),flush=True)
def checked(root):
 root=Path(root);p=C.read(root/"protocol.json")
 if p["code_sha256"]!=hashes() or p["source_root"]!=str(SOURCE) or p["cells"]!=[list(c) for c in CELLS] or set(p["reused"])!=set(C.SCENES):raise ValueError("fixed continuation changed")
 for ref in p["bindings"].values():C.unchanged(ref)
 T.protocol(SOURCE);C.require_review(SOURCE)
 for scene in C.SCENES:
  if reuse(scene)!=p["reused"][scene]:raise ValueError("reused evidence changed")
 return p
def run(root):
 root=Path(root);cpu_parent();p=checked(root);review=C.read(root/"independent_pre_gpu_review.json")
 if review.get("verdict")!="PASS" or review.get("protocol_sha256")!=C.sha256(root/"protocol.json"):raise ValueError("continuation review absent/stale")
 if any((SOURCE/"evaluations"/s/m).exists() for m,s in CELLS):raise FileExistsError("remaining output exists")
 C.write_new(root/"start.json",dict(parent_pid=os.getpid(),cuda_initialized=torch.cuda.is_initialized(),protocol=C.evidence(root/"protocol.json"),review=C.evidence(root/"independent_pre_gpu_review.json")))
 mirror=C.REPO/"outputs/anygsloc_mainline_ablations_20261011_v2"
 for mode in MODES:
  for scene in C.SCENES:
   cpu_parent();waiting=wait_empty();admission=S.current_admission(1)
   C.write_new(root/"admission"/(mode+"_"+scene+".json"),dict(wait_observations=waiting,admission=admission,cuda_initialized=torch.cuda.is_initialized()))
   subprocess.run(S.query_command("T6",SOURCE,mode,scene),env=S.worker_env(),check=True,timeout=7200)
   cpu_parent();wait_empty()
   S.mirror_status("T6",dict(stage="cell_complete",configuration=mode,scene=scene,concurrency=1))
  T.report(SOURCE,mode);cpu_parent()
  report=SOURCE/"reports"/(mode+".json");(mirror/("T6_"+mode+".json")).write_bytes(report.read_bytes())
  S.mirror_status("T6",dict(stage="five_scene_complete",configuration=mode,report=C.read(report),concurrency=1))
 T.report(SOURCE,"filter_on_fused");cpu_parent();checked(root)
 C.write_new(SOURCE/"report.json",dict(complete=True,design=T.DESIGN,results={m:C.read(SOURCE/"reports"/(m+".json")) for m in T.ALL},execution_protocol=C.evidence(root/"protocol.json"),original_execution=C.evidence(SOURCE/"serial_execution.json")))
 (mirror/"T6_report.json").write_bytes((SOURCE/"report.json").read_bytes())
 S.mirror_status("T6",dict(stage="all_complete",cells=15,report=C.evidence(SOURCE/"report.json")))
 print("T6_ALL_COMPLETE",C.sha256(SOURCE/"report.json"),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","run"));p.add_argument("--root",type=Path,required=True);a=p.parse_args()
 (prepare if a.stage=="prepare" else run)(a.root)

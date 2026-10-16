"""Accuracy-only process scheduler and bounded execution-consistency check.

This layer never changes a frozen map, threshold, query order, or method.
Its short pilot uses predicted outputs, never ground-truth accuracy.
"""
from __future__ import annotations
import argparse, concurrent.futures, ctypes, hashlib, importlib, json, math, os
from pathlib import Path
import subprocess, sys, threading, time
import numpy as np
from PIL import Image
from experiments.cambridge_mainline_ablations import t7 as C

RULES = {
 "schema":"mainline_accuracy_scheduler_v1", "cpu_threads_per_worker":4,
 "maximum_concurrency":2, "gpu":"cuda:0", "gpu_index":0,
 "memory_limit_bytes":20*1024**3, "context_reserve_bytes":2*1024**3,
 "minimum_free_cpu_cores":12, "minimum_available_ram_bytes":16*1024**3,
 "minimum_throughput_gain":0.10, "pose_atol":1e-7, "pose_rtol":1e-6,
 "pilot_queries":8, "pilot_repetitions":2, "warmup_queries":2,
 "nvml_poll_seconds":0.02, "child_timeout_seconds":600,
 "scope":"accuracy only; all measured runtimes are telemetry, not paper timing",
 "selection":"largest N map; first 7 ordered queries plus largest pixel-count query",
 "fallback":"serial if any contract, consistency, resource, or speed gate fails",
 "ground_truth_used":False, "pilot_outputs_used_for_method_selection":False,
}
def driver(table):
 if table not in ("T5","T6"):raise ValueError("outside approved tables")
 return importlib.import_module("experiments.cambridge_mainline_ablations."+table.lower())
def hashes():
 return {p:C.sha256(C.REPO/p) for p in (
  "experiments/cambridge_mainline_ablations/accuracy_scheduler.py",
  "tests/test_cambridge_accuracy_scheduler.py")}
def worker_env():
 env=os.environ.copy()
 env.update({k:"4" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
 env["CUDA_VISIBLE_DEVICES"]="0"
 return env
def memory_available():
 for line in Path("/proc/meminfo").read_text().splitlines():
  if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
 raise ValueError("RAM telemetry unavailable")
def cpu_free():
 return len(os.sched_getaffinity(0))-math.ceil(os.getloadavg()[0])
def finite_cell_order(table):
 modes=driver(table).CONFIGS if table=="T5" else driver(table).NEW
 return [(mode,scene) for mode in modes for scene in C.SCENES]
def source(root):
 root=Path(root);p=C.read(root/"protocol.json")
 if p["rules"]!=RULES or p["code_sha256"]!=hashes():raise ValueError("scheduler code/rules changed")
 for ref in p["bindings"].values():C.unchanged(ref)
 d=driver(p["table"]);d.protocol(p["source_root"]);C.require_review(p["source_root"])
 return p
def require_review(root):
 root=Path(root);r=C.read(root/"independent_pre_gpu_review.json")
 if r.get("verdict")!="PASS" or r.get("protocol_sha256")!=C.sha256(root/"protocol.json"):raise ValueError("scheduler review missing/stale")
 source(root)
def choose_names(cameras,sizes):
 ordered=[c.image_name for c in cameras]
 largest=max(ordered,key=lambda n:(sizes[n][0]*sizes[n][1],-ordered.index(n)))
 names=ordered[:RULES["pilot_queries"]-1]
 if largest not in names:names.append(largest)
 else:names=ordered[:RULES["pilot_queries"]]
 return names
def prepare(root,table,source_root):
 from anygsloc.data.datasets import ColmapDataset
 root=Path(root);source_root=Path(source_root).resolve()
 if root.exists():raise FileExistsError(root)
 d=driver(table);p=d.protocol(source_root);C.require_review(source_root)
 refs={}
 for mode,scene in finite_cell_order(table):
  refs[mode+"/"+scene]=d.checked_geometry(source_root,scene,mode) if table=="T5" else d.map_for(source_root,scene,mode)
 key=max(refs,key=lambda k:(refs[k]["count"],k));mode,scene=key.split("/")
 sizes={};cameras={}
 for name in C.SCENES:
  dataset=ColmapDataset(p["scenes"][name]["dataset"],load_masks=False)
  cameras[name]=dataset.split("test");sizes[name]={}
  for camera in cameras[name]:
   with Image.open(camera.image_path) as image:sizes[name][camera.image_name]=list(image.size)
 selected=choose_names(cameras[scene],sizes[scene])
 refkey="full_references" if table=="T5" else "reused"
 reference=C.read(Path(p[refkey][scene]["full"]["evaluation_folder"])/"manifest.json")
 maxw=max(v[0] for ss in sizes.values() for v in ss.values())
 maxh=max(v[1] for ss in sizes.values() for v in ss.values())
 covered=any(sizes[scene][n][0]>=maxw and sizes[scene][n][1]>=maxh for n in selected)
 C.write_new(root/"protocol.json",{"rules":RULES,"code_sha256":hashes(),"table":table,
  "source_root":str(source_root),"bindings":{n:C.evidence(source_root/n) for n in
   ("protocol.json","preflight.json","independent_pre_gpu_review.json")},
  "cells":finite_cell_order(table),"maps":refs,"pilot":{"configuration":mode,"scene":scene,
   "map":refs[key],"query_names":selected,"expected_config":reference["config"],"expected_calibration":reference["calibration"],"image_sizes":{n:sizes[scene][n] for n in selected},
   "covers_all_map_N_and_query_HW":covered,"all_query_max_hw":[maxh,maxw]},
  "query_snapshots":{s:C.evidence(source_root/"query_inputs"/(s+".json")) for s in C.SCENES}})
 print("SCHEDULER_PREPARED",table,C.sha256(root/"protocol.json"),flush=True)
def checked_snapshots(p):
 for scene,ref in p["query_snapshots"].items():
  expected=Path(p["source_root"])/"query_inputs"/(scene+".json")
  if Path(ref["path"]).resolve()!=expected.resolve():raise ValueError("snapshot target differs")
  C.unchanged(ref)
def digest(value):
 return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def pilot_worker(root,folder):
 import torch
 from anygsloc.data.datasets import ColmapDataset
 from anygsloc.data.calibration import resolve_scene_calibration
 from anygsloc.online.runtime import Localizer
 root=Path(root);folder=Path(folder);p=source(root);require_review(root);checked_snapshots(p)
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 base=driver(p["table"]).protocol(p["source_root"]);cell=p["pilot"];scene=cell["scene"]
 s=base["scenes"][scene];before=C.snapshot(p["source_root"],scene,s)
 C.unchanged(cell["map"]);dataset=ColmapDataset(s["dataset"],load_masks=True)
 cameras={c.image_name:c for c in dataset.split("test")}
 calibration=resolve_scene_calibration(dataset.split("mapping"))
 if calibration!=cell["expected_calibration"]:raise ValueError("pilot calibration differs")
 engine=Localizer(cell["map"]["path"],keypoint_count=calibration["keypoint_count"],
   reprojection_error_px=calibration["reprojection_error_px"],mask_policy="explicit_valid_mask",device="cuda:0",seed=2026)
 def one(name):
  c=cameras[name];mask=dataset.valid_mask(c)
  if mask is None:raise ValueError("missing query mask")
  result=engine.localize(dataset.load_image(c),fov_x=c.fov_x,fov_y=c.fov_y,valid_mask=mask)
  pose=result["pose"].pose_w2c
  if not np.isfinite(pose).all():raise ValueError("nonfinite predicted pose")
  return {"name":name,"success":bool(result["success"]),"pose":pose.tolist(),
    "inliers":result["pose"].inliers.tolist(),
    "matched_keypoints_sha256":digest(result["matches"].keypoint_indices.cpu().numpy()),
    "matched_anchors_sha256":digest(result["matches"].anchor_indices.cpu().numpy())}
 for i in range(RULES["warmup_queries"]):one(cell["query_names"][i%len(cell["query_names"])])
 C.write_new(folder/"ready.json",{"pid":os.getpid()})
 deadline=time.monotonic()+RULES["child_timeout_seconds"]
 while not (folder/"go").exists():
  if time.monotonic()>deadline:raise TimeoutError("pilot barrier")
  time.sleep(.01)
 started=time.perf_counter();rows=[]
 for _ in range(RULES["pilot_repetitions"]):
  for name in cell["query_names"]:rows.append(one(name))
 ended=time.perf_counter()
 if C.snapshot(p["source_root"],scene,s)!=before:raise ValueError("pilot input changed")
 config={k:v for k,v in engine.config.items() if k!="actual_cpu_threads"}
 if config!=cell["expected_config"]:raise ValueError("pilot config differs")
 C.write_new(folder/"result.json",{"rows":rows,"start":started,"end":ended,
  "protocol_sha256":C.sha256(root/"protocol.json"),"map_sha256":cell["map"]["sha256"],
  "query_inputs_sha256":before,"configuration":cell["configuration"],"scene":scene,
  "config":config,"calibration":calibration,"query_names":cell["query_names"],
  "elapsed":ended-started,"torch_max_reserved_bytes":torch.cuda.max_memory_reserved(0),
  "torch_threads":torch.get_num_threads(),"seed":2026,"queries":len(rows)})
class NvmlMonitor:
 class Info(ctypes.Structure):
  _fields_=[("total",ctypes.c_ulonglong),("free",ctypes.c_ulonglong),("used",ctypes.c_ulonglong)]
 def __init__(self):
  self.lib=ctypes.CDLL("libnvidia-ml.so.1");self.handle=ctypes.c_void_p()
  if self.lib.nvmlInit_v2()!=0 or self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0:raise ValueError("NVML unavailable")
  self.stop_event=threading.Event();self.samples=[];self.error=None
 def read(self):
  info=self.Info()
  if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(info))!=0:raise ValueError("NVML sample failed")
  return int(info.used)
 def loop(self):
  try:
   while not self.stop_event.is_set():
    self.samples.append(self.read());self.stop_event.wait(RULES["nvml_poll_seconds"])
  except Exception as e:self.error=str(e)
 def __enter__(self):
  self.samples.append(self.read());self.thread=threading.Thread(target=self.loop);self.thread.start();return self
 def __exit__(self,*args):
  self.stop_event.set();self.thread.join();self.samples.append(self.read());self.lib.nvmlShutdown()
def managed_group(root,folders):
 children=[];handles=[];deadline=time.monotonic()+RULES["child_timeout_seconds"]
 try:
  for folder in folders:
   folder.mkdir(parents=True,exist_ok=False)
   log=(folder/"stdout.log").open("w");handles.append(log)
   command=[sys.executable,"-u","-m",__name__ if __name__!="__main__" else "experiments.cambridge_mainline_ablations.accuracy_scheduler",
    "worker","--root",str(root),"--folder",str(folder)]
   children.append(subprocess.Popen(command,env=worker_env(),stdout=log,stderr=subprocess.STDOUT))
  while not all((f/"ready.json").exists() for f in folders):
   if any(c.poll() is not None for c in children):raise RuntimeError("pilot child failed before ready")
   if time.monotonic()>deadline:raise TimeoutError("pilot ready")
   time.sleep(.05)
  for folder in folders:(folder/"go").write_text("start")
  for child in children:
   if child.wait(timeout=max(1,deadline-time.monotonic()))!=0:raise RuntimeError("pilot child failed")
  return [C.read(f/"result.json") for f in folders]
 finally:
  for child in children:
   if child.poll() is None:child.terminate()
  for child in children:
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  for h in handles:h.close()
def validate_pilot_record(record,p,protocol_sha):
 cell=p["pilot"];expected=cell["query_names"]*RULES["pilot_repetitions"]
 identity={"protocol_sha256":protocol_sha,"map_sha256":cell["map"]["sha256"],
  "query_inputs_sha256":p["query_snapshots"][cell["scene"]]["sha256"],
  "configuration":cell["configuration"],"scene":cell["scene"],"config":cell["expected_config"],
  "calibration":cell["expected_calibration"],"query_names":cell["query_names"],"seed":2026,"torch_threads":4}
 for key,value in identity.items():
  if record.get(key)!=value:raise ValueError("pilot identity differs: "+key)
 if not expected or record.get("queries")!=len(expected) or [r["name"] for r in record.get("rows",[])]!=expected:raise ValueError("pilot query coverage differs")
 for row in record["rows"]:
  pose=np.asarray(row["pose"])
  if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError("invalid pilot pose")
  if type(row["success"]) is not bool or not isinstance(row["inliers"],list):raise ValueError("invalid pilot result")
 for k in ("start","end","elapsed","torch_max_reserved_bytes"):
  if not math.isfinite(record[k]):raise ValueError("nonfinite pilot timing/memory")
 if record["end"]<=record["start"] or record["elapsed"]<=0 or record["torch_max_reserved_bytes"]<=0:raise ValueError("invalid pilot telemetry")
 if not math.isclose(record["end"]-record["start"],record["elapsed"],abs_tol=1e-8):raise ValueError("pilot interval differs")
 return record
PILOT_FILES=[str(Path("pilot")/name/"result.json") for name in ("serial0","serial1","parallel0","parallel1")]+["pilot/telemetry.json"]
def pilot_evidence(root):
 root=Path(root)
 return {name:C.evidence(root/name) for name in PILOT_FILES if (root/name).is_file()}
def checked_pilot_evidence(root,p,result):
 root=Path(root);refs=result.get("evidence",{})
 if not set(refs)<=set(PILOT_FILES):raise ValueError("unknown pilot evidence")
 for name,ref in refs.items():
  if Path(ref["path"]).resolve()!=(root/name).resolve():raise ValueError("pilot evidence path differs")
  C.unchanged(ref)
 if result["concurrency"]==2 and set(refs)!=set(PILOT_FILES):raise ValueError("parallel evidence incomplete")
 records=[]
 for name in PILOT_FILES[:4]:
  if name in refs:records.append(validate_pilot_record(C.read(root/name),p,C.sha256(root/"protocol.json")))
 if set(refs)==set(PILOT_FILES):
  telemetry=C.read(root/"pilot/telemetry.json")
  checked=decide(records[:2],records[2:],max(telemetry["whole_gpu_bytes"]),telemetry["baseline_bytes"],
   telemetry["free_cores"],telemetry["ram_available"],p["pilot"]["covers_all_map_N_and_query_HW"])
  for key,value in checked.items():
   if result.get(key)!=value:raise ValueError("pilot decision differs from raw evidence")
 return refs
def same_predictions(a,b):
 if not a["rows"] or not b["rows"] or len(a["rows"])!=len(b["rows"]):return False
 for x,y in zip(a["rows"],b["rows"]):
  for key in ("name","success","inliers","matched_keypoints_sha256","matched_anchors_sha256"):
   if x[key]!=y[key]:return False
  if not np.isfinite(x["pose"]).all() or not np.isfinite(y["pose"]).all():return False
  if not np.allclose(x["pose"],y["pose"],atol=RULES["pose_atol"],rtol=RULES["pose_rtol"]):return False
 return True
def decide(serial,parallel,peak_bytes,baseline_bytes,free_cores,ram,covered=True):
 consistent=all(same_predictions(serial[i],parallel[i]) for i in range(2))
 serial_time=sum(v["elapsed"] for v in serial)
 parallel_time=max(v["end"] for v in parallel)-min(v["start"] for v in parallel)
 gain=serial_time/parallel_time-1 if parallel_time>0 else -1
 reserved_bound=baseline_bytes+RULES["context_reserve_bytes"]+sum(v["torch_max_reserved_bytes"] for v in parallel)
 gates={"resource_coverage":covered,"predictions_equal":consistent,
  "sampled_whole_gpu_peak":peak_bytes<RULES["memory_limit_bytes"],
  "conservative_reserved_bound":reserved_bound<RULES["memory_limit_bytes"],
  "cpu_headroom":free_cores>=RULES["minimum_free_cpu_cores"],
  "ram_headroom":ram>=RULES["minimum_available_ram_bytes"],
  "throughput_gain":gain>=RULES["minimum_throughput_gain"]}
 return {"concurrency":2 if all(gates.values()) else 1,"gates":gates,"throughput_gain":gain,
  "sampled_peak_bytes":peak_bytes,"conservative_reserved_bound_bytes":reserved_bound,
  "nvml_sampling_seconds":RULES["nvml_poll_seconds"]}
def pilot(root):
 root=Path(root);p=source(root);require_review(root);checked_snapshots(p)
 if (root/"pilot_result.json").exists():raise FileExistsError("pilot already completed")
 free=cpu_free();ram=memory_available();covered=p["pilot"]["covers_all_map_N_and_query_HW"]
 try:
  if not covered or free<RULES["minimum_free_cpu_cores"] or ram<RULES["minimum_available_ram_bytes"]:
   result={"concurrency":1,"reason":"pre-execution resource coverage/headroom gate"}
  else:
   if gpu_process_ids():raise RuntimeError("GPU0 compute process still active before pilot")
   with NvmlMonitor() as monitor:
    baseline=monitor.read()
    if baseline>1024**3:raise RuntimeError("GPU0 is not idle enough for resource pilot")
    serial=[]
    for i in range(2):serial+=managed_group(root,[root/"pilot"/("serial"+str(i))])
    predicted=baseline+RULES["context_reserve_bytes"]+sum(v["torch_max_reserved_bytes"] for v in serial)
    if predicted>=RULES["memory_limit_bytes"]:raise RuntimeError("serial memory evidence rules out parallel pilot")
    if cpu_free()<RULES["minimum_free_cpu_cores"] or memory_available()<RULES["minimum_available_ram_bytes"]:raise RuntimeError("headroom changed before parallel pilot")
    parallel=managed_group(root,[root/"pilot"/"parallel0",root/"pilot"/"parallel1"])
   if monitor.error:raise RuntimeError(monitor.error)
   for record in serial+parallel:validate_pilot_record(record,p,C.sha256(root/"protocol.json"))
   free=min(free,cpu_free());ram=min(ram,memory_available())
   result=decide(serial,parallel,max(monitor.samples),baseline,free,ram,covered)
   C.write_new(root/"pilot"/"telemetry.json",{"whole_gpu_bytes":monitor.samples,"baseline_bytes":baseline,"free_cores":free,"ram_available":ram})
 except Exception as e:
  result={"concurrency":1,"reason":type(e).__name__+": "+str(e)}
 result.update(protocol_sha256=C.sha256(root/"protocol.json"),complete=True,evidence=pilot_evidence(root),
  formal_latency_benchmark=False,predictions_not_used_to_select_method=True)
 C.write_new(root/"pilot_result.json",result);print("PILOT_RESULT",json.dumps(result),flush=True)
def checked_policy(root):
 root=Path(root);p=source(root);require_review(root);r=C.read(root/"pilot_result.json")
 if not r.get("complete") or r["protocol_sha256"]!=C.sha256(root/"protocol.json") or r["concurrency"] not in (1,2):raise ValueError("invalid execution policy")
 required={"resource_coverage","predictions_equal","sampled_whole_gpu_peak","conservative_reserved_bound","cpu_headroom","ram_headroom","throughput_gain"}
 if r["concurrency"]==2 and (set(r.get("gates",{}))!=required or not all(r["gates"].values())):raise ValueError("failed parallel gate")
 checked_pilot_evidence(root,p,r)
 review=C.read(root/"independent_execution_review.json")
 if review.get("verdict")!="PASS" or review.get("protocol_sha256")!=C.sha256(root/"protocol.json") or review.get("pilot_result_sha256")!=C.sha256(root/"pilot_result.json"):raise ValueError("execution review missing/stale")
 return r
def query_command(table,source_root,mode,scene):
 if (mode,scene) not in finite_cell_order(table):raise ValueError("outside finite cell list")
 return [sys.executable,"-u","-m","experiments.cambridge_mainline_ablations."+table.lower(),
  "evaluate","--root",str(source_root),"--scene",scene,"--config",mode]
def gpu_process_ids():
 uuid=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=uuid","--format=csv,noheader,nounits"],text=True).strip()
 if not uuid.startswith("GPU-") or "\n" in uuid:raise ValueError("GPU0 identity unavailable")
 lines=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True).splitlines()
 return {int(line.split(",")[1].strip()) for line in lines if line.split(",")[0].strip()==uuid}
def current_admission(preferred,owned=()):
 foreign=gpu_process_ids()-set(owned)
 if foreign:raise RuntimeError("GPU0 foreign compute processes; wait: "+str(sorted(foreign)))
 free=cpu_free();ram=memory_available()
 if ram<8*1024**3:raise RuntimeError("insufficient RAM even for serial query")
 concurrent=preferred if free>=RULES["minimum_free_cpu_cores"] and ram>=RULES["minimum_available_ram_bytes"] else 1
 return {"concurrency":concurrent,"estimated_free_cores":free,"ram_available":ram,"foreign_gpu_pids":[]}
def mirror_status(table,value):
 # Existing repository ignore rule: outputs/. Only small derived receipts.
 path=C.REPO/"outputs/anygsloc_mainline_ablations_20261011_v2"/(table+"_query_status.json")
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".tmp")
 temp.write_text(json.dumps(C.safe(value),indent=2,allow_nan=False));os.replace(temp,path)
def run(root):
 root=Path(root);p=source(root);policy=checked_policy(root);d=driver(p["table"])
 concurrency=policy["concurrency"];src=Path(p["source_root"]);logroot=root/"query_logs";logroot.mkdir(exist_ok=False)
 owned=set();launch_lock=threading.Lock()
 initial=current_admission(concurrency);C.write_new(root/"admission/initial.json",initial)
 def execute(cell):
  mode,scene=cell;folder=src/"evaluations"/scene/mode
  if folder.exists():raise FileExistsError("existing cell requires explicit recovery review: "+str(folder))
  with (logroot/(mode+"_"+scene+".log")).open("x") as log:
   with launch_lock:
    if gpu_process_ids()-owned:raise RuntimeError("GPU0 foreign process appeared; batch paused")
    proc=subprocess.Popen(query_command(p["table"],src,mode,scene),env=worker_env(),stdout=log,stderr=subprocess.STDOUT)
    owned.add(proc.pid)
   try:rc=proc.wait(timeout=7200)
   finally:
    if proc.poll() is None:
     proc.terminate()
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
    with launch_lock:owned.discard(proc.pid)
   if rc:raise RuntimeError("query process failed: "+str(cell))
 for mode in (d.CONFIGS if p["table"]=="T5" else d.NEW):
  admission=current_admission(policy["concurrency"]);concurrency=admission["concurrency"]
  C.write_new(root/"admission"/(mode+".json"),admission)
  mirror_status(p["table"],{"stage":"running","configuration":mode,"concurrency":concurrency})
  cells=[(mode,s) for s in C.SCENES]
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:list(pool.map(execute,cells))
  d.report(src,mode)
  mirror_status(p["table"],{"stage":"five_scene_complete","configuration":mode,"concurrency":concurrency,"report":C.read(src/"reports"/(mode+".json"))})
  C.write_new(root/"completed_batches"/(mode+".json"),{"report":C.evidence(src/"reports"/(mode+".json")),
   "concurrency":concurrency,"pilot_result":C.evidence(root/"pilot_result.json")})
 if p["table"]=="T6":d.report(src,"filter_on_fused")
 all_modes=d.CONFIGS if p["table"]=="T5" else d.ALL
 C.write_new(src/"report.json",{"complete":True,"design":d.DESIGN,
  "results":{c:C.read(src/"reports"/(c+".json")) for c in all_modes},
  "execution_protocol":C.evidence(root/"protocol.json"),"execution_policy":C.evidence(root/"pilot_result.json")})
 print("ACCURACY_BATCH_COMPLETE",p["table"],flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","worker","pilot","run"))
 p.add_argument("--root",type=Path,required=True);p.add_argument("--table",choices=("T5","T6"));p.add_argument("--source-root",type=Path);p.add_argument("--folder",type=Path)
 a=p.parse_args()
 if a.stage=="prepare":prepare(a.root,a.table,a.source_root)
 elif a.stage=="worker":pilot_worker(a.root,a.folder)
 elif a.stage=="pilot":pilot(a.root)
 else:run(a.root)
if __name__=="__main__":main()

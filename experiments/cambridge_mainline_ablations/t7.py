"""Finite T7 selection ablation: four new controls per scene, GPU0 only.
No training, rendering, classic reconstruction, or changes to production kernels.
"""
from __future__ import annotations
import argparse, copy, gc, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from anygsloc.artifacts import SCHEMA, sha256, validate
from experiments.cambridge_input_matrix import driver as M

REPO = Path(__file__).resolve().parents[2]
FULL = Path("/mnt/pool/sqy/anygsloc_full_repaired_baseline_20261011_v1")
STAGE = Path("/mnt/pool/sqy/anygsloc_mechanism_stage_a_20261011_v2")
REPAIR = Path("/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v2/mechanism")
MATRIX = Path("/mnt/pool/sqy/anygsloc_input_matrix_vanilla3d_20261011_v3")
SCENES = ("ShopFacade", "OldHospital", "KingsCollege", "GreatCourt", "StMarysChurch")
NEW = ("quality_stage", "coverage_stage", "quality_topk", "tracklength_topk")
REUSED = ("full", "native_compact", "spatial")
ALL = ("full", "quality_stage", "coverage_stage", "native_compact", "spatial", "quality_topk", "tracklength_topk")
DESIGN = {
 "schema": "mainline_t7_repaired_vanilla3d_v1", "scene_order": list(SCENES),
 "input": "pure GS vanilla3D; unmasked SP2048 then V2; no mapping semantics",
 "query": "rectified processed RGB, semantic masks on, seed2026, unchanged scene-calibrated PoseLib",
 "new": list(NEW), "reuse": list(REUSED), "total_cells": 35, "new_evaluations": 20,
 "selection": "same Full pool; native v2 unchanged; natural stage sizes; alternatives exact per-scene native N",
 "quality_topk": "descending authentic anchor_matchability, ascending source row tie-break",
 "tracklength_topk": "descending distinct physical-camera support, ascending source row tie-break",
 "stage_rows": "archived pre-cleanup precision and precision+matching_completion; final native includes existing cleanup",
 "map_change": "rows only; XYZ and descriptors unchanged", "uses_test_for_selection": False,
 "timing": "accuracy telemetry only; concurrent other-GPU work; no formal timing",
 "test_status": "benchmark-informed development, not blind testing",
}

def read(p): return json.loads(Path(p).read_text())
def load(p): return torch.load(p, map_location="cpu", weights_only=False)
def safe(v):
 if isinstance(v, dict): return {k:safe(x) for k,x in v.items()}
 if isinstance(v, (list, tuple)): return [safe(x) for x in v]
 if isinstance(v, (float, np.floating)) and not math.isfinite(v): return None
 if isinstance(v, np.generic): return v.item()
 return v
def write_new(p, value):
 p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
 with p.open("x") as f: json.dump(safe(value),f,indent=2,allow_nan=False)
def evidence(p, expected=None):
 p=Path(p).resolve()
 h=sha256(p)
 if expected is not None and h!=expected: raise ValueError("Source hash differs: "+str(p))
 return {"path":str(p),"sha256":h,"bytes":p.stat().st_size}
def unchanged(ref): return evidence(ref["path"],ref["sha256"])

def checked_files(folder, refs, names):
 folder=Path(folder).resolve()
 if set(refs)!=set(names):raise ValueError("evidence file names differ")
 for name in names:
  ref=refs[name]
  if Path(ref["path"]).resolve()!=(folder/name).resolve():raise ValueError("evidence file path differs: "+name)
  unchanged(ref)
 return refs

def check_full_review(base=FULL):
 base=Path(base);review=read(base/"independent_final_review.json")
 if review.get("verdict")!="PASS" or review.get("complete") is not True:raise ValueError("Full source not accepted")
 for key,name in (("protocol_sha256","protocol.json"),("report_sha256","report.json")):
  if review.get(key)!=sha256(base/name):raise ValueError("Full review is stale: "+key)
 return review

def check_frozen_inventory(root, ready):
 root=Path(root)
 for key,folder,name in (("map_manifests","maps","manifest.json"),("query_snapshots","query_inputs",None)):
  refs=ready.get(key,{})
  if set(refs)!=set(SCENES):raise ValueError("incomplete frozen "+key)
  for scene,ref in refs.items():
   target=root/folder/scene/name if name else root/folder/(scene+".json")
   if Path(ref["path"]).resolve()!=target.resolve():raise ValueError("frozen inventory path differs")
   unchanged(ref)
   if key=="map_manifests":
    maps=read(target)
    if set(maps)!=set(NEW):raise ValueError("finite map inventory differs")
    for config,item in maps.items():
     dest=root/"maps"/scene/config/"deployment_map.pt"
     if Path(item["path"]).resolve()!=dest.resolve() or Path(item["rows"]["path"]).resolve()!=dest.with_suffix(".source_rows.npy").resolve():raise ValueError("map inventory path differs")

def require_review(root):
 root=Path(root);review=read(root/"independent_pre_gpu_review.json")
 if review.get("verdict")!="PASS" or review.get("protocol_sha256")!=sha256(root/"protocol.json") or review.get("preflight_sha256")!=sha256(root/"preflight.json"):raise ValueError("independent review missing or stale")
 return review
def mapref(p):
 state=validate(load(p)); ref=evidence(p);ref["count"]=len(state["anchor_xyz"]);return ref

def code_hashes():
 paths=list((REPO/"anygsloc").rglob("*.py"))
 paths+=list((REPO/"experiments/cambridge_input_matrix").glob("*.py"))
 paths += [Path(__file__).resolve(), REPO/"tests/test_cambridge_t7_repaired.py"]
 return {str(p.relative_to(REPO)):sha256(p) for p in sorted(set(paths))}

def check_subset(full, subset, rows):
 rows=np.asarray(rows,dtype=np.int64);n=len(full["anchor_xyz"])
 if rows.ndim!=1 or len(np.unique(rows))!=len(rows) or len(rows)==0 or rows.min()<0 or rows.max()>=n:
  raise ValueError("invalid source rows")
 for k in ("anchor_xyz","anchor_features"):
  if not torch.equal(full[k][torch.from_numpy(rows.copy())],subset[k]):
   raise ValueError("subset changed "+k)
 return rows

def validate_csr(state):
 if state.get("provenance",{}).get("uses_test_queries") is not False: raise ValueError("non-mapping provenance")
 if state.get("provenance",{}).get("uses_source_mapping_rgb") is not False: raise ValueError("not pure GS")
 n=len(state["anchor_xyz"]);csr=state["projective_anchor_observations"]
 off=np.asarray(csr["observation_offsets"],dtype=np.int64)
 q=np.asarray(csr["query_indices"],dtype=np.int64);k=np.asarray(csr["keypoint_indices"],dtype=np.int64)
 if len(off)!=n+1 or off[0]!=0 or off[-1]!=len(q) or len(q)!=len(k) or (np.diff(off)<1).any():
  raise ValueError("invalid CSR offsets")
 if (q<0).any() or (q>=len(state["v6_mapping_query_names"])).any() or (k<0).any(): raise ValueError("invalid CSR indices")
 for a,b in zip(off[:-1],off[1:]):
  if len(np.unique(q[a:b]))!=b-a: raise ValueError("duplicate physical camera")
 quality=np.asarray(state["anchor_matchability"],dtype=np.float64)
 if len(quality)!=n or not np.isfinite(quality).all():raise ValueError("invalid authentic quality")
 return off,quality

def derive_rows(state,trace,native_rows):
 off,quality=validate_csr(state);n=len(quality)
 original=np.asarray(trace["selected_universe_ids"],dtype=np.int64)
 records=trace["trace"]
 ids=np.asarray([r["candidate_universe_id"] for r in records],dtype=np.int64)
 if len(np.unique(ids))!=len(ids) or not np.array_equal(np.sort(ids),np.sort(original)):raise ValueError("trace IDs differ")
 if len(ids)==0 or ids.min()<0 or ids.max()>=n:raise ValueError("trace outside candidate pool")
 reasons={r["primary_reason"] for r in records}
 if not reasons <= {"precision","matching_completion","observability_completion"}:raise ValueError("unknown stage")
 native_rows=np.asarray(native_rows,dtype=np.int64)
 if not np.array_equal(np.sort(np.asarray(trace["final_source_ids"])),np.sort(native_rows)):raise ValueError("final native IDs differ")
 count=len(native_rows)
 if not 0<count<=n:raise ValueError("invalid native N")
 out={
  "quality_stage":np.sort([r["candidate_universe_id"] for r in records if r["primary_reason"]=="precision"]),
  "coverage_stage":np.sort([r["candidate_universe_id"] for r in records if r["primary_reason"] in {"precision","matching_completion"}]),
  "quality_topk":np.sort(np.lexsort((np.arange(n),-quality))[:count]),
  "tracklength_topk":np.sort(np.lexsort((np.arange(n),-np.diff(off)))[:count]),
 }
 for key,rows in out.items():
  if len(rows)==0 or len(np.unique(rows))!=len(rows):raise ValueError("invalid selected "+key)
 return out

def save_rows(path,state,rows,provenance):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 rows=np.asarray(rows,dtype=np.int64)
 rowpath=path.with_suffix(".source_rows.npy")
 with rowpath.open("xb") as f:np.save(f,rows)
 out={"schema":SCHEMA,"anchor_ids":torch.arange(len(rows)),
      "anchor_xyz":state["anchor_xyz"][torch.from_numpy(rows.copy())].float().clone(),
      "anchor_features":state["anchor_features"][torch.from_numpy(rows.copy())].float().clone(),
      "provenance":dict(provenance,uses_source_mapping_rgb=False,uses_test_queries=False,
                       retained_source_rows_sha256=sha256(rowpath))}
 validate(out);check_subset(state,out,rows);torch.save(out,path)
 return dict(evidence(path),count=len(rows),rows=evidence(rowpath))

def snapshot(root,scene,s):
 from anygsloc.data.datasets import ColmapDataset
 return M.evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)

def same_snapshot(a,b):
 for key in ("mapping_registry","calibration","mask_sha256","queries"):
  if a[key]!=b[key]:raise ValueError("query snapshot differs: "+key)

def manifest_check(folder,ref,s,reference=None):
 man=read(Path(folder)/"manifest.json")
 if man["map_sha256"]!=ref["sha256"] or man["queries"]!=s["test_names"]:raise ValueError("map/query manifest differs")
 expect={"dataset":s["dataset"],"images":"processed","image_domain":"rectified","split":"test","seed":2026,"rectification_contract":"opencv_same_k"}
 for k,v in expect.items():
  if man.get(k)!=v:raise ValueError("manifest setting differs: "+k)
 if man["config"]["mask_policy"]!="explicit_valid_mask":raise ValueError("query semantics not enabled")
 if reference is not None:
  for k in ("dataset","images","image_domain","split","seed","queries","calibration","config","rectification_contract"):
   if man[k]!=reference[k]:raise ValueError("reference condition differs: "+k)
 return man

def prepare(root):
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 fp=read(FULL/"protocol.json"); ap=read(STAGE/"protocol.json")
 check_full_review()
 sources={};scenes={};cells={}
 for scene in SCENES:
  repaired=scene=="StMarysChurch"
  s=fp["scenes"][scene] if repaired else ap["scenes"][scene]
  scenes[scene]={k:copy.deepcopy(s[k]) for k in ("dataset","registry","mapping_names","test_names","mask_path","mask_sha256","prior")}
  if set(s["mapping_names"])&set(s["test_names"]):raise ValueError("mapping/test overlap")
  base=REPAIR if repaired else STAGE
  full=fp["full"]["anygsloc"] if repaired else fp["reused"][scene]["anygsloc"]["map"]
  state_path=REPAIR/"new_full/anygsloc/selector_state.pt" if repaired else Path(read(STAGE/"validation"/(scene+".json"))["state"]["path"])
  trace_path=REPAIR/"new_native_trace.pt" if repaired else STAGE/"selection"/scene/"native_trace.pt"
  sources[scene]={"state":evidence(state_path),"trace":evidence(trace_path),"full":dict(evidence(full["path"],full["sha256"]),count=full["count"])}
  cells[scene]={}
  for config,old in (("native_compact","new_native"),("spatial","new_spatial_N")):
   d=base/"maps"/old if repaired else base/"maps"/scene/old
   ref=read(d/"selection.json");ref=dict(evidence(ref["path"],ref["sha256"]),count=ref["count"],rows=evidence(Path(ref["path"]).with_suffix(".source_rows.npy")))
   folder=base/"evaluations"/old if repaired else base/"evaluations"/scene/old
   cells[scene][config]={"map":ref,"evaluation_folder":str(folder),"files":{n:evidence(folder/n) for n in ("results.json","summary.json","manifest.json","attestation.json")}}
  folder=FULL/"evaluations/anygsloc" if repaired else Path(fp["reused"][scene]["anygsloc"]["evaluation"]["results.json"]["path"]).parent
  cells[scene]["full"]={"map":sources[scene]["full"],"evaluation_folder":str(folder),"files":{n:evidence(folder/n) for n in ("results.json","summary.json","manifest.json")}}
  old_snapshot=FULL/"query_inputs"/(scene+".json") if repaired else MATRIX/"query_inputs"/(scene+".json")
  sources[scene]["old_snapshot"]=evidence(old_snapshot)
  reference=manifest_check(folder,cells[scene]["full"]["map"],scenes[scene])
  for config in ("native_compact","spatial"):manifest_check(cells[scene][config]["evaluation_folder"],cells[scene][config]["map"],scenes[scene],reference)
 protocol={"design":DESIGN,"code_sha256":code_hashes(),"scenes":scenes,"sources":sources,"reused":cells,
           "source_evidence":[evidence(FULL/"protocol.json"),evidence(FULL/"report.json"),evidence(FULL/"independent_final_review.json"),evidence(STAGE/"protocol.json"),evidence(REPAIR/"protocol.json")]}
 write_new(root/"protocol.json",protocol)
 for scene in SCENES:
  snapshot(root,scene,scenes[scene])
  same_snapshot(read(root/"query_inputs"/(scene+".json")),read(sources[scene]["old_snapshot"]["path"]))
  state=load(sources[scene]["state"]["path"]);full_state=validate(load(sources[scene]["full"]["path"]))
  for k in ("anchor_xyz","anchor_features"):
   if not torch.equal(state[k],full_state[k]):raise ValueError("selector state differs from production Full")
  if state["v6_mapping_query_names"]!=scenes[scene]["mapping_names"]:raise ValueError("mapping camera registry differs")
  native_rows=np.load(cells[scene]["native_compact"]["map"]["rows"]["path"])
  for c in ("native_compact","spatial"):
   ref=cells[scene][c]["map"];check_subset(state,load(ref["path"]),np.load(ref["rows"]["path"]))
  if cells[scene]["spatial"]["map"]["count"]!=len(native_rows):raise ValueError("spatial budget differs")
  rows=derive_rows(state,load(sources[scene]["trace"]["path"]),native_rows)
  refs={}
  for config,selected in rows.items():
   refs[config]=save_rows(root/"maps"/scene/config/"deployment_map.pt",state,selected,
    {"experiment":DESIGN["schema"],"configuration":config,"protocol_sha256":sha256(root/"protocol.json"),
     "source_full_sha256":sources[scene]["full"]["sha256"],"source_state_sha256":sources[scene]["state"]["sha256"],
     "native_N":len(native_rows),"source_camera_registry":scenes[scene]["registry"]})
  write_new(root/"maps"/scene/"manifest.json",refs)
  print("prepared",scene,{k:v["count"] for k,v in refs.items()},flush=True)
  del state,full_state;gc.collect()
 write_new(root/"preflight.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
  "new_maps":20,"reused_cells":15,"source_subset_exact":True,"cpu_only":not torch.cuda.is_initialized(),
  "map_manifests":{scene:evidence(root/"maps"/scene/"manifest.json") for scene in SCENES},
  "query_snapshots":{scene:evidence(root/"query_inputs"/(scene+".json")) for scene in SCENES}})
 print("CPU_PREPARED",root,flush=True)

def protocol(root):
 root=Path(root);p=read(root/"protocol.json")
 if p["design"]!=DESIGN or p["code_sha256"]!=code_hashes():raise ValueError("frozen design/code changed")
 ready=read(root/"preflight.json")
 if ready.get("complete") is not True or ready["protocol_sha256"]!=sha256(root/"protocol.json"):raise ValueError("preflight missing")
 check_full_review()
 for ref in p["source_evidence"]:unchanged(ref)
 check_frozen_inventory(root,ready)
 return p

def independent_metrics(rows,names):
 if [r["image_name"] for r in rows]!=names or len(set(names))!=len(names):raise ValueError("missing/duplicate queries")
 t=np.array([r["translation_error_cm"] if r["success"] else np.inf for r in rows],dtype=float)
 a=np.array([r["rotation_error_deg"] if r["success"] else np.inf for r in rows],dtype=float)
 if np.isnan(t).any() or np.isnan(a).any():raise ValueError("NaN query metric")
 return {"query_count":len(rows),"median_TE_cm":float(np.median(t)),"median_RE_deg":float(np.median(a)),
  "mean_TE_cm":float(np.mean(t)),"P95_TE_cm":float(np.percentile(t,95)),"R5_pct":float(100*np.mean((t<5)&(a<5))),
  "R15_pct":float(100*np.mean((t<15)&(a<5))),"failures":sum(not r["success"] for r in rows)}

def evaluate(root,scene,config):
 from anygsloc.data.datasets import ColmapDataset
 from anygsloc.evaluation.runner import main as run
 root=Path(root);p=protocol(root);require_review(root)
 if scene not in SCENES or config not in NEW:raise ValueError("outside finite T7 plan")
 s=p["scenes"][scene];ref=read(root/"maps"/scene/"manifest.json")[config]
 unchanged(ref);unchanged(ref["rows"])
 output=root/"evaluations"/scene/config
 if output.exists():raise FileExistsError(output)
 snap=snapshot(root,scene,s)
 dataset=ColmapDataset(s["dataset"],load_masks=True)
 if any(dataset.valid_mask(c) is None for c in dataset.split("test")):raise ValueError("missing semantic masks")
 run(["--dataset",s["dataset"],"--map",ref["path"],"--output",str(output),"--images","processed",
      "--image-domain","rectified","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k",
      "--seed","2026","--device","cuda:0"])
 if snapshot(root,scene,s)!=snap:raise ValueError("query inputs changed")
 unchanged(ref);protocol(root)
 reference=read(Path(p["reused"][scene]["full"]["evaluation_folder"])/"manifest.json")
 manifest_check(output,ref,s,reference)
 raw=independent_metrics(read(output/"results.json"),s["test_names"])
 write_new(output/"attestation.json",{"protocol_sha256":sha256(root/"protocol.json"),"scene":scene,"configuration":config,
  "map_sha256":ref["sha256"],"query_inputs_sha256":snap,"device":"cuda:0","independent_metrics":raw,
  "files":{n:evidence(output/n) for n in ("results.json","manifest.json","summary.json")}})
 print("CELL_COMPLETE",scene,config,json.dumps(safe(raw)),flush=True)

def report_config(root,config):
 root=Path(root);p=protocol(root)
 if config not in ALL:raise ValueError("unknown report")
 per={}
 for scene in SCENES:
  s=p["scenes"][scene]
  if config in REUSED:
   cell=p["reused"][scene][config];ref=cell["map"];folder=Path(cell["evaluation_folder"])
   checked_files(folder,cell["files"],("results.json","summary.json","manifest.json") if config=="full" else ("results.json","summary.json","manifest.json","attestation.json"))
  else:
   ref=read(root/"maps"/scene/"manifest.json")[config];folder=root/"evaluations"/scene/config
   att=read(folder/"attestation.json")
   if att["protocol_sha256"]!=sha256(root/"protocol.json") or att["map_sha256"]!=ref["sha256"] or att["scene"]!=scene or att["configuration"]!=config:raise ValueError("cell attestation differs")
   if att["query_inputs_sha256"]!=sha256(root/"query_inputs"/(scene+".json")):raise ValueError("snapshot binding differs")
   checked_files(folder,att["files"],("results.json","summary.json","manifest.json"))
  unchanged(ref)
  manifest_check(folder,ref,s,read(Path(p["reused"][scene]["full"]["evaluation_folder"])/"manifest.json"))
  v=independent_metrics(read(folder/"results.json"),s["test_names"]);v.update(points=ref["count"],bytes=ref["bytes"])
  summary=read(folder/"summary.json")
  for our,their in (("median_TE_cm","median_te_cm"),("median_RE_deg","median_ae_deg"),("R5_pct","recall_5cm_5deg_percent"),("R15_pct","recall_15cm_5deg_percent")):
   old=summary[their]
   if old is None and not math.isfinite(v[our]):continue
   if old is None or not np.isclose(v[our],old,rtol=1e-9,atol=1e-8):raise ValueError("independent aggregate differs")
  per[scene]=v
 macro={k:float(np.mean([v[k] for v in per.values()])) for k in ("median_TE_cm","median_RE_deg","mean_TE_cm","P95_TE_cm","R5_pct","R15_pct","points")}
 macro.update(total_points=sum(v["points"] for v in per.values()),total_bytes=sum(v["bytes"] for v in per.values()),
              queries=sum(v["query_count"] for v in per.values()),failures=sum(v["failures"] for v in per.values()))
 out={"complete":True,"configuration":config,"scene_count":5,"per_scene":per,"scene_equal":macro,
      "protocol_sha256":sha256(root/"protocol.json"),"independent_raw_result_recomputation":True}
 write_new(root/"reports"/(config+".json"),out)
 print("FIVE_SCENE_COMPLETE",config,json.dumps(safe(macro)),flush=True)
 return out

def run_all(root):
 root=Path(root);protocol(root)
 require_review(root)
 for config in NEW:
  for scene in SCENES:
   folder=root/"evaluations"/scene/config
   if folder.exists():
    if not (folder/"attestation.json").is_file():raise ValueError("incomplete existing cell; investigate before retry")
   else:evaluate(root,scene,config)
  if not (root/"reports"/(config+".json")).exists():report_config(root,config)
 for config in REUSED:
  if not (root/"reports"/(config+".json")).exists():report_config(root,config)
 result={c:read(root/"reports"/(c+".json")) for c in ALL}
 write_new(root/"report.json",{"complete":True,"design":DESIGN,"results":result,"new_evaluations":20,"reused_evaluations":15})
 print("T7_COMPLETE",root,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("stage",choices=("prepare","evaluate","report","run"))
 p.add_argument("--root",type=Path,required=True);p.add_argument("--scene",choices=SCENES);p.add_argument("--config",choices=ALL)
 a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.stage=="prepare":prepare(a.root)
 elif a.stage=="evaluate":evaluate(a.root,a.scene,a.config)
 elif a.stage=="report":report_config(a.root,a.config)
 else:run_all(a.root)
if __name__=="__main__":main()

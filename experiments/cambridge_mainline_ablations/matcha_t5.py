"""MAtCha Full T5 adapter: identical frozen T5 numerical code, new source bindings.

Per-scene prepare/build/seal/evaluate, followed by a strict five-scene collection.
Source publication remains separate from planning.
No training, rendering, new Full reconstruction, or Compact selection occurs here.
"""
from __future__ import annotations
import argparse,copy,json,os,types
from pathlib import Path
import torch
from experiments.cambridge_mainline_ablations import t5 as K,t7 as C
BASE=Path("/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2/T5")
CONFIGS=K.CONFIGS
MATCHA_CONDITION="matcha_v2"
UPSTREAM_SCHEMA="independent_matcha_v2_five_scene_prior_acceptance_v1"
UPSTREAM_CASE_SCHEMA="independent_matcha_v2_prior_case_acceptance_v1"
DESIGN=copy.deepcopy(K.DESIGN)
DESIGN.update(schema="matcha_t5_common_geometry_full_v1",prior="MAtCha; accepted T8 Full candidate pool",
 production_full="Common-support ray is not the unmodified MAtCha production Full row",
 cross_prior="Same mapping/query protocol; separate common-support IDs and natural N inside each prior.",
 source_dependency="T8 owner supplies accepted new pure-GS SP2048 then V2 cache, Full and original-row CSR; repaired StMary training prior required.",
 reuse_mathematics="Exact original T5 function code objects and helpers; no mathematical changes.")
def hashes():
 h=K.hashes()
 for rel in ("experiments/cambridge_mainline_ablations/matcha_t5.py","tests/test_matcha_t5_contract.py"):
  h[rel]=C.sha256(C.REPO/rel)
 return h
def cpu_only():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("CPU-only preparation requires no visible CUDA devices")
def validate_index(index):
 schema=index.get("schema")
 if schema not in ("matcha_t5_source_index_v1","matcha_t5_scene_source_index_v1") or index.get("prior_label")!="MAtCha":raise ValueError("wrong source-index family")
 if index.get("condition")!=MATCHA_CONDITION:raise ValueError("approved MAt condition required")
 if schema=="matcha_t5_scene_source_index_v1":
  scene=index.get("scene")
  if scene not in C.SCENES or index.get("scope")!="single_scene":raise ValueError("single-scene scope required")
  names=(scene,)
 else:names=C.SCENES
 if set(index.get("cases",{}))!=set(names):raise ValueError("accepted scene roots differ from scope")
 if len(set(str(Path(x).resolve()) for x in index["cases"].values()))!=len(names):raise ValueError("duplicate scene root")
 if set(index.get("source_index_review",{}))!={"path","sha256"}:raise ValueError("independent source-index review required")
 if set(index.get("upstream_source_review",{}))!={"path","sha256"}:raise ValueError("direct upstream PLY review required")
 if set(index.get("downstream_reviews",{}))!=set(names):raise ValueError("downstream bindings differ from scope")
 return names
def checked_index(path):
 path=Path(path);index=C.read(path);validate_index(index)
 ref=index["source_index_review"];C.unchanged(ref);review=C.read(ref["path"])
 # The separate review binds the payload excluding its own pointer, avoiding a hash cycle.
 payload={k:v for k,v in index.items() if k!="source_index_review"}
 import hashlib
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 if review.get("verdict")!="PASS" or review.get("complete") is not True or review.get("source_payload_sha256")!=digest or review.get("scene_scope")!=list(validate_index(index)):
  raise ValueError("source-index review incomplete or scope/payload differs")
 if "StMarysChurch" in index["cases"] and review.get("stmary_training_camera_repair_verified") is not True:
  raise ValueError("source review missing/stale or repaired StMary training unverified")
 C.unchanged(index["upstream_source_review"])
 for scene,case in index["cases"].items():
  refs=index["downstream_reviews"][scene]
  if set(refs)!={"protocol.json","independent_final_review.json"}:raise ValueError("downstream evidence names differ")
  C.checked_files(Path(case),refs,("protocol.json","independent_final_review.json"))
 return index
def accepted_case(case,scene,condition,upstream_ref):
 if condition!=MATCHA_CONDITION:raise ValueError("approved MAt condition required")
 case=Path(case).resolve();p=C.read(case/"protocol.json");r=C.read(case/"independent_final_review.json");summary=C.read(case/"maps/summary.json")
 if p["scene_name"]!=scene or p["condition"]!=condition or p["mapping_semantics"] is not False or p["query_semantics"] is not True:raise ValueError("case identity or mapping/query policy differs")
 if p["prior"].get("gaussian_type")!="2dgs" or p["prior"].get("sh_degree")!=3 or p["prior"].get("background")!="white":raise ValueError("MAt prior must be fixed 2dgs SH3 white")
 if r.get("verdict")!="PASS" or not r.get("complete") or r.get("scene")!=scene or r.get("condition")!=condition or r.get("protocol_sha256")!=C.sha256(case/"protocol.json"):raise ValueError("case review missing/stale")
 if not summary.get("complete") or summary["protocol_sha256"]!=C.sha256(case/"protocol.json"):raise ValueError("map summary stale")
 expected={
  "state":(case/"selector_state.pt",summary["selector_state_sha256"]),
  "cache":(case/"observations/pure_gs.pt",summary["cache_sha256"]),
  "full":(case/"maps/Full/deployment_map.pt",summary["Full"]["sha256"]),
  "ply":(Path(p["prior"]["ply"]),p["prior"]["ply_sha256"])}
 if Path(summary["Full"]["path"]).resolve()!=expected["full"][0]:raise ValueError("Full path substituted")
 refs={k:C.evidence(path,h) for k,(path,h) in expected.items()};refs["full"]["count"]=summary["Full"]["count"]
 for name in ("protocol.json","maps/summary.json"):
  ref=C.evidence(case/name)
  if r.get("evidence_sha256",{}).get(ref["path"])!=ref["sha256"]:raise ValueError("case review does not bind source inventory")
 C.unchanged(upstream_ref);upstream=C.read(upstream_ref["path"])
 if upstream.get("verdict")!="PASS" or upstream.get("complete") is not True:raise ValueError("upstream source review not complete/accepted")
 if upstream.get("schema")==UPSTREAM_CASE_SCHEMA:
  if upstream.get("scope")!="single_scene" or upstream.get("scene_names")!=[scene] or upstream.get("count")!=1 or set(upstream.get("cases",{}))!={condition+"/"+scene}:raise ValueError("upstream single-scene scope differs")
 elif upstream.get("schema")==UPSTREAM_SCHEMA:
  if set(upstream.get("cases",{}))!={condition+"/"+n for n in C.SCENES}:raise ValueError("upstream five-scene scope incomplete")
 else:raise ValueError("upstream source schema differs")
 if Path(upstream_ref["path"]).resolve()!=Path(p["upstream_review"]["path"]).resolve() or upstream_ref["sha256"]!=p["upstream_review"]["sha256"]:raise ValueError("upstream source review substituted")
 source=upstream.get("cases",{}).get(condition+"/"+scene,{})
 if source.get("verdict")!="PASS" or source.get("complete") is not True or source.get("scene")!=scene or source.get("condition")!=condition:raise ValueError("direct PLY source case not accepted")
 prior=source.get("prior",{})
 if Path(prior.get("ply","")).resolve()!=Path(p["prior"]["ply"]).resolve():raise ValueError("reviewed PLY path differs")
 for key in ("ply_sha256","gaussian_type","sh_degree","background"):
  if prior.get(key)!=p["prior"][key]:raise ValueError("reviewed PLY identity differs: "+key)
 # PLY is directly bound by the new independent upstream source receipt;
 # state/cache/Full remain directly bound by the original downstream final review.
 reviews={"state":r,"cache":r,"full":r,"ply":source}
 for name,ref in refs.items():
  if reviews[name].get("evidence_sha256",{}).get(ref["path"])!=ref["sha256"]:raise ValueError("independent review does not bind source artifact: "+name)
 refs["old_snapshot"]=C.evidence(p["evaluation_snapshot_reference"])
 refs["case_protocol"]=C.evidence(case/"protocol.json");refs["case_review"]=C.evidence(case/"independent_final_review.json")
 refs["case_maps"]=C.evidence(case/"maps/summary.json");refs["upstream_review"]=C.evidence(p["upstream_review"]["path"],p["upstream_review"]["sha256"])
 return p,refs
def prepare(root,index_path):
 cpu_only();root=Path(root)
 if root.exists():raise FileExistsError(root)
 index=checked_index(index_path);names=validate_index(index);baseline=K.protocol(BASE);scenes={};sources={}
 if len(names)==1 and root.name!=names[0]:raise ValueError("case directory differs from single-scene scope")
 for scene in names:
  p,refs=accepted_case(index["cases"][scene],scene,index["condition"],index["upstream_source_review"])
  src=baseline["scenes"][scene]
  for key in ("dataset","registry","mapping_names","test_names","mask_path","mask_sha256"):
   if p["scene"][key]!=src[key]:raise ValueError("cross-prior mapping/query condition differs: "+scene+" "+key)
  C.same_snapshot(C.read(refs["old_snapshot"]["path"]),C.read(BASE/"query_inputs"/(scene+".json")))
  scenes[scene]=copy.deepcopy(src);scenes[scene]["prior"]=copy.deepcopy(p["prior"]);sources[scene]=refs
 C.write_new(root/"protocol.json",dict(design=DESIGN,scene_scope=list(names),new_evaluations=3*len(names),code_sha256=hashes(),scenes=scenes,sources=sources,
  full_references=baseline["full_references"],query_reference_only_prior="vanilla3D",
  reference_protocol=C.evidence(BASE/"protocol.json"),source_index=C.evidence(index_path),source_index_review=index["source_index_review"]))
 for scene in names:
  C.snapshot(root,scene,scenes[scene])
  C.same_snapshot(C.read(root/"query_inputs"/(scene+".json")),C.read(BASE/"query_inputs"/(scene+".json")))
 C.write_new(root/"prepared.json",dict(complete=True,protocol_sha256=C.sha256(root/"protocol.json"),cuda_initialized=torch.cuda.is_initialized(),
  query_snapshots={s:C.evidence(root/"query_inputs"/(s+".json")) for s in names}))
 print("MATCHA_T5_PREPARED",C.sha256(root/"protocol.json"),flush=True)
def protocol(root):
 root=Path(root);p=C.read(root/"protocol.json");ready=C.read(root/"prepared.json")
 if p["design"]!=DESIGN or p["code_sha256"]!=hashes():raise ValueError("MAt T5 code/design changed")
 if not ready.get("complete") or ready["protocol_sha256"]!=C.sha256(root/"protocol.json"):raise ValueError("prepare missing/stale")
 C.unchanged(p["reference_protocol"]);K.protocol(BASE);C.unchanged(p["source_index"]);C.unchanged(p["source_index_review"])
 index=checked_index(p["source_index"]["path"]);names=validate_index(index)
 if p.get("scene_scope")!=list(names) or set(p["scenes"])!=set(names) or set(p["sources"])!=set(names) or p.get("new_evaluations")!=3*len(names):raise ValueError("protocol scope differs")
 if len(names)==1 and root.name!=names[0]:raise ValueError("case directory differs")
 if set(ready.get("query_snapshots",{}))!=set(names):raise ValueError("snapshot scope differs")
 for scene in names:
  for key in ("case_protocol","case_review","case_maps","upstream_review","old_snapshot"):C.unchanged(p["sources"][scene][key])
  ref=ready["query_snapshots"][scene]
  if Path(ref["path"]).resolve()!=(root/"query_inputs"/(scene+".json")).resolve():raise ValueError("snapshot path substituted")
  C.unchanged(ref)
 return p
def seal(root):
 root=Path(root);p=protocol(root);refs={}
 for scene in p["scene_scope"]:
  path=root/"geometry"/scene/"manifest.json";info=C.read(path)
  if not info.get("complete") or set(info["maps"])!=set(CONFIGS):raise ValueError("incomplete three-arm geometry")
  C.unchanged(info["common_observations"]);common=K.np.load(info["common_observations"]["path"])
  ids=K.np.asarray(common["source_rows"]);offsets=K.np.asarray(common["offsets"])
  if len(ids)!=info["common_count"] or len(offsets)!=len(ids)+1 or (K.np.diff(offsets)<3).any():raise ValueError("common CSR differs")
  source=K.validate(C.load(p["sources"][scene]["full"]["path"]))
  C.unchanged(p["sources"][scene]["full"])
  for mode,ref in info["maps"].items():
   if Path(ref["path"]).resolve()!=(root/"geometry"/scene/mode/"deployment_map.pt").resolve():raise ValueError("map path differs")
   C.unchanged(ref);state=K.validate(C.load(ref["path"]))
   if not K.np.array_equal(K.np.asarray(state["anchor_ids"]),ids):raise ValueError("common IDs differ")
   if state["provenance"]["common_observations_sha256"]!=info["common_observations"]["sha256"] or state["provenance"]["source_full_sha256"]!=p["sources"][scene]["full"]["sha256"]:raise ValueError("map provenance differs")
   if not torch.equal(source["anchor_features"][ids].float(),state["anchor_features"]):raise ValueError("descriptor differs from production Full source IDs")
  refs[scene]=C.evidence(path)
 C.write_new(root/"preflight.json",dict(complete=True,protocol_sha256=C.sha256(root/"protocol.json"),scene_scope=p["scene_scope"],
  geometry_manifests=refs,common_ids_and_descriptors_exact=True,new_maps=3*len(refs),cuda_initialized=torch.cuda.is_initialized()))
 print("MAT_T5_SCOPE_SEALED",p["scene_scope"],3*len(refs),flush=True)

def checked_geometry(root,scene,mode):
 root=Path(root);p=protocol(root);ready=C.read(root/"preflight.json")
 if not ready.get("complete") or ready["protocol_sha256"]!=C.sha256(root/"protocol.json"):raise ValueError("geometry seal missing/stale")
 if ready.get("scene_scope")!=p["scene_scope"] or ready.get("new_maps")!=3*len(p["scene_scope"]) or set(ready.get("geometry_manifests",{}))!=set(p["scene_scope"]):raise ValueError("geometry seal scope differs")
 for name,ref in ready["geometry_manifests"].items():
  if Path(ref["path"]).resolve()!=(root/"geometry"/name/"manifest.json").resolve():raise ValueError("geometry manifest path differs")
  C.unchanged(ref)
 if scene not in p["scene_scope"] or mode not in CONFIGS:raise ValueError("outside frozen scene/three-arm scope")
 info=C.read(root/"geometry"/scene/"manifest.json");C.unchanged(info["common_observations"])
 if set(info["maps"])!=set(CONFIGS):raise ValueError("three maps required")
 ref=info["maps"][mode]
 if Path(ref["path"]).resolve()!=(root/"geometry"/scene/mode/"deployment_map.pt").resolve():raise ValueError("map path differs")
 C.unchanged(ref);return ref

# Rebind only input/validation globals in an isolated dictionary. Original numerical
# code objects/helpers and original K/C module globals remain unchanged.
_scope=dict(vars(K));_scope.update(protocol=protocol,checked_geometry=checked_geometry,DESIGN=DESIGN)
def _bind(name):
 f=getattr(K,name)
 return types.FunctionType(f.__code__,_scope,f.__name__,f.__defaults__,f.__closure__)
build=_bind("build");evaluate=_bind("evaluate")

def report_collection(index_path,out):
 cpu_only();index=C.read(index_path)
 if index.get("schema")!="matcha_t5_case_collection_v1" or set(index.get("cases",{}))!=set(C.SCENES):raise ValueError("five unique scenes required for final collection")
 if len(set(str(Path(x["root"]).resolve()) for x in index["cases"].values()))!=5:raise ValueError("duplicate case root")
 results={mode:{} for mode in CONFIGS};case_evidence={}
 for scene in C.SCENES:
  item=index["cases"][scene];root=Path(item["root"]).resolve();p=protocol(root)
  if p["scene_scope"]!=[scene]:raise ValueError("collection scene differs")
  for key,name in (("protocol_sha256","protocol.json"),("preflight_sha256","preflight.json"),("review_sha256","independent_pre_gpu_review.json")):
   C.evidence(root/name,item[key])
  C.require_review(root);case_evidence[scene]=dict(item)
  for mode in CONFIGS:
   ref=checked_geometry(root,scene,mode);folder=root/"evaluations"/scene/mode;att=C.read(folder/"attestation.json")
   if att["protocol_sha256"]!=item["protocol_sha256"] or att["scene"]!=scene or att["configuration"]!=mode or att["map_sha256"]!=ref["sha256"] or att["query_inputs_sha256"]!=C.sha256(root/"query_inputs"/(scene+".json")):raise ValueError("evaluation identity differs")
   C.checked_files(folder,att["files"],("results.json","manifest.json","summary.json"))
   C.manifest_check(folder,ref,p["scenes"][scene],C.read(Path(p["full_references"][scene]["full"]["evaluation_folder"])/"manifest.json"))
   rows=C.read(folder/"results.json");v=C.independent_metrics(rows,p["scenes"][scene]["test_names"])
   v["P90_TE_cm"]=float(K.np.percentile([r["translation_error_cm"] if r["success"] else K.np.inf for r in rows],90))
   v.update(points=ref["count"],bytes=ref["bytes"]);results[mode][scene]=v
 reports={}
 for mode,per in results.items():
  macro={k:float(K.np.mean([v[k] for v in per.values()])) for k in ("median_TE_cm","median_RE_deg","mean_TE_cm","P90_TE_cm","P95_TE_cm","R5_pct","R15_pct","points")}
  macro.update(total_points=sum(v["points"] for v in per.values()),total_bytes=sum(v["bytes"] for v in per.values()),queries=sum(v["query_count"] for v in per.values()),failures=sum(v["failures"] for v in per.values()))
  reports[mode]=dict(per_scene=per,scene_equal=macro)
 C.write_new(Path(out)/"report.json",dict(complete=True,scene_count=5,evaluation_cells=15,design=DESIGN,code_sha256=hashes(),
  collection_index=C.evidence(index_path),cases=case_evidence,results=reports,common_support_not_production_full=True))
 print("MAT_T5_FIVE_SCENE_COMPLETE",flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","build","seal","evaluate","collect"));p.add_argument("--root",type=Path,required=True)
 p.add_argument("--source-index",type=Path);p.add_argument("--scene",choices=C.SCENES);p.add_argument("--config",choices=CONFIGS);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.stage=="evaluate":
  visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
  if not visible.startswith("GPU-") or "," in visible:raise ValueError("evaluation requires one assigned physical GPU UUID")
  protocol(a.root);C.require_review(a.root);torch.cuda.set_device(0);evaluate(a.root,a.scene,a.config);return
 cpu_only()
 if a.stage=="prepare":prepare(a.root,a.source_index)
 elif a.stage=="build":build(a.root,a.scene)
 elif a.stage=="seal":seal(a.root)
 else:report_collection(a.source_index,a.root)
if __name__=="__main__":main()


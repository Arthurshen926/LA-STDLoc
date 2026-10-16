"""Bounded T6 V2-filter x descriptor-fusion ablation on unselected maps."""
from __future__ import annotations
import argparse, copy, gc, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from anygsloc.artifacts import SCHEMA, sha256, validate
from experiments.cambridge_mainline_ablations import t7 as C
from experiments.cambridge_input_matrix import driver as M
from experiments.cambridge_input_matrix import projective_association as P
from experiments.cambridge_input_matrix.contracts import MatrixObservationProvider, source_contract, ASSOCIATION_GRAPH_SCHEMA

ROOT7=Path("/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2/T7")
NEW=("filter_on_single","filter_off_fused","filter_off_single")
ALL=("filter_on_fused",*NEW)
DESIGN={"schema":"mainline_t6_repaired_full_v1","scene_order":list(C.SCENES),"conditions":list(ALL),
 "new_evaluations":15,"reuse":5,"selection":"none; Full in all four cells",
 "filter":"same unmasked SP2048 cache; V2 certificate before association on/off",
 "completion":"enabled in all cells with its unchanged positive-depth/alpha physical proposal checks",
 "fusion_off":"lexicographically first (mapping_image_name, source_cache_keypoint_row), normalized single observation",
 "fusion_on":"unchanged mainline view-balanced robust group fusion",
 "fixed":"same vanilla3D prior, cameras, mapping cache, query semantic-on, seed2026, PoseLib",
 "filter_effect":"rebuild associations/landmarks; natural N; no selector",
 "fusion_effect":"only exported descriptors change within each filtering condition",
 "no_mean_or_classic":"single-observation control is not a mean-fusion or classical baseline",
 "uses_test_for_mapping":False,"test_status":"benchmark-informed development","timing":"telemetry only"}
def hashes():
 h=C.code_hashes()
 for rel in ("experiments/cambridge_mainline_ablations/t6.py","tests/test_cambridge_t6_repaired.py"):
  h[rel]=sha256(C.REPO/rel)
 return h

def prepare(root):
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 parent=C.protocol(ROOT7);sources=copy.deepcopy(parent["sources"])
 for scene in C.SCENES:
  cache=C.REPAIR/"new_observations/pure_gs.pt" if scene=="StMarysChurch" else C.MATRIX/"observations"/scene/"pure_gs.pt"
  expected=C.read(C.REPAIR/"new_full/anygsloc/summary.json")["cache_sha256"] if scene=="StMarysChurch" else C.read(cache.parent/"summary.json")["cache_sha256"]["pure_gs"]
  sources[scene]["cache"]=C.evidence(cache,expected)
 C.write_new(root/"protocol.json",{"design":DESIGN,"code_sha256":hashes(),"sources":sources,
  "scenes":parent["scenes"],"reused":parent["reused"],"reference_protocol":C.evidence(ROOT7/"protocol.json")})
 for scene in C.SCENES:
  C.snapshot(root,scene,parent["scenes"][scene])
  C.same_snapshot(C.read(root/"query_inputs"/(scene+".json")),C.read(sources[scene]["old_snapshot"]["path"]))
 C.write_new(root/"prepared.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
  "query_snapshots":{s:C.evidence(root/"query_inputs"/(s+".json")) for s in C.SCENES},
  "cuda_initialized":torch.cuda.is_initialized()})
def protocol(root):
 root=Path(root);p=C.read(root/"protocol.json");ready=C.read(root/"prepared.json")
 if p["design"]!=DESIGN or p["code_sha256"]!=hashes():raise ValueError("T6 code/design changed")
 if not ready.get("complete") or ready["protocol_sha256"]!=sha256(root/"protocol.json"):raise ValueError("T6 not prepared")
 C.unchanged(p["reference_protocol"]);C.protocol(ROOT7)
 if set(ready["query_snapshots"])!=set(C.SCENES):raise ValueError("T6 snapshot inventory differs")
 for scene,ref in ready["query_snapshots"].items():
  if Path(ref["path"]).resolve()!=(root/"query_inputs"/(scene+".json")).resolve():raise ValueError("T6 snapshot path differs")
  C.unchanged(ref)
 return p

def validate_cache(data,names):
 if data.get("input_mode")!="pure_gs" or data.get("uses_source_mapping_rgb") is not False or data.get("uses_test_queries") is not False:raise ValueError("T6 requires pure mapping cache")
 if data["mapping_names"]!=names or list(data["queries"])!=names:raise ValueError("T6 mapping registry differs")
 if set(names)&set(data.get("test_names",())):raise ValueError("mapping/test overlap")
 for name in names:
  r=data["queries"][name];n=len(r["native_keypoints"])
  if len(r["native_descriptors"])!=n or len(r["native_scores"])!=n or len(r["v2_valid"])!=n:raise ValueError("cache rows differ")
 return data

def single_descriptors(csr,names,records):
 off=np.asarray(csr["observation_offsets"]);q=np.asarray(csr["query_indices"]);k=np.asarray(csr["keypoint_indices"])
 values=[];chosen=[]
 for a,b in zip(off[:-1],off[1:]):
  if a==b:raise ValueError("empty track")
  j=min(range(int(a),int(b)),key=lambda j:(names[int(q[j])],int(k[j])))
  values.append(torch.as_tensor(records[names[int(q[j])]]["native_descriptors"])[int(k[j])].float())
  chosen.append((int(q[j]),int(k[j])))
 return F.normalize(torch.stack(values),dim=1),np.asarray(chosen,dtype=np.int64)

def unfiltered_association(observations,device):
 """Exact mainline pair/track kernels; explicitly omit only V2 admission."""
 views=[observations.build_view(i) for i in range(len(observations))]
 source_contract(observations)
 descriptors=[v.descriptors.float() for v in views]
 keypoints=[v.physical_keypoints for v in views]
 scores=[v.detector_scores.float() for v in views]
 K=torch.stack([v.intrinsics.float() for v in views]);poses=torch.stack([v.pose_w2c.float() for v in views])
 hw=torch.tensor([v.image_hw for v in views],dtype=torch.long)
 bins=P.camera_pose_bins(poses,8,direction_weight=.5)
 tracks,diagnostics,sidecar=P.build_cycle_consistent_tracks(descriptors=descriptors,keypoints=keypoints,
  detector_scores=scores,camera_K=K,pose_w2c=poses,pair_neighbors=6,pair_policy="nearest",pair_image_hw=hw,
  minimum_baseline_m=.03,maximum_baseline_m=5.,maximum_axis_angle_deg=75.,minimum_similarity=.65,
  minimum_margin=.01,maximum_epipolar_error_px=2.,epipolar_candidate_topk=4,minimum_track_views=3,
  require_cycle=True,allow_chain_tracks=True,return_pair_sidecar=True,device=device)
 return {"schema":ASSOCIATION_GRAPH_SCHEMA,"version":2,**source_contract(observations),
  "query_names":list(observations.names),"query_bins":bins,"tracks":tracks,"pair_sidecar":sidecar,
  "diagnostics":diagnostics,"component_statistics":P._component_statistics(tracks,descriptors,bins,int(diagnostics["track_count"])),
  "contract":{"render_valid_observations_required":False,"v2_filter_disabled_for_ablation":True,
   "reciprocal_descriptor":True,"known_pose_epipolar":True,"one_observation_per_camera_per_component":True,
   "posthoc_support_repair":False,"parent_child_semantics":False,"child_cap":False,
   "cycle_chain_are_confidence_attributes_not_candidate_types":True}}

def build_review(root):
 root=Path(root);r=C.read(root/"independent_pre_build_review.json")
 if r.get("verdict")!="PASS" or r.get("protocol_sha256")!=sha256(root/"protocol.json") or r.get("prepared_sha256")!=sha256(root/"prepared.json"):raise ValueError("T6 build review missing/stale")

def export_full(path,xyz,features,prov):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 state={"schema":SCHEMA,"anchor_ids":torch.arange(len(xyz)),"anchor_xyz":torch.as_tensor(xyz).float().clone(),
  "anchor_features":torch.as_tensor(features).float().clone(),"provenance":dict(prov,uses_test_queries=False,uses_source_mapping_rgb=False)}
 validate(state);torch.save(state,path);return dict(C.evidence(path),count=len(xyz))

def build(root,scene):
 root=Path(root);p=protocol(root);build_review(root)
 if scene not in C.SCENES:raise ValueError("unknown T6 scene")
 out=root/"maps"/scene
 if out.exists():raise FileExistsError(out)
 s=p["sources"][scene]
 for key in ("cache","state","full"):C.unchanged(s[key])
 data=validate_cache(C.load(s["cache"]["path"]),p["scenes"][scene]["mapping_names"])
 state=C.load(s["state"]["path"]);C.validate_csr(state);full=validate(C.load(s["full"]["path"]))
 for key in ("anchor_xyz","anchor_features"):
  if not torch.equal(state[key],full[key]):raise ValueError("T6 production Full differs")
 names=data["mapping_names"];records=data["queries"];on_single,on_rows=single_descriptors(state["projective_anchor_observations"],names,records)
 prov={"experiment":DESIGN["schema"],"protocol_sha256":sha256(root/"protocol.json"),"cache_sha256":s["cache"]["sha256"],"source_full_sha256":s["full"]["sha256"],"selector_enabled":False}
 refs={"filter_on_single":export_full(out/"filter_on_single/deployment_map.pt",state["anchor_xyz"],on_single,dict(prov,configuration="filter_on_single"))}
 with (out/"filter_on_single/chosen_observations.npy").open("xb") as f:np.save(f,on_rows)
 del state,full,on_single;gc.collect()
 provider=MatrixObservationProvider(copy.deepcopy(data))
 association=unfiltered_association(provider,"cuda:0")
 base=M.reconstruct_projective_anchors(provider,association,parallel_workers=1)
 parts=[base];reason=None
 try:parts.append(M.build_projective_completion(provider,association,voxel_size_m=.05,alpha_minimum=.05,minimum_similarity=.7,device="cuda:0"))
 except ValueError as e:
  if str(e) not in ("no unused render-valid observations for completion","depth proposals produced no descriptor-consistent component","association graph contains no ray-triangulated Anchor"):raise
  reason=str(e)
 for part in parts:M.assert_unique_track_cameras(part["projective_anchor_observations"])
 xyz=torch.cat([part["anchor_xyz"] for part in parts]);fused=torch.cat([part["anchor_features"] for part in parts])
 singles=[];chosen=[]
 for part in parts:
  desc,rows=single_descriptors(part["projective_anchor_observations"],names,records);singles.append(desc);chosen.append(rows)
 single=torch.cat(singles)
 refs["filter_off_fused"]=export_full(out/"filter_off_fused/deployment_map.pt",xyz,fused,dict(prov,configuration="filter_off_fused",v2_enabled=False,completion_enabled=True))
 refs["filter_off_single"]=export_full(out/"filter_off_single/deployment_map.pt",xyz,single,dict(prov,configuration="filter_off_single",v2_enabled=False,completion_enabled=True))
 with (out/"filter_off_single/chosen_observations.npy").open("xb") as f:np.save(f,np.concatenate(chosen))
 torch.save({"association":association,"parts":parts,"completion_reason":reason,"v2_enabled":False,"mapping_only":True},out/"construction.pt")
 C.write_new(out/"manifest.json",{"complete":True,"maps":refs,"construction":C.evidence(out/"construction.pt"),
  "chosen_observations":{c:C.evidence(out/c/"chosen_observations.npy") for c in ("filter_on_single","filter_off_single")},
  "filter_on_count":p["reused"][scene]["full"]["map"]["count"],"filter_off_count":len(xyz),
  "unchanged_completion_checks":True,"source_cache_sha256":s["cache"]["sha256"]})
 print("T6_BUILD_COMPLETE",scene,len(xyz),flush=True)

def seal(root):
 root=Path(root);p=protocol(root);refs={}
 for scene in C.SCENES:
  path=root/"maps"/scene/"manifest.json";info=C.read(path)
  if not info.get("complete") or set(info["maps"])!=set(NEW):raise ValueError("T6 maps incomplete")
  C.unchanged(info["construction"])
  for ref in info["chosen_observations"].values():C.unchanged(ref)
  states={}
  for config,ref in info["maps"].items():
   if Path(ref["path"]).resolve()!=(root/"maps"/scene/config/"deployment_map.pt").resolve():raise ValueError("T6 map path differs")
   C.unchanged(ref);states[config]=validate(C.load(ref["path"]))
  if not torch.equal(states["filter_off_fused"]["anchor_xyz"],states["filter_off_single"]["anchor_xyz"]):raise ValueError("T6 fusion changed XYZ")
  original=validate(C.load(p["sources"][scene]["full"]["path"]))
  if not torch.equal(original["anchor_xyz"],states["filter_on_single"]["anchor_xyz"]):raise ValueError("T6 on fusion changed XYZ")
  refs[scene]=C.evidence(path)
 C.write_new(root/"preflight.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
  "map_manifests":refs,"new_maps":15,"fusion_XYZ_exact":True})

def map_for(root,scene,config):
 root=Path(root);p=protocol(root);ready=C.read(root/"preflight.json")
 if not ready.get("complete") or ready["protocol_sha256"]!=sha256(root/"protocol.json") or set(ready["map_manifests"])!=set(C.SCENES):raise ValueError("T6 seal missing")
 for name,ref in ready["map_manifests"].items():
  if Path(ref["path"]).resolve()!=(root/"maps"/name/"manifest.json").resolve():raise ValueError("T6 inventory path differs")
  C.unchanged(ref)
 ref=C.read(root/"maps"/scene/"manifest.json")["maps"][config];C.unchanged(ref);return ref

def evaluate(root,scene,config):
 from anygsloc.data.datasets import ColmapDataset
 from anygsloc.evaluation.runner import main as run
 root=Path(root);p=protocol(root);C.require_review(root)
 if scene not in C.SCENES or config not in NEW:raise ValueError("outside T6 cells")
 ref=map_for(root,scene,config);s=p["scenes"][scene];out=root/"evaluations"/scene/config
 if out.exists():raise FileExistsError(out)
 snap=C.snapshot(root,scene,s);dataset=ColmapDataset(s["dataset"],load_masks=True)
 if any(dataset.valid_mask(c) is None for c in dataset.split("test")):raise ValueError("query masks incomplete")
 run(["--dataset",s["dataset"],"--map",ref["path"],"--output",str(out),"--images","processed","--image-domain","rectified",
      "--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k","--seed","2026","--device","cuda:0"])
 if C.snapshot(root,scene,s)!=snap:raise ValueError("query inputs changed")
 protocol(root);C.unchanged(ref)
 C.manifest_check(out,ref,s,C.read(Path(p["reused"][scene]["full"]["evaluation_folder"])/"manifest.json"))
 C.write_new(out/"attestation.json",{"scene":scene,"configuration":config,"protocol_sha256":sha256(root/"protocol.json"),
  "map_sha256":ref["sha256"],"query_inputs_sha256":snap,"files":{n:C.evidence(out/n) for n in ("results.json","manifest.json","summary.json")}})

def report(root,config):
 root=Path(root);p=protocol(root);per={}
 if config not in ALL:raise ValueError("unknown T6 report")
 for scene in C.SCENES:
  if config=="filter_on_fused":
   cell=p["reused"][scene]["full"];ref=cell["map"];folder=Path(cell["evaluation_folder"]);C.checked_files(folder,cell["files"],("results.json","summary.json","manifest.json"))
  else:
   folder=root/"evaluations"/scene/config;ref=map_for(root,scene,config);att=C.read(folder/"attestation.json")
   if att["scene"]!=scene or att["configuration"]!=config or att["protocol_sha256"]!=sha256(root/"protocol.json") or att["map_sha256"]!=ref["sha256"] or att["query_inputs_sha256"]!=sha256(root/"query_inputs"/(scene+".json")):raise ValueError("T6 result binding differs")
   C.checked_files(folder,att["files"],("results.json","manifest.json","summary.json"))
  C.unchanged(ref);s=p["scenes"][scene]
  C.manifest_check(folder,ref,s,C.read(Path(p["reused"][scene]["full"]["evaluation_folder"])/"manifest.json"))
  v=C.independent_metrics(C.read(folder/"results.json"),s["test_names"]);v.update(points=ref["count"],bytes=ref["bytes"]);per[scene]=v
 macro={k:float(np.mean([v[k] for v in per.values()])) for k in ("median_TE_cm","median_RE_deg","mean_TE_cm","P95_TE_cm","R5_pct","R15_pct","points")}
 macro.update(total_points=sum(v["points"] for v in per.values()),total_bytes=sum(v["bytes"] for v in per.values()),queries=sum(v["query_count"] for v in per.values()),failures=sum(v["failures"] for v in per.values()))
 C.write_new(root/"reports"/(config+".json"),{"complete":True,"configuration":config,"per_scene":per,"scene_equal":macro,"protocol_sha256":sha256(root/"protocol.json")})
 print("T6_FIVE_SCENE_COMPLETE",config,json.dumps(C.safe(macro)),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","build","seal","evaluate","report","run"))
 p.add_argument("--root",type=Path,required=True);p.add_argument("--scene",choices=C.SCENES);p.add_argument("--config",choices=ALL)
 a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.stage=="prepare":prepare(a.root)
 elif a.stage=="build":build(a.root,a.scene)
 elif a.stage=="seal":seal(a.root)
 elif a.stage=="evaluate":evaluate(a.root,a.scene,a.config)
 elif a.stage=="report":report(a.root,a.config)
 else:
  for config in NEW:
   for scene in C.SCENES:evaluate(a.root,scene,config)
   report(a.root,config)
  report(a.root,"filter_on_fused")
  C.write_new(a.root/"report.json",{"complete":True,"design":DESIGN,"results":{c:C.read(a.root/"reports"/(c+".json")) for c in ALL}})
if __name__=="__main__":main()

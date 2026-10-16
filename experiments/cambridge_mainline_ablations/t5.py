"""T5 common-observation XYZ control from the repaired, unselected mainline pool."""
from __future__ import annotations
import argparse, copy, gc, json, time
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from anygsloc.artifacts import SCHEMA, sha256, validate
from experiments.cambridge_mainline_ablations import t7 as C
from experiments.cambridge_input_matrix import driver as M
from gaussian_center_control import projected_centers
from supplement_geometry import refit_ray

ROOT7=Path("/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2/T7")
CONFIGS=("ray","depth","center")
DESIGN={"schema":"mainline_t5_common_geometry_repaired_v1","scene_order":list(C.SCENES),
 "conditions":list(CONFIGS),"new_evaluations":15,"prior":"same repaired vanilla3D as production Full",
 "selection":"none; all common-eligible landmarks from Full candidate pool",
 "coordinates":"center/depth coordinatewise lower median; uniform-base all-common-observation ray LS + three 2px Huber reweights",
 "common":"same source landmark IDs, same finite depth and visible-center observation rows, minimum three distinct physical cameras",
 "center":"one-pixel nearest-depth z-buffer, nearest projected center within 2px; not alpha-composited center",
 "ray_failure":"nonfinite ray solve or fewer than three positive depths removes track from ALL arms",
 "descriptor":"unchanged production Full descriptor on the same retained source IDs",
 "estimand":"XYZ-estimator comparison conditional on common support; does not remove track association",
 "production_full":"common-support ray row is NOT the 7.817cm production Full row",
 "query":"same query semantic-on calibration/seed/PnP; all failed queries retained",
 "timing":"telemetry only; no formal benchmark","test_status":"benchmark-informed development"}
def hashes():
 h=C.code_hashes()
 for rel in ("experiments/cambridge_mainline_ablations/t5.py","tests/test_cambridge_t5_repaired.py",
             "tools/gaussian_center_control.py","tools/supplement_geometry.py","tools/figure_cache_reader.py","tools/supplement_experiments.py"):
  h[rel]=sha256(C.REPO/rel)
 return h

def prepare(root):
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 parent=C.protocol(ROOT7)
 sources=copy.deepcopy(parent["sources"])
 for scene in C.SCENES:
  if scene=="StMarysChurch":
   cache=C.REPAIR/"new_observations/pure_gs.pt"
   expected=C.read(C.REPAIR/"new_full/anygsloc/summary.json")["cache_sha256"]
  else:
   cache=C.MATRIX/"observations"/scene/"pure_gs.pt"
   expected=C.read(cache.parent/"summary.json")["cache_sha256"]["pure_gs"]
  sources[scene]["cache"]=C.evidence(cache,expected)
  prior=parent["scenes"][scene]["prior"]
  sources[scene]["ply"]=C.evidence(prior["ply"],prior["ply_sha256"])
 C.write_new(root/"protocol.json",{"design":DESIGN,"code_sha256":hashes(),"scenes":parent["scenes"],
  "sources":sources,"full_references":parent["reused"],"reference_protocol":C.evidence(ROOT7/"protocol.json")})
 for scene in C.SCENES:
  C.snapshot(root,scene,parent["scenes"][scene])
  C.same_snapshot(C.read(root/"query_inputs"/(scene+".json")),C.read(sources[scene]["old_snapshot"]["path"]))
 C.write_new(root/"prepared.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
                                "cpu_only":not torch.cuda.is_initialized(),
                                "query_snapshots":{scene:C.evidence(root/"query_inputs"/(scene+".json")) for scene in C.SCENES}})
def protocol(root):
 root=Path(root);p=C.read(root/"protocol.json")
 if p["design"]!=DESIGN or p["code_sha256"]!=hashes():raise ValueError("T5 code/design changed")
 a=C.read(root/"prepared.json")
 if a["protocol_sha256"]!=sha256(root/"protocol.json") or not a["complete"]:raise ValueError("T5 prepare missing")
 C.unchanged(p["reference_protocol"]);C.protocol(ROOT7)
 if set(a.get("query_snapshots",{}))!=set(C.SCENES):raise ValueError("T5 snapshot inventory incomplete")
 for scene,ref in a["query_snapshots"].items():
  if Path(ref["path"]).resolve()!=(root/"query_inputs"/(scene+".json")).resolve():raise ValueError("T5 snapshot path differs")
  C.unchanged(ref)
 return p

def seal(root):
 root=Path(root);p=protocol(root);refs={}
 for scene in C.SCENES:
  path=root/"geometry"/scene/"manifest.json";info=C.read(path)
  if not info.get("complete") or set(info["maps"])!=set(CONFIGS):raise ValueError("T5 incomplete geometry")
  C.unchanged(info["common_observations"])
  common=np.load(info["common_observations"]["path"])
  source=np.asarray(common["source_rows"]);offsets=np.asarray(common["offsets"])
  if len(source)!=info["common_count"] or len(offsets)!=len(source)+1 or (np.diff(offsets)<3).any():raise ValueError("T5 common CSR differs")
  base=None
  for mode,ref in info["maps"].items():
   if Path(ref["path"]).resolve()!=(root/"geometry"/scene/mode/"deployment_map.pt").resolve():raise ValueError("T5 map path differs")
   C.unchanged(ref);state=validate(C.load(ref["path"]))
   if not np.array_equal(np.asarray(state["anchor_ids"]),source):raise ValueError("T5 common IDs differ")
   if state["provenance"]["common_observations_sha256"]!=info["common_observations"]["sha256"]:raise ValueError("T5 common binding differs")
   if state["provenance"]["source_full_sha256"]!=p["sources"][scene]["full"]["sha256"]:raise ValueError("T5 Full binding differs")
   if base is not None and not torch.equal(base,state["anchor_features"]):raise ValueError("T5 descriptors differ")
   base=state["anchor_features"]
  refs[scene]=C.evidence(path)
 C.write_new(root/"preflight.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
  "geometry_manifests":refs,"common_ids_and_descriptors_exact":True,"new_maps":15,"cuda_initialized":torch.cuda.is_initialized()})
 print("T5_SEALED",root,flush=True)

def checked_geometry(root,scene,mode):
 root=Path(root);p=protocol(root);ready=C.read(root/"preflight.json")
 if not ready.get("complete") or ready["protocol_sha256"]!=sha256(root/"protocol.json"):raise ValueError("T5 seal incomplete")
 if set(ready["geometry_manifests"])!=set(C.SCENES):raise ValueError("T5 geometry inventory incomplete")
 for name,item in ready["geometry_manifests"].items():
  if Path(item["path"]).resolve()!=(root/"geometry"/name/"manifest.json").resolve():raise ValueError("T5 manifest path differs")
  C.unchanged(item)
 if scene not in C.SCENES or mode not in CONFIGS:raise ValueError("outside T5 plan")
 info=C.read(root/"geometry"/scene/"manifest.json");C.unchanged(info["common_observations"])
 ref=info["maps"][mode];C.unchanged(ref)
 return ref

def depth_xyz(uv,depth,K,pose):
 pixel=np.asarray(uv,dtype=np.float64)+.5;z=np.asarray(depth,dtype=np.float64)
 camera=np.column_stack(((pixel[:,0]-K[0,2])*z/K[0,0],(pixel[:,1]-K[1,2])*z/K[1,1],z))
 return ((camera-pose[:3,3])@pose[:3,:3]).astype(np.float32)

def coordinate_group(cp,dp,uv,K,poses):
 mask=np.isfinite(cp).all(1)&np.isfinite(dp).all(1)
 if mask.sum()<3:return None
 ray=refit_ray(uv[mask],K[mask],poses[mask])
 center=torch.as_tensor(cp[mask]).median(0).values.numpy()
 depth=torch.as_tensor(dp[mask]).median(0).values.numpy()
 if not all(np.isfinite(x).all() for x in (ray,center,depth)):raise ValueError("nonfinite common coordinate")
 return mask,{"ray":ray,"depth":depth,"center":center}

def project_residual(point,uv,K,poses):
 camera=np.einsum("nij,j->ni",poses[:,:3,:3],point)+poses[:,:3,3]
 p=np.einsum("nij,nj->ni",K,camera)
 valid=np.isfinite(p).all(1)&(camera[:,2]>0)
 out=np.full(len(uv),np.inf,dtype=np.float64)
 out[valid]=np.linalg.norm(p[valid,:2]/p[valid,2,None]-uv[valid],axis=1)
 return out

def build(root,scene):
 root=Path(root);p=protocol(root)
 if scene not in C.SCENES:raise ValueError("unknown scene")
 out=root/"geometry"/scene
 if out.exists():raise FileExistsError(out)
 s=p["sources"][scene]
 for k in ("state","cache","ply","full"):C.unchanged(s[k])
 state=C.load(s["state"]["path"]);C.validate_csr(state)
 deployed=validate(C.load(s["full"]["path"]))
 for k in ("anchor_xyz","anchor_features"):
  if not torch.equal(state[k],deployed[k]):raise ValueError("Full/source state differs")
 del deployed;gc.collect()
 data=C.load(s["cache"]["path"]);names=state["v6_mapping_query_names"]
 if data.get("uses_test_queries") is not False or data.get("uses_source_mapping_rgb") is not False:raise ValueError("cache is not mapping-only pureGS")
 if names!=p["scenes"][scene]["mapping_names"] or names!=data["mapping_names"] or list(data["queries"])!=names:raise ValueError("mapping registry differs")
 if set(names)&set(p["scenes"][scene]["test_names"]):raise ValueError("test views in mapping")
 verts=PlyData.read(s["ply"]["path"])["vertex"]
 gaussian_xyz=np.column_stack([verts[k] for k in ("x","y","z")]).astype(np.float32)
 centers=[];depths=[];pixels=[];Ks=[];poses=[]
 started=time.perf_counter()
 for j,name in enumerate(names):
  r=data["queries"][name];uv=np.asarray(r["native_keypoints"],dtype=np.float32)
  K=np.asarray(r["native_K"],dtype=np.float32);pose=np.asarray(r["pose_w2c"],dtype=np.float32)
  h,w=map(int,r["native_input_hw"])
  z=np.asarray(r["native_depth_at_keypoints"],dtype=np.float32).reshape(-1)
  alpha=np.asarray(r["native_alpha_at_keypoints"],dtype=np.float32).reshape(-1)
  if len(z)!=len(uv) or len(alpha)!=len(uv):raise ValueError("unaligned sampled buffers")
  valid=np.isfinite(z)&(z>0)&np.isfinite(alpha)&(alpha>=.05)
  dp=depth_xyz(uv,z,K,pose);cp=projected_centers(gaussian_xyz,K,pose,w,h,uv)
  dp[~valid]=np.nan;cp[~valid]=np.nan
  depths.append(dp);centers.append(cp);pixels.append(uv+.5);Ks.append(K);poses.append(pose)
  if j%100==0:print("T5_PROJECT",scene,j+1,len(names),flush=True)
 del data,verts,gaussian_xyz;gc.collect()
 Ks=np.stack(Ks);poses=np.stack(poses);csr=state["projective_anchor_observations"]
 off=np.asarray(csr["observation_offsets"]);qs=np.asarray(csr["query_indices"]);ks=np.asarray(csr["keypoint_indices"])
 retained=[];common_q=[];common_k=[];offsets=[0];xyz={k:[] for k in CONFIGS};residuals={k:[] for k in CONFIGS}
 eligible=0;ray_rejected=0
 for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
  q=qs[a:b];key=ks[a:b]
  cp=np.stack([centers[v][k] for v,k in zip(q,key)])
  dp=np.stack([depths[v][k] for v,k in zip(q,key)])
  uv=np.stack([pixels[v][k] for v,k in zip(q,key)])
  valid=np.isfinite(cp).all(1)&np.isfinite(dp).all(1)
  if valid.sum()<3:continue
  eligible+=1
  try:item=coordinate_group(cp,dp,uv,Ks[q],poses[q])
  except (RuntimeError,ValueError,np.linalg.LinAlgError):
   ray_rejected+=1;continue
  mask,coords=item
  for mode in CONFIGS:
   xyz[mode].append(coords[mode])
   residuals[mode].append(project_residual(coords[mode],uv[mask],Ks[q[mask]],poses[q[mask]]))
  retained.append(i);common_q.extend(q[mask].tolist());common_k.extend(key[mask].tolist());offsets.append(len(common_q))
  if i%10000==0:print("T5_FIT",scene,i,len(off)-1,flush=True)
 if not retained:raise ValueError("empty common support")
 out.mkdir(parents=True)
 with (out/"common_observations.npz").open("xb") as f:
  np.savez(f,source_rows=np.asarray(retained,dtype=np.int64),offsets=np.asarray(offsets,dtype=np.int64),
    query_indices=np.asarray(common_q,dtype=np.int64),keypoint_indices=np.asarray(common_k,dtype=np.int64))
 features=state["anchor_features"][retained].float().clone();refs={}
 for mode in CONFIGS:
  folder=out/mode;folder.mkdir();path=folder/"deployment_map.pt"
  candidate={"schema":SCHEMA,"anchor_ids":torch.as_tensor(retained,dtype=torch.long),
   "anchor_xyz":torch.as_tensor(np.stack(xyz[mode]),dtype=torch.float32),"anchor_features":features.clone(),
   "provenance":{"uses_test_queries":False,"uses_source_mapping_rgb":False,"experiment":DESIGN["schema"],
     "configuration":mode,"protocol_sha256":sha256(root/"protocol.json"),"source_full_sha256":s["full"]["sha256"],
     "cache_sha256":s["cache"]["sha256"],"common_observations_sha256":sha256(out/"common_observations.npz"),
     "coordinate_only":True,"selector_enabled":False}}
  validate(candidate);torch.save(candidate,path);refs[mode]=dict(C.evidence(path),count=len(retained))
  with (folder/"mapping_residuals.npy").open("xb") as f:np.save(f,np.concatenate(residuals[mode]))
 C.write_new(out/"manifest.json",{"complete":True,"maps":refs,"source_full_count":len(state["anchor_xyz"]),
  "common_count":len(retained),"common_observation_count":len(common_q),"common_eligible":eligible,
  "shared_ray_rejections":ray_rejected,"common_observations":C.evidence(out/"common_observations.npz"),
  "wall_seconds":time.perf_counter()-started,"cuda_initialized":torch.cuda.is_initialized(),
  "source_full_sha256":s["full"]["sha256"],"mapping_geometry_only":True})
 print("T5_BUILD_COMPLETE",scene,len(retained),"from",len(state["anchor_xyz"]),flush=True)

def evaluate(root,scene,mode):
 from anygsloc.data.datasets import ColmapDataset
 from anygsloc.evaluation.runner import main as run
 root=Path(root);p=protocol(root)
 C.require_review(root)
 if scene not in C.SCENES or mode not in CONFIGS:raise ValueError("outside T5 plan")
 s=p["scenes"][scene];ref=checked_geometry(root,scene,mode)
 output=root/"evaluations"/scene/mode
 if output.exists():raise FileExistsError(output)
 snap=C.snapshot(root,scene,s);dataset=ColmapDataset(s["dataset"],load_masks=True)
 if any(dataset.valid_mask(c) is None for c in dataset.split("test")):raise ValueError("query masks missing")
 run(["--dataset",s["dataset"],"--map",ref["path"],"--output",str(output),"--images","processed",
  "--image-domain","rectified","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k","--seed","2026","--device","cuda:0"])
 if C.snapshot(root,scene,s)!=snap:raise ValueError("query changed")
 protocol(root);C.unchanged(ref)
 reference=C.read(Path(p["full_references"][scene]["full"]["evaluation_folder"])/"manifest.json")
 C.manifest_check(output,ref,s,reference)
 C.write_new(output/"attestation.json",{"protocol_sha256":sha256(root/"protocol.json"),"map_sha256":ref["sha256"],
  "query_inputs_sha256":snap,"scene":scene,"configuration":mode,
  "files":{n:C.evidence(output/n) for n in ("results.json","manifest.json","summary.json")}})

def report(root,mode):
 root=Path(root);p=protocol(root);per={}
 for scene in C.SCENES:
  folder=root/"evaluations"/scene/mode;ref=checked_geometry(root,scene,mode)
  att=C.read(folder/"attestation.json")
  if att["protocol_sha256"]!=sha256(root/"protocol.json") or att["map_sha256"]!=ref["sha256"] or att["scene"]!=scene or att["configuration"]!=mode:raise ValueError("T5 result identity changed")
  if att["query_inputs_sha256"]!=sha256(root/"query_inputs"/(scene+".json")):raise ValueError("T5 snapshot changed")
  C.checked_files(folder,att["files"],("results.json","manifest.json","summary.json"))
  C.unchanged(ref)
  C.manifest_check(folder,ref,p["scenes"][scene],C.read(Path(p["full_references"][scene]["full"]["evaluation_folder"])/"manifest.json"))
  v=C.independent_metrics(C.read(folder/"results.json"),p["scenes"][scene]["test_names"]);v.update(points=ref["count"],bytes=ref["bytes"]);per[scene]=v
 macro={k:float(np.mean([v[k] for v in per.values()])) for k in ("median_TE_cm","median_RE_deg","mean_TE_cm","P95_TE_cm","R5_pct","R15_pct","points")}
 macro.update(total_points=sum(v["points"] for v in per.values()),total_bytes=sum(v["bytes"] for v in per.values()),queries=sum(v["query_count"] for v in per.values()))
 C.write_new(root/"reports"/(mode+".json"),{"complete":True,"configuration":mode,"per_scene":per,"scene_equal":macro,
  "protocol_sha256":sha256(root/"protocol.json"),"common_support_not_production_full":True})
 print("T5_FIVE_SCENE_COMPLETE",mode,json.dumps(C.safe(macro)),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","build","seal","evaluate","report","run"))
 p.add_argument("--root",type=Path,required=True);p.add_argument("--scene",choices=C.SCENES);p.add_argument("--config",choices=CONFIGS)
 a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.stage=="prepare":prepare(a.root)
 elif a.stage=="build":build(a.root,a.scene)
 elif a.stage=="seal":seal(a.root)
 elif a.stage=="evaluate":evaluate(a.root,a.scene,a.config)
 elif a.stage=="report":report(a.root,a.config)
 else:
  for mode in CONFIGS:
   for scene in C.SCENES:evaluate(a.root,scene,mode)
   report(a.root,mode)
  C.write_new(a.root/"report.json",{"complete":True,"design":DESIGN,"results":{c:C.read(a.root/"reports"/(c+".json")) for c in CONFIGS}})
if __name__=="__main__":main()

"""StMary-only repaired-prior reproduction of frozen Stage A and B1.

No training or automatic GPU launch. Freeze only after independently accepted
training. Old gated construction is produced by the legacy runner; this adapter
never reuses its bad-prior selector trace. Existing four-scene results stay read-only.
"""
from __future__ import annotations
import argparse, copy, json, time, os, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.cambridge_mechanism_stage_a import driver as A
from experiments.cambridge_input_matrix import driver as M
from experiments.cambridge_mechanism_b1 import same_snapshot
from anygsloc.artifacts import sha256
from supplement_experiments import save_map, spatial_rows
from figure_cache_reader import read_cache

SCENE="StMarysChurch"
A_ROOT=Path("/mnt/pool/sqy/anygsloc_mechanism_stage_a_20261011_v2")
B_ROOT=Path("/mnt/pool/sqy/anygsloc_mechanism_b1_20261011_v1")
CONFIGS=(*A.CONFIGS,"classic_spatial_B")
CHUNK_SIZE=256
DESIGN={
 "schema":"stmary_repaired_mechanism_v1","scene":SCENE,"configs":list(CONFIGS),
 "prior":"new independently accepted vanilla3DGS, original training seed0/30k recipe",
 "mapping_seed":2026,"query_seed":2026,"query_semantics":True,
 "mapping_input":"pure_gs only; no mapping RGB consumption",
 "old_frontend":"legacy alpha>=0.05 radius4 before native NMS/top2048, dense alpha float16/depth float32, then V2",
 "new_frontend":"v3 unmasked RGB top2048 before V2, sampled alpha/depth float32; no refill",
 "native":"unchanged select_compact_rows_v2 with authentic quality/covariance/CSR",
 "B":"repaired old-gated native natural count, frozen before any repaired query evaluation",
 "spatial":"unchanged spatial_rows(0.5m) at B or actual new-native N",
 "classic":"existing known-camera RGB-track construction, no alpha/depth or pose recovery",
 "four_scene_reuse":"exact accepted Stage A/B1 results; only repaired StMary evaluated anew",
 "interpretation":"old/new is frontend plus cache representation package, not pure alpha",
 "chunk_size":CHUNK_SIZE,
 "test_boundary":"shared metadata loader may parse test poses; construction consumes only mapping registry",
 "timing":"parallel diagnostic telemetry, not formal benchmark",
}
FIELDS=("native_keypoints","native_scores","native_K","pose_w2c","native_input_hw")

def code_hashes():
    hashes=A.code_hashes()
    hashes.update(M.fingerprints())
    for p in (Path(__file__).resolve(),A.ROOT/"tests/test_stmary_pose_repair_mechanism.py"):
        hashes[str(p.relative_to(A.ROOT))]=sha256(p)
    return hashes

def evidence(path,expected=None):
    item=A.checked(path,expected);s=Path(path).stat()
    item.update(mtime_ns=s.st_mtime_ns,inode=s.st_ino)
    return item

def unchanged(item,hash_content=False):
    p=Path(item["path"]);s=p.stat()
    if s.st_size!=item["bytes"] or s.st_mtime_ns!=item["mtime_ns"] or s.st_ino!=item["inode"]:
        raise ValueError("frozen input file metadata changed: "+str(p))
    if hash_content:A.checked(p,item["sha256"])
    return p

def check_acceptance(training,cpu,review):
    if training.get("complete") is not True or training.get("recipe_matches_original_except_source_model_gpu") is not True:
        raise ValueError("training is not accepted")
    if training.get("seed")!=0 or training.get("iterations")!=30000:
        raise ValueError("native training seed must remain zero")
    if cpu.get("complete") is not True or cpu.get("cuda_initialized") is not False:
        raise ValueError("CPU repair validation missing")
    if review.get("complete") is not True or review.get("verdict")!="PASS":
        raise ValueError("independent repair review missing")
    if (cpu.get("mapping_count"),cpu.get("test_count"))!=(1487,530):
        raise ValueError("repaired scene split changed")
    ply=training.get("ply",{})
    if not ply.get("path") or not ply.get("sha256") or not ply.get("bytes"):
        raise ValueError("placeholder prior is forbidden")

def bind_release_evidence(repair_root,training,review):
    repair_root=Path(repair_root)
    refs={"training_acceptance.json":evidence(repair_root/"training_acceptance.json")}
    for name,key in (("cpu_validation.json","cpu_validation_sha256"),("independent_review.json","independent_review_sha256"),
                     ("training_launch.json","training_launch_sha256"),("training_outcome.json","training_outcome_sha256")):
        refs[name]=evidence(repair_root/name,training[key])
    if A.read(repair_root/"training_outcome.json").get("returncode")!=0:
        raise ValueError("training did not complete successfully")
    for path,h in review["reviewed_paths_sha256"].items():refs["reviewed:"+path]=evidence(path,h)
    for path,h in training["source_hashes"].items():refs["trainer:"+path]=evidence(path,h)
    return refs

def evaluation_reference(dataset_root,mapping,test_names):
    from anygsloc.config import FROZEN
    from anygsloc.data.calibration import resolve_scene_calibration
    calibration=resolve_scene_calibration(mapping)
    return {"dataset":str(Path(dataset_root).resolve()),"images":"processed","image_domain":"rectified",
            "split":"test","seed":2026,"queries":list(test_names),"calibration":calibration,
            "config":FROZEN.resolved(keypoint_count=calibration["keypoint_count"],
                reprojection_error_px=calibration["reprojection_error_px"],mask_policy="explicit_valid_mask")}

def freeze(repair_root):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.data.calibration import resolve_scene_calibration
    from anygsloc.mapping.virtual_camera_registry import build_virtual_camera_registry
    repair_root=Path(repair_root).resolve();root=repair_root/"mechanism"
    if root.exists():raise FileExistsError(root)
    training=A.read(repair_root/"training_acceptance.json")
    cpu=A.read(repair_root/"cpu_validation.json");review=A.read(repair_root/"independent_review.json")
    check_acceptance(training,cpu,review)
    native=repair_root/"prior/vanilla_3dgs/native"
    target=native/"point_cloud/iteration_30000/point_cloud.ply"
    if Path(training["ply"]["path"]).resolve()!=target or Path(training["native_path"]).resolve()!=native:
        raise ValueError("accepted prior belongs to a different native run")
    ply=evidence(target,training["ply"]["sha256"])
    if ply["bytes"]!=training["ply"]["bytes"]:raise ValueError("prior byte count changed")
    refs=bind_release_evidence(repair_root,training,review)
    refs["training_job"]=evidence(repair_root/"training_job.json",training["training_job_sha256"])
    job=A.read(refs["training_job"]["path"])
    if job["effective_training_seed"]!=0:raise ValueError("training job seed differs")
    refs["cfg_args"]=evidence(native/"cfg_args",training["cfg_args_sha256"])
    refs["native_cameras"]=evidence(native/"cameras.json",training["cameras_json_sha256"])
    dataset_root=repair_root/"dataset"/SCENE
    refs["repaired_images"]=evidence(dataset_root/"sparse/0/images.bin",cpu["repaired_images_sha256"])
    prepared=repair_root/"vanilla_input"/SCENE
    refs["prepared_images"]=evidence(prepared/"sparse/0/images.txt",cpu["prepared_images_sha256"])
    A.protocol(A_ROOT)
    from experiments.cambridge_mechanism_b1 import protocol as bprotocol
    bprotocol(B_ROOT)
    refs["A_protocol"]=evidence(A_ROOT/"protocol.json");refs["A_report"]=evidence(A_ROOT/"report.json")
    refs["B1_protocol"]=evidence(B_ROOT/"protocol.json");refs["B1_report"]=evidence(B_ROOT/"report.json")
    dataset=ColmapDataset(dataset_root,load_masks=False)
    cameras,registry=build_virtual_camera_registry(dataset.split("mapping"),0,deduplicate_geometry=False)
    names=[c.image_name for c in cameras];tests=[c.image_name for c in dataset.split("test")]
    if len(names)!=1487 or len(tests)!=530 or set(names)&set(tests):raise ValueError("mapping/test registry changed")
    legacy=M.read(MATRIX_PROTOCOL())["scenes"][SCENE]
    if names!=legacy["mapping_names"] or tests!=legacy["test_names"]:raise ValueError("repaired image-name identity changed")
    mask=M.effective_mask_path(dataset_root)
    scene={"dataset":str(dataset_root),"mapping_names":names,"test_names":tests,"registry":registry,
           "mask_path":str(mask),"mask_sha256":sha256(mask),
           "prior":{"ply":str(target),"ply_sha256":ply["sha256"],"gaussian_type":"3dgs","sh_degree":3,"background":"black"}}
    reference=evaluation_reference(dataset_root,cameras,tests)
    A.write_new(root/"evaluation_reference.json",reference)
    refs["evaluation_reference"]=evidence(root/"evaluation_reference.json")
    scene["historical_evaluation"]={"manifest.json":refs["evaluation_reference"]}
    old=repair_root/"old_frontend"/(SCENE+"_vanilla_3dgs")
    A.write_new(root/"protocol.json",{"design":DESIGN,"scenes":{SCENE:scene},"prior":ply,"evidence":refs,
          "old_producer":str(old),"code_sha256":code_hashes()})

def MATRIX_PROTOCOL():
    return A.MATRIX/"protocol.json"

def protocol(root):
    root=Path(root);p=A.read(root/"protocol.json")
    if p["design"]!=DESIGN or p["code_sha256"]!=code_hashes():raise ValueError("frozen mechanism code/protocol changed")
    unchanged(p["prior"])
    for item in p["evidence"].values():unchanged(item,hash_content=item["bytes"]<1024*1024)
    return p

def atomic_save_new(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        torch.save(value,temporary)
        os.link(temporary,path)  # Atomic exclusive publication: never overwrite a completed chunk.
    finally:
        temporary.unlink(missing_ok=True)

def load_chunk(path,names,start,protocol_sha):
    data=A.load(path)
    if data["protocol_sha256"]!=protocol_sha or data["start"]!=start or data["names"]!=names or list(data["queries"])!=names:
        raise ValueError("chunk identity mismatch")
    return data["queries"]

def scalar_record(camera,pose,package,sparse):
    """Same v3 orchestration on rendered RGB only; all numerical kernels reused."""
    from anygsloc.mapping.virtual_camera_registry import camera_intrinsic
    from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
    from anygsloc.mapping.render_certificate import CertificateThresholds,certify_v7_render,extreme_distortion_row_mask
    rendered=package["render"][:3].float().clamp(0,1)
    alpha=package.get("alphas",package.get("rend_alpha"));depth=package["depth"]
    if alpha is None:raise ValueError("renderer alpha missing")
    xy=sparse["keypoints"].float();thresholds=CertificateThresholds()
    distortion=package.get("rend_dist")
    artifact=None if distortion is None else extreme_distortion_row_mask(distortion,xy,
        mad_multiplier=thresholds.distortion_mad_multiplier,tail_quantile=thresholds.distortion_tail_quantile)
    cert=certify_v7_render(rgb=rendered,alpha=alpha,depth=depth,keypoints=xy,
        nearest_mapping_distance_m=0.,median_adjacent_baseline_m=1.,source_family_support=1,
        artifact_row_mask=artifact,thresholds=thresholds)
    return {"native_keypoints":xy.cpu(),"native_descriptors":F.normalize(sparse["descriptors"].cpu().float(),dim=1),
       "native_scores":sparse["keypoint_scores"].cpu().float(),"native_K":camera_intrinsic(camera),"pose_w2c":pose,
       "native_input_hw":torch.tensor([camera.height,camera.width]),"row_source":torch.ones(len(xy),dtype=torch.long),
       "source_row":torch.arange(len(xy)),"v2_valid":cert["row_valid"].cpu(),
       "native_depth_at_keypoints":sample_raster_at_grid_uv(depth.squeeze(),xy).cpu(),
       "native_alpha_at_keypoints":sample_raster_at_grid_uv(alpha.squeeze(),xy).cpu()}

@torch.inference_mode()
def extract_chunk(root,start,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.mapping.virtual_camera_registry import resolve_virtual_camera_registry
    from anygsloc.features.extractor import FeatureExtractor
    from anygsloc.priors.models import GaussianModel3D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    root=Path(root);p=protocol(root);s=p["scenes"][SCENE];device=torch.device(device)
    if device.type!="cuda" or device.index is None:raise ValueError("explicit indexed CUDA device required")
    if start not in range(0,len(s["mapping_names"]),CHUNK_SIZE):raise ValueError("invalid fixed chunk start")
    out=root/"new_observations/chunks"/f"{start:06d}.pt"
    if out.exists():raise FileExistsError(out)
    dataset=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(dataset.split("mapping"),s["registry"])
    chosen=cameras[start:start+CHUNK_SIZE];records={}
    with torch.cuda.device(device):
        model=GaussianModel3D(3,device=device);model.load_ply(p["prior"]["path"],loc_feature_dim=0);model=model.to(device).eval()
        detector=FeatureExtractor("sp",nms_radius=4).to(device).eval()
        for camera in chosen:
            if camera.image_name not in s["mapping_names"] or camera.image_name in s["test_names"]:raise ValueError("test camera in extraction")
            pose=torch.from_numpy(camera.pose_w2c).float()
            pkg=render_from_pose_gsplat(model,pose.to(device),camera.fov_x,camera.fov_y,camera.width,camera.height,
                 bg_color=torch.zeros(3,device=device),render_mode="RGB+ED",rgb_only=True,rasterize_mode="antialiased")
            rgb=pkg["render"][:3].float().clamp(0,1)
            sparse=detector.detectAndCompute(rgb[None],top_k=2048,detection_threshold=0.)[0]
            records[camera.image_name]=scalar_record(camera,pose,pkg,sparse)
        unchanged(p["prior"])
    atomic_save_new({"protocol_sha256":sha256(root/"protocol.json"),"start":start,
                   "names":[c.image_name for c in chosen],"queries":records},out)

def finalize_cache(root):
    root=Path(root);p=protocol(root);s=p["scenes"][SCENE];records={};chunks=[]
    expected={f"{start:06d}.pt" for start in range(0,len(s["mapping_names"]),CHUNK_SIZE)}
    actual={x.name for x in (root/"new_observations/chunks").glob("*.pt")}
    if actual!=expected:raise ValueError("missing or extra extraction chunks")
    for start in range(0,len(s["mapping_names"]),CHUNK_SIZE):
        path=root/"new_observations/chunks"/f"{start:06d}.pt"
        names=s["mapping_names"][start:start+CHUNK_SIZE]
        records.update(load_chunk(path,names,start,sha256(root/"protocol.json")));chunks.append(evidence(path))
    target=root/"new_observations/pure_gs.pt"
    A.save_new(target,M.payload("pure_gs",records,s,sha256(root/"protocol.json")))
    A.write_new(root/"new_observations/summary.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
          "prior_sha256":p["prior"]["sha256"],"cache":evidence(target),"chunks":chunks,
          "mapping_views":len(records),"rows":sum(len(r["native_keypoints"]) for r in records.values())})

def old_inputs(root):
    root=Path(root);p=protocol(root);s=p["scenes"][SCENE];old=Path(p["old_producer"])
    manifest=A.read(old/"anygsloc_scene_manifest.json")
    if manifest.get("complete") is not True or manifest["dataset"]!=s["dataset"]:
        raise ValueError("old frontend producer incomplete or wrong dataset")
    if manifest["prior"]["ply_sha256"]!=p["prior"]["sha256"] or Path(manifest["prior"]["ply"]).resolve()!=Path(p["prior"]["path"]):
        raise ValueError("old frontend used wrong Gaussian prior")
    scope=manifest["scientific_scope"]
    if scope["mapping_uses_source_rgb"] or scope["mapping_uses_test_queries"]:raise ValueError("old frontend mapping scope violation")
    cache=old/"observations/rendered_rgb_feature_cache.pt";full=old/"projective_map/projective_anchor_map.pt"
    refs={"manifest":evidence(old/"anygsloc_scene_manifest.json"),
       "cache":evidence(cache,manifest["results"]["observations"]["output_sha256"]),
       "full":evidence(full,manifest["results"]["projective_map"]["artifact_sha256"][str(full)])}
    acceptance_path=root.parent/"old_frontend_acceptance.json"
    acceptance=A.read(acceptance_path)
    if (acceptance.get("complete") is not True or acceptance.get("recipe_matches_original_except_input_output_gpu") is not True or
        acceptance.get("training_acceptance_sha256")!=p["evidence"]["training_acceptance.json"]["sha256"]):
        raise ValueError("old frontend has no matching accepted recipe/prior")
    refs["job"]=evidence(root.parent/"old_frontend_job.json",acceptance["old_frontend_job_sha256"])
    for name,key in (("producer_manifest","manifest"),("full_map","full"),("observation_cache","cache")):
        expected=acceptance[name];actual=refs[key]
        if Path(expected["path"]).resolve()!=Path(actual["path"]) or expected["sha256"]!=actual["sha256"] or expected["bytes"]!=actual["bytes"]:
            raise ValueError("old frontend acceptance artifact mismatch: "+name)
    refs["acceptance"]=evidence(acceptance_path)
    lazy=read_cache(cache)
    if list(lazy["queries"])!=s["mapping_names"]:raise ValueError("old cache contains extra/missing camera records")
    gate=lazy["configuration"]["render_valid_observation_extraction"]
    cfg=lazy["configuration"]
    if (gate["score_gate_stage"]!="before_native_nms_and_topk" or gate["minimum_alpha"]!=.05 or
        gate["neighborhood_radius"]!=4 or gate["appearance_reliability_used"] is not False or
        cfg["keypoints"]!=2048 or cfg["nms_radius"]!=4 or cfg["detection_threshold"]!=0. or
        cfg["gaussian_type"]!="3dgs" or cfg["sh_degree"]!=3 or cfg["background"]!="black" or
        cfg["mapping_render_sampling"]["supersampling_factor"]!=1):
        raise ValueError("wrong old frontend")
    current=A.load(unchanged(A.read(root/"new_observations/summary.json")["cache"]))["queries"]
    light={n:{k:torch.from_numpy(lazy["queries"][n][k].numpy()) for k in FIELDS} for n in s["mapping_names"]}
    A.equal_geometry(light,current,s["mapping_names"])
    state=A.load(full)
    if state["v6_mapping_query_names"]!=s["mapping_names"]:raise ValueError("old full camera identity differs")
    if state["provenance"].get("uses_source_mapping_rgb") is not False or state["provenance"].get("uses_test_queries") is not False:
        raise ValueError("old full source-free provenance differs")
    M.assert_unique_track_cameras(state["projective_anchor_observations"])
    return refs,state,light

def no_query_outputs(root):
    if any((Path(root)/"evaluations").glob("*/results.json")) or any((Path(root)/"evaluations").glob("*/*/results.json")):
        raise ValueError("budget must be frozen before query scoring")

def deploy(root,config,state,rows,extra,write_selection=True):
    root=Path(root);target=root/"maps"/config/"deployment_map.pt"
    binding={}
    if (root/"budget.json").exists():
        b=budget(root)
        if config in ("old_spatial_B","new_spatial_B","classic_spatial_B") and len(rows)!=b:
            raise ValueError("fixed-B selection count differs")
        if config=="new_spatial_N" and len(rows)!=A.read(root/"maps/new_native/selection.json")["count"]:
            raise ValueError("spatial N differs from natural new native")
        binding={"budget_sha256":sha256(root/"budget.json")}
    info=save_map(target,state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),np.asarray(rows),
         {"protocol_sha256":sha256(root/"protocol.json"),"configuration":config,
          "uses_source_mapping_rgb":False,"uses_test_queries":False,**binding,**extra})
    info.update(binding)
    if write_selection:A.write_new(target.parent/"selection.json",info)
    return info

def select_old(root):
    root=Path(root);p=protocol(root);no_query_outputs(root)
    if (root/"budget.json").exists():raise FileExistsError(root/"budget.json")
    refs,state,cache=old_inputs(root);started=time.perf_counter()
    rows,trace=A.select_compact_rows_v2(state,cache);b=int(rows.numel())
    if b<=0:raise ValueError("native selection empty")
    native=deploy(root,"old_native",state,rows.numpy(),{"natural_B":b,"source_full_sha256":refs["full"]["sha256"]},write_selection=False)
    A.save_new(root/"old_native_trace.pt",trace)
    A.write_new(root/"budget.json",{"B":b,"origin":"repaired old-gated native natural N","query_results_present":False,
         "native_map":native,"source":refs,"trace":evidence(root/"old_native_trace.pt"),
         "selection_seconds":time.perf_counter()-started,"protocol_sha256":sha256(root/"protocol.json")})
    A.write_new(root/"maps/old_native/selection.json",dict(native,budget_sha256=sha256(root/"budget.json")))
    deploy(root,"old_spatial_B",state,spatial_rows(state["anchor_xyz"].numpy(),b),{"B":b,"source_full_sha256":refs["full"]["sha256"]})

def cache_source(root):
    root=Path(root);p=protocol(root);s=A.read(root/"new_observations/summary.json");ref=s["cache"]
    if (s.get("complete") is not True or s.get("protocol_sha256")!=sha256(root/"protocol.json") or
        s.get("prior_sha256")!=p["prior"]["sha256"] or Path(ref["path"]).resolve()!=(root/"new_observations/pure_gs.pt").resolve()):
        raise ValueError("new cache source binding differs")
    unchanged(ref)
    return ref

def require_full_binding(state,method,protocol_sha,cache_sha):
    v=state["provenance"]
    if (v.get("protocol_sha256")!=protocol_sha or v.get("cache_sha256")!=cache_sha or
        v.get("method")!=method or v.get("input_mode")!="pure_gs" or
        v.get("uses_source_mapping_rgb") is not False or v.get("uses_test_queries") is not False):
        raise ValueError("full map constructor/protocol/cache binding differs")

def bound_full(root,method):
    root=Path(root);p=protocol(root);source=cache_source(root);folder=root/"new_full"/method
    summary=A.read(folder/"summary.json");info=summary["full"]
    if (summary.get("complete") is not True or summary.get("method")!=method or
        summary.get("protocol_sha256")!=sha256(root/"protocol.json") or
        summary.get("cache_sha256")!=source["sha256"] or
        Path(info["path"]).resolve()!=(folder/"deployment_map.pt").resolve()):
        raise ValueError("full-map summary binding differs")
    A.checked(info["path"],info["sha256"]);full=A.load(info["path"])
    require_full_binding(full,method,sha256(root/"protocol.json"),source["sha256"])
    return summary,full,source

def build(root,method,device,workers):
    root=Path(root);p=protocol(root);source=cache_source(root)
    data=A.load(unchanged(source,hash_content=True))
    if data["protocol_sha256"]!=sha256(root/"protocol.json"):raise ValueError("new cache identity differs")
    if method not in ("anygsloc","classic"):raise ValueError("invalid constructor")
    fn=M.construct_any if method=="anygsloc" else M.construct_classic
    xyz,features,proof=fn(data,device=device,workers=workers)
    out=root/"new_full"/method
    info=save_map(out/"deployment_map.pt",xyz,features,np.arange(len(xyz)),
                 {"protocol_sha256":sha256(root/"protocol.json"),"input_mode":"pure_gs","method":method,
                  "uses_source_mapping_rgb":False,"uses_test_queries":False,"cache_sha256":source["sha256"]})
    A.save_new(out/"construction.pt",proof)
    state=None
    if method=="anygsloc":
        state=A.adapt_new_state(proof,data,A.load(out/"deployment_map.pt"))
        state["provenance"]=dict(state["provenance"],mechanism_binding={
            "protocol_sha256":sha256(root/"protocol.json"),"cache_sha256":source["sha256"],"full_sha256":info["sha256"]})
        A.save_new(out/"selector_state.pt",state)
    A.write_new(out/"summary.json",{"complete":True,"method":method,"protocol_sha256":sha256(root/"protocol.json"),
       "cache_sha256":source["sha256"],"full":info,"construction":evidence(out/"construction.pt"),
       "selector_state":evidence(out/"selector_state.pt") if state is not None else None})

def budget(root):
    root=Path(root);p=protocol(root);b=A.read(root/"budget.json")
    if b["protocol_sha256"]!=sha256(root/"protocol.json") or b["query_results_present"] is not False:
        raise ValueError("budget identity differs")
    expected=root/"maps/old_native/deployment_map.pt";native=b["native_map"]
    selected=A.read(expected.parent/"selection.json")
    if (b["B"]!=native["count"] or Path(native["path"]).resolve()!=expected.resolve() or
        any(selected[k]!=native[k] for k in ("path","sha256","bytes","count")) or
        selected.get("budget_sha256")!=sha256(root/"budget.json")):
        raise ValueError("budget/native identity or count differs")
    A.checked(expected,native["sha256"]);state=A.load(expected)
    if len(state["anchor_xyz"])!=b["B"] or len(state["anchor_features"])!=b["B"]:
        raise ValueError("budget differs from actual old-native map")
    return int(b["B"])

def select_new(root):
    root=Path(root);b=budget(root);summary,full,source=bound_full(root,"anygsloc")
    if Path(summary["selector_state"]["path"]).resolve()!=(root/"new_full/anygsloc/selector_state.pt").resolve():
        raise ValueError("selector-state path differs")
    state=A.load(unchanged(summary["selector_state"],hash_content=True))
    expected={"protocol_sha256":sha256(root/"protocol.json"),"cache_sha256":source["sha256"],"full_sha256":summary["full"]["sha256"]}
    if state["provenance"].get("mechanism_binding")!=expected or any(not torch.equal(state[k],full[k]) for k in ("anchor_xyz","anchor_features")):
        raise ValueError("selector state detached from current full map/cache")
    cache=A.load(unchanged(source))["queries"]
    started=time.perf_counter();rows,trace=A.select_compact_rows_v2(state,cache);n=int(rows.numel())
    lineage={"source_full_sha256":summary["full"]["sha256"],"cache_sha256":source["sha256"]}
    deploy(root,"new_native",state,rows.numpy(),{"natural_N":n,"B":b,**lineage})
    deploy(root,"new_spatial_B",state,spatial_rows(state["anchor_xyz"].numpy(),b),{"B":b,**lineage})
    deploy(root,"new_spatial_N",state,spatial_rows(state["anchor_xyz"].numpy(),n),{"N":n,**lineage})
    A.save_new(root/"new_native_trace.pt",trace)
    A.write_new(root/"new_native_summary.json",{"N":n,"B":b,"seconds":time.perf_counter()-started,
                                               "trace":evidence(root/"new_native_trace.pt")})

def select_classic(root):
    root=Path(root);b=budget(root);summary,state,source=bound_full(root,"classic")
    deploy(root,"classic_spatial_B",state,spatial_rows(state["anchor_xyz"].numpy(),b),{"B":b,"constructor":"classic",
         "source_full_sha256":summary["full"]["sha256"],"cache_sha256":source["sha256"]})

def checked_selection(root,config):
    root=Path(root);b=budget(root)
    if config not in CONFIGS:raise ValueError("invalid configuration")
    path=root/"maps"/config/"deployment_map.pt";sel=A.read(path.parent/"selection.json")
    bh=sha256(root/"budget.json")
    if Path(sel["path"]).resolve()!=path.resolve() or sel.get("budget_sha256")!=bh:
        raise ValueError("selection path or frozen budget binding differs")
    A.checked(path,sel["sha256"]);state=A.load(path);prov=state["provenance"]
    if (len(state["anchor_xyz"])!=sel["count"] or len(state["anchor_features"])!=sel["count"] or
        prov.get("protocol_sha256")!=sha256(root/"protocol.json") or prov.get("configuration")!=config or
        prov.get("uses_source_mapping_rgb") is not False or prov.get("uses_test_queries") is not False):
        raise ValueError("selection map shape/provenance differs")
    if config!="old_native" and prov.get("budget_sha256")!=bh:
        raise ValueError("map provenance has wrong budget")
    if config in ("old_native","old_spatial_B","new_spatial_B","classic_spatial_B") and sel["count"]!=b:
        raise ValueError("fixed-B map count differs")
    if config=="new_spatial_N" and sel["count"]!=A.read(root/"maps/new_native/selection.json")["count"]:
        raise ValueError("spatial N is not actual native N")
    A.checked(path.with_suffix(".source_rows.npy"),prov["retained_source_rows_sha256"])
    if config.startswith("old_"):
        expected_full=A.read(root/"budget.json")["source"]["full"]["sha256"]
    else:
        method="classic" if config=="classic_spatial_B" else "anygsloc"
        expected_full=A.read(root/"new_full"/method/"summary.json")["full"]["sha256"]
        if prov.get("cache_sha256")!=cache_source(root)["sha256"]:
            raise ValueError("selection cache source differs")
    if prov.get("source_full_sha256")!=expected_full:
        raise ValueError("selection full source differs")
    return sel

def verify_cell(root,config,p):
    root=Path(root);sel=checked_selection(root,config);folder=root/"evaluations"/config
    a=A.verify_evaluation(folder,SCENE,config,p,sha256(root/"protocol.json"),
                          sha256(root/"query_inputs"/(SCENE+".json")),sel)
    if a.get("budget_sha256")!=sha256(root/"budget.json"):
        raise ValueError("evaluation budget binding differs")
    return sel

def evaluate(root,config,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root);s=p["scenes"][SCENE];budget(root)
    if config not in CONFIGS:raise ValueError("invalid configuration")
    for c in CONFIGS:checked_selection(root,c)
    dataset=ColmapDataset(s["dataset"],load_masks=True);queries=dataset.split("test")
    if [c.image_name for c in queries]!=s["test_names"] or any(dataset.valid_mask(c) is None for c in queries):
        raise ValueError("incomplete query-on input")
    snap=M.evaluation_snapshot(root,SCENE,dataset,s)
    sel=checked_selection(root,config)
    output=root/"evaluations"/config
    run(["--dataset",s["dataset"],"--map",sel["path"],"--output",str(output),"--image-domain","rectified",
         "--images","processed","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k",
         "--seed","2026","--device",device])
    if M.evaluation_snapshot(root,SCENE,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:raise ValueError("query input changed")
    A.write_new(output/"attestation.json",{"protocol_sha256":sha256(root/"protocol.json"),
         "query_inputs_sha256":snap,"map_sha256":sel["sha256"],"budget_sha256":sha256(root/"budget.json"),
         "fresh_evaluation":True,"scene":SCENE,
         "configuration":config,"files":{n:A.checked(output/n) for n in ("results.json","manifest.json","summary.json")}})
    verify_cell(root,config,p)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);a=A.read(p["evidence"]["A_report"]["path"])["results"]
    b=A.read(p["evidence"]["B1_report"]["path"])["classic_spatial_B"];results={}
    keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent","p90_te_cm",
          "failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
    for c in CONFIGS:
        folder=root/"evaluations"/c;sel=verify_cell(root,c,p)
        rows=A.read(folder/"results.json")
        if [r["image_name"] for r in rows]!=p["scenes"][SCENE]["test_names"]:raise ValueError("missing repaired queries")
        v=summarize(rows,A.read(folder/"summary.json")["loop_wall_seconds"]);v.update(map_landmarks=sel["count"],map_bytes=sel["bytes"])
        old=b if c=="classic_spatial_B" else a[c]
        scenes=copy.deepcopy(old["per_scene"]);scenes[SCENE]=v
        if list(scenes)!=[*A.BUDGET,SCENE]:raise ValueError("five-scene registry differs")
        results[c]={"per_scene":scenes,"scene_equal":{k:A.scene_mean([s[k] for s in scenes.values()]) for k in keys}}
    A.write_new(root/"report.json",json_safe({"complete":True,"scope":"five-scene A/B1 after isolated StMary pose/prior repair",
       "reused_scenes":list(A.BUDGET),"fresh_scene":SCENE,"B_StMary":budget(root),"results":results,
       "source_reports":{k:p["evidence"][k] for k in ("A_report","B1_report")}}))

def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("command",choices=("freeze","extract-chunk","finalize-cache","old-native","build","new-native","classic-spatial","evaluate","report"))
    q.add_argument("--repair-root",type=Path);q.add_argument("--root",type=Path)
    q.add_argument("--start",type=int,default=0);q.add_argument("--device",default="cuda:0")
    q.add_argument("--method",choices=("anygsloc","classic"));q.add_argument("--workers",type=int,default=4)
    q.add_argument("--config",choices=CONFIGS);args=q.parse_args();torch.set_num_threads(4)
    if args.command=="freeze":freeze(args.repair_root);return
    if args.root is None:raise ValueError("--root required")
    if args.command=="extract-chunk":extract_chunk(args.root,args.start,args.device)
    elif args.command=="finalize-cache":finalize_cache(args.root)
    elif args.command=="old-native":select_old(args.root)
    elif args.command=="build":build(args.root,args.method,args.device,args.workers)
    elif args.command=="new-native":select_new(args.root)
    elif args.command=="classic-spatial":select_classic(args.root)
    elif args.command=="evaluate":evaluate(args.root,args.config,args.device)
    else:report(args.root)
if __name__=="__main__":main()

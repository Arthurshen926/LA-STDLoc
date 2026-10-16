"""Finite real2048 + GS completion experiment. No work starts on import."""
from __future__ import annotations
import argparse, copy, gc, json, os, time
from pathlib import Path
import numpy as np
import torch
from anygsloc.artifacts import sha256
from anygsloc.data.datasets import ColmapDataset
from anygsloc.mapping.virtual_camera_registry import resolve_virtual_camera_registry, camera_intrinsic
from anygsloc.mapping.compact_v2 import select_compact_rows_v2
from experiments.cambridge_mainline_inputs import driver as D
from experiments.cambridge_input_matrix import driver as M
from .contracts import MODE, GEOMETRY, validate_raw
from .core import construct, assert_equal
from .artifacts import identity, checked_stage, checked_map, checked_cell, exact_ref, check_aux_lineage, check_state_lineage

REPO = M.ROOT
PARENT = Path("/mnt/pool/sqy/anygsloc_mainline_inputs_20261011_v1")
DEFAULT_ROOT = Path("/mnt/pool/sqy/anygsloc_gs_assisted_real_20261012_v1")
DESIGN_FILE = REPO/"outputs/anygsloc_gs_assisted_real_20261012_v1/design.json"
SCENES = D.SCENES
read, write_new, save_new, ref, checked = D.read, D.write_new, D.save_new, D.ref, D.checked
PHYSICAL_UUID = "GPU-1f21bf0e-0ae0-aad5-c30e-653ee1e13775"

def code_hashes():
    old = read(PARENT/"protocol.json")["code_sha256"]
    paths = set(old) | {str(p.relative_to(REPO)) for p in (REPO/"experiments/cambridge_gs_assisted_real").glob("*.py")}
    paths.add("tests/test_gs_assisted_real.py")
    paths.add(str(DESIGN_FILE.relative_to(REPO)))
    for key, value in old.items():
        if sha256(REPO/key) != value: raise ValueError("accepted science changed: "+key)
    for name in ("projective_completion.py", "projective_reconstruction.py"):
        a=(REPO/"experiments/cambridge_input_matrix"/name).read_text().strip()
        b=(REPO/"experiments/cambridge_gs_assisted_real"/name).read_text().strip()
        if a != b: raise ValueError("copied numerical kernel changed: "+name)
    return {key:sha256(REPO/key) for key in sorted(paths)}

def protocol(root):
    p=read(Path(root)/"protocol.json")
    if p["design"] != read(DESIGN_FILE) or p["code_sha256"] != code_hashes():
        raise ValueError("assisted protocol/source changed")
    return p

def require_review(root, stage):
    p=protocol(root); r=read(Path(root)/"independent_review.json")
    approved = r.get("cpu_approved") if stage == "audit" else r.get("approved")
    if approved is not True or r.get("protocol_sha256") != sha256(Path(root)/"protocol.json") or r.get("code_sha256") != p["code_sha256"]:
        raise ValueError("independent review missing or stale")
    release=read(Path(root)/"execution_release.json")
    if release.get("released") is not True or release.get("protocol_sha256") != sha256(Path(root)/"protocol.json"):
        raise ValueError("formal timing release absent")
    if stage in ("sample", "build", "evaluate") and release.get("gpu_uuid") != PHYSICAL_UUID:
        raise ValueError("GPU1 lease absent")
    return p

def prepare(root):
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    prior=D.protocol(PARENT)
    final=read(PARENT/"final_independent_review.json")
    if final.get("approved") is not True: raise ValueError("parent final review absent")
    for key,name in (("protocol","protocol.json"),("report","report.json"),("delivery_manifest","delivery_manifest.json")):
        exact_ref(final[key],PARENT/name,rehash=True)
    if final["code_sha256"] != prior["code_sha256"]:raise ValueError("parent final code binding differs")
    sources={}
    for scene in SCENES:
        cache, construction, full=D.source_inputs(PARENT,scene,"raw_only")
        compact=read(PARENT/"maps"/scene/"raw_only/native_compact/summary.json")
        cell=read(D.cell_path(PARENT,scene,"raw_only","full"))
        if full["sha256"] != cell["map"]["sha256"]: raise ValueError("raw Full lineage differs")
        sources[scene]=dict(raw_cache=cache, raw_construction=construction,
            raw_full=cell["map"], raw_state=compact["state"],
            raw_compact_cell=ref(D.cell_path(PARENT,scene,"raw_only","native_compact")),
            raw_full_cell=ref(D.cell_path(PARENT,scene,"raw_only","full")),
            query_snapshot=ref(PARENT/"query_inputs"/(scene+".json")),
            reference_manifest=cell["files"]["manifest.json"])
        for value in sources[scene].values(): checked(value)
    write_new(root/"protocol.json",dict(design=read(DESIGN_FILE),code_sha256=code_hashes(),
        scenes=prior["scenes"],sources=sources,
        parent_protocol=ref(PARENT/"protocol.json"),parent_report=ref(PARENT/"report.json"),
        parent_final_review=ref(PARENT/"final_independent_review.json"),
        parent_delivery_manifest=ref(PARENT/"delivery_manifest.json")))
    print("PREPARED",str(root),flush=True)

def audit_scene(root,scene):
    root=Path(root);p=require_review(root,"audit");s=p["scenes"][scene]
    sources=p["sources"][scene]
    for value in sources.values(): checked(value,rehash=True)
    prior=ref(s["prior"]["ply"],s["prior"]["ply_sha256"])
    if s["prior"]["gaussian_type"]!="3dgs": raise ValueError("wrong prior family")
    snap=D.snapshot(root,scene,s)
    D.compare_snapshot(read(root/"query_inputs"/(scene+".json")),read(sources["query_snapshot"]["path"]))
    dataset=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(dataset.split("mapping"),s["registry"])
    if [c.image_name for c in cameras]!=s["mapping_names"]:raise ValueError("mapping registry changed")
    for c in cameras:
        if sha256(c.image_path)!=s["mapping_image_sha256"][c.image_name]:raise ValueError("mapping RGB changed")
    write_new(root/"audit"/(scene+".json"),dict(**identity(root,scene,"audit"),complete=True,
        sources=sources,prior=prior,query_inputs_sha256=snap))
    print("AUDITED",scene,flush=True)

def load_source(root,scene,key):
    p=protocol(root);a=checked_stage(root,scene,"audit")
    if a["sources"] != p["sources"][scene]:raise ValueError("audited source identity differs")
    return torch.load(checked(p["sources"][scene][key]),map_location="cpu",weights_only=False)

@torch.inference_mode()
def sample(root,scene,device):
    root=Path(root);p=require_review(root,"sample");s=p["scenes"][scene]
    from .resource import assert_gpu
    assert_gpu(device)
    raw=load_source(root,scene,"raw_cache");validate_raw(raw)
    from anygsloc.priors.models import GaussianModel3D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.mapping.render_certificate import CertificateThresholds, certify_v7_render, extreme_distortion_row_mask
    from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
    data=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(data.split("mapping"),s["registry"])
    if [c.image_name for c in cameras]!=raw["mapping_names"]:raise ValueError("raw camera order")
    out=root/"auxiliary"/scene
    if out.exists():raise FileExistsError(out)
    records={};counts=dict(raw_rows=0,certificate_valid=0,finite_depth_alpha=0)
    thresholds=CertificateThresholds();dev=torch.device(device)
    model=GaussianModel3D(3,device=dev)
    model.load_ply(s["prior"]["ply"],loc_feature_dim=0);model=model.to(dev).eval()
    for i,camera in enumerate(cameras):
        r=raw["queries"][camera.image_name]
        for key,val in dict(native_K=camera_intrinsic(camera),
            pose_w2c=torch.from_numpy(camera.pose_w2c).float(),
            native_input_hw=torch.tensor([camera.height,camera.width])).items():
            assert_equal(r[key],val,"render camera."+key)
        xy=r["native_keypoints"].to(dev)
        pkg=render_from_pose_gsplat(model,r["pose_w2c"].to(dev),camera.fov_x,camera.fov_y,camera.width,camera.height,
            bg_color=torch.zeros(3,device=dev),render_mode="RGB+ED",rgb_only=True,rasterize_mode="antialiased")
        alpha=pkg.get("alphas",pkg.get("rend_alpha"));depth=pkg["depth"]
        if alpha is None:raise ValueError("render alpha missing")
        distortion=pkg.get("rend_dist")
        artifact=None if distortion is None else extreme_distortion_row_mask(distortion,xy,
            mad_multiplier=thresholds.distortion_mad_multiplier,tail_quantile=thresholds.distortion_tail_quantile)
        cert=certify_v7_render(rgb=pkg["render"][:3].float().clamp(0,1),alpha=alpha,depth=depth,keypoints=xy,
            nearest_mapping_distance_m=0.,median_adjacent_baseline_m=1.,source_family_support=1,
            artifact_row_mask=artifact,thresholds=thresholds)
        z=sample_raster_at_grid_uv(depth.squeeze(),xy).cpu()
        a=sample_raster_at_grid_uv(alpha.squeeze(),xy).cpu()
        records[camera.image_name]={**{k:r[k] for k in GEOMETRY},
            "native_keypoints":r["native_keypoints"],"depth":z,"alpha":a,"certificate_valid":cert["row_valid"].cpu()}
        counts["raw_rows"]+=len(xy);counts["certificate_valid"]+=int(cert["row_valid"].sum())
        counts["finite_depth_alpha"]+=int((torch.isfinite(z)&(z>0)&torch.isfinite(a)&(a>=.05)).sum())
        if (i+1)%100==0:print("AUX_SAMPLE",scene,i+1,len(cameras),flush=True)
    checked(read(root/"audit"/(scene+".json"))["prior"],rehash=True)
    save_new(out/"auxiliary.pt",dict(**identity(root,scene,"sample"),mapping_names=raw["mapping_names"],uses_test_queries=False,
        queries=records,raw_cache_sha256=p["sources"][scene]["raw_cache"]["sha256"],
        prior_sha256=s["prior"]["ply_sha256"]))
    write_new(out/"summary.json",dict(**identity(root,scene,"sample"),complete=True,auxiliary=ref(out/"auxiliary.pt"),counts=counts,
        raw_cache=p["sources"][scene]["raw_cache"],prior=read(root/"audit"/(scene+".json"))["prior"],
        certificate_role="completion_proposals_only; never remove raw rows"))
    print("SAMPLED",scene,json.dumps(counts),flush=True)

def build(root,scene,device):
    from .resource import assert_gpu
    root=Path(root);p=require_review(root,"build");assert_gpu(device)
    out=root/"maps"/scene/"full"
    if out.exists():raise FileExistsError(out)
    raw=load_source(root,scene,"raw_cache")
    evidence=load_source(root,scene,"raw_construction")
    base=load_source(root,scene,"raw_state");full=load_source(root,scene,"raw_full")
    info=checked_stage(root,scene,"sample")
    if info["raw_cache"] != p["sources"][scene]["raw_cache"] or info["prior"]["sha256"] != p["scenes"][scene]["prior"]["ply_sha256"]:
        raise ValueError("auxiliary summary lineage differs")
    aux=torch.load(exact_ref(info["auxiliary"],root/"auxiliary"/scene/"auxiliary.pt",rehash=True),map_location="cpu",weights_only=False)
    check_aux_lineage(aux,root,scene,p["sources"][scene],p["scenes"][scene]["prior"]["ply_sha256"])
    state,new_evidence=construct(raw,aux,evidence["association"],base,full,sha256(root/"protocol.json"),device)
    n=len(state["anchor_xyz"]);m=n-len(base["anchor_xyz"])
    state["provenance"].update(**identity(root,scene,"build"),
        raw_cache_sha256=p["sources"][scene]["raw_cache"]["sha256"],raw_full_sha256=p["sources"][scene]["raw_full"]["sha256"],
        raw_state_sha256=p["sources"][scene]["raw_state"]["sha256"],raw_construction_sha256=p["sources"][scene]["raw_construction"]["sha256"],
        prior_sha256=p["scenes"][scene]["prior"]["ply_sha256"],auxiliary_sha256=info["auxiliary"]["sha256"])
    check_state_lineage(state,root,scene,p["sources"][scene],p["scenes"][scene]["prior"]["ply_sha256"],info["auxiliary"]["sha256"])
    save_new(out/"selector_state.pt",state);save_new(out/"completion.pt",new_evidence)
    result=M.save_map(out/"deployment_map.pt",state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),
        np.arange(n),state["provenance"])
    mi=ref(result["path"],result["sha256"]);mi["count"]=n
    write_new(out/"summary.json",dict(**identity(root,scene,"build"),complete=True,full=mi,state=ref(out/"selector_state.pt"),
        completion=ref(out/"completion.pt"),raw_base_count=n-m,appended_count=m,
        raw_rows=sum(len(r["native_keypoints"]) for r in raw["queries"].values()),base_prefix_exact=True,
        actual_budget=new_evidence["actual_budget"],
        source_raw=p["sources"][scene],auxiliary=info["auxiliary"]))
    print("BUILT",scene,"base",n-m,"new",m,"full",n,flush=True)

def select(root,scene):
    root=Path(root);p=require_review(root,"select")
    out=root/"maps"/scene/"native_compact"
    if out.exists():raise FileExistsError(out)
    full=checked_map(root,scene,"full")
    state=torch.load(exact_ref(full["state"],root/"maps"/scene/"full/selector_state.pt",rehash=True),map_location="cpu",weights_only=False)
    if full["source_raw"]!=p["sources"][scene]:raise ValueError("Full source identity differs")
    aux_info=checked_stage(root,scene,"sample")
    if full["auxiliary"]!=aux_info["auxiliary"]:raise ValueError("Full auxiliary identity differs")
    check_state_lineage(state,root,scene,p["sources"][scene],p["scenes"][scene]["prior"]["ply_sha256"],aux_info["auxiliary"]["sha256"])
    raw=load_source(root,scene,"raw_cache")
    start=time.monotonic();rows,trace=select_compact_rows_v2(state,raw["queries"])
    if not len(rows) or len(torch.unique(rows))!=len(rows):raise ValueError("bad native rows")
    result=M.save_map(out/"deployment_map.pt",state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),rows.numpy(),
        {**state["provenance"],"selection":"native_compact","natural_N":len(rows)})
    save_new(out/"selector_trace.pt",trace)
    mi=ref(result["path"],result["sha256"]);mi["count"]=len(rows)
    write_new(out/"summary.json",dict(**identity(root,scene,"select"),complete=True,map=mi,source_full=full["full"],state=full["state"],
        trace=ref(out/"selector_trace.pt"),source_rows=ref(out/"deployment_map.source_rows.npy"),
        selected_base_count=int((rows<full["raw_base_count"]).sum()),
        selected_completion_count=int((rows>=full["raw_base_count"]).sum()),
        selection_seconds=time.monotonic()-start))
    print("SELECTED",scene,len(rows),flush=True)

def evaluate(root,scene,selection,device):
    from .resource import assert_gpu
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=require_review(root,"evaluate");assert_gpu(device);s=p["scenes"][scene]
    summary=checked_map(root,scene,selection)
    mi=summary["full" if selection=="full" else "map"];checked(mi,rehash=True)
    before=D.snapshot(root,scene,s)
    D.compare_snapshot(read(root/"query_inputs"/(scene+".json")),read(p["sources"][scene]["query_snapshot"]["path"]))
    out=root/"evaluations"/scene/selection
    if out.exists():raise FileExistsError(out)
    run(["--dataset",s["dataset"],"--map",mi["path"],"--output",str(out),"--image-domain","rectified",
         "--images","processed","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k",
         "--seed","2026","--device",device])
    if D.snapshot(root,scene,s)!=before:raise ValueError("query inputs changed during evaluation")
    fs=D.check_eval(out,s,mi["sha256"],read(p["sources"][scene]["reference_manifest"]["path"]))
    write_new(root/"cells"/scene/(selection+".json"),dict(**identity(root,scene,"evaluate",selection),complete=True,
        reused=False,
        query_inputs_sha256=before,map=mi,files=fs))
    print("EVALUATED",scene,selection,flush=True)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);results={}
    for selection in ("full","native_compact"):
        scenes={}
        for scene in SCENES:
            c=checked_cell(root,scene,selection)
            checked_files=D.check_eval(root/"evaluations"/scene/selection,p["scenes"][scene],c["map"]["sha256"],read(p["sources"][scene]["reference_manifest"]["path"]))
            if checked_files!=c["files"]:raise ValueError("cell evaluation files differ")
            if c["protocol_sha256"]!=sha256(root/"protocol.json") or c["query_inputs_sha256"]!=sha256(root/"query_inputs"/(scene+".json")):
                raise ValueError("cell lineage differs")
            checked(c["map"],rehash=True)
            for value in c["files"].values():checked(value,rehash=True)
            rows=read(c["files"]["results.json"]["path"])
            if [r["image_name"] for r in rows]!=p["scenes"][scene]["test_names"]:raise ValueError("query order differs")
            v=summarize(rows,read(c["files"]["summary.json"]["path"])["loop_wall_seconds"])
            v.update(map_landmarks=c["map"]["count"],map_bytes=c["map"]["bytes"])
            scenes[scene]=v
        keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","failed_query_count","map_landmarks","map_bytes")
        results[selection]=dict(per_scene=scenes,scene_equal={
            key:float(np.mean([float("inf") if v[key] is None else v[key] for v in scenes.values()])) for key in keys})
    write_new(root/"report.json",json_safe(dict(**identity(root,None,"report"),complete=True,new_cells=10,query_rows=3836,
        parent_report=p["parent_report"],results=results)))
    print("REPORTED",json.dumps(json_safe(results)),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=("prepare","audit","sample","build","select","evaluate","report"))
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--scene",choices=SCENES)
    ap.add_argument("--selection",choices=("full","native_compact"));ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args();torch.set_num_threads(2)
    if args.command=="prepare":prepare(args.root)
    elif args.command=="audit":audit_scene(args.root,args.scene)
    elif args.command=="sample":sample(args.root,args.scene,args.device)
    elif args.command=="build":build(args.root,args.scene,args.device)
    elif args.command=="select":select(args.root,args.scene)
    elif args.command=="evaluate":evaluate(args.root,args.scene,args.selection,args.device)
    else:report(args.root)
if __name__=="__main__":main()

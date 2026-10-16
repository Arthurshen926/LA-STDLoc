"""Finite five-scene mainline input-availability experiment; GPU 1 only."""
from __future__ import annotations
import argparse, copy, gc, json, os, time
from pathlib import Path
import numpy as np
import torch
from anygsloc.artifacts import sha256
from anygsloc.data.datasets import ColmapDataset
from anygsloc.mapping.virtual_camera_registry import resolve_virtual_camera_registry, camera_intrinsic
from anygsloc.mapping.compact_v2 import select_compact_rows_v2
from experiments.cambridge_input_matrix import driver as M
from experiments.cambridge_mechanism_stage_a import driver as A
from experiments.stmary_pose_repair import mechanism as S
from .adapter import adapt_selector_state, source_contract

ROOT = M.ROOT
MATRIX = Path("/mnt/pool/sqy/anygsloc_input_matrix_vanilla3d_20261011_v3")
STAGE = Path("/mnt/pool/sqy/anygsloc_mechanism_stage_a_20261011_v2")
REPAIR = Path("/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v2/mechanism")
BASE = Path("/mnt/pool/sqy/anygsloc_full_repaired_baseline_20261011_v1")
SCENES = M.SCENES
MODES = ("pure_gs", "hybrid", "raw_only")
DESIGN = {
    "schema": "cambridge_mainline_inputs_v1", "scenes": list(SCENES),
    "modes": list(MODES), "selections": ["full", "native_compact"],
    "constructor": "mainline only; original fixed-camera association/ray/fusion kernels",
    "prior": "fixed vanilla3DGS; repaired StMary v2 accepted prior; no training",
    "prior_choice_history": "exploratory prior selection, not held-out model selection",
    "mapping": "unmasked SP2048 before V2; no mapping semantic mask",
    "quota": {"total": 2048, "hybrid_real": 1024, "hybrid_render": 1024,
              "stage": "stable detector-score before V2", "refill": False},
    "raw_only": "no Gaussian fields, V2 or depth completion",
    "camera_identity": "both modalities share one physical camera; no duplicate view votes",
    "compact": "unchanged select_compact_rows_v2 with authentic quality/covariance/CSR, natural N",
    "query": "same rectified processed semantic-on SP2048, native scene calibration and PoseLib, seed2026",
    "test_boundary": "metadata loader may parse test poses; construction/selection receive mapping only",
    "aggregation": "all five scenes, equal scene weights, failures retained",
    "execution": "physical GPU1 only, one process; mapping 2 CPU threads/workers, unchanged evaluator 4 CPU threads",
    "timing": "parallel accuracy telemetry only, not a formal latency benchmark",
}
read, write_new, save_new = M.read, M.write_new, M.save_new

def ref(path, expected=None):
    path=Path(path).resolve()
    h=sha256(path)
    if expected is not None and h != expected:
        raise ValueError("source hash mismatch: "+str(path))
    st=path.stat()
    return {"path":str(path), "sha256":h, "bytes":st.st_size,
            "mtime_ns":st.st_mtime_ns, "inode":st.st_ino}

def checked(r, rehash=False):
    path=Path(r["path"]); st=path.stat()
    if st.st_size != r["bytes"] or st.st_mtime_ns != r["mtime_ns"] or st.st_ino != r["inode"]:
        raise ValueError("audited artifact stat changed: "+str(path))
    if rehash and sha256(path) != r["sha256"]:
        raise ValueError("audited artifact content changed: "+str(path))
    return path

def code_hashes():
    paths=list((ROOT/"anygsloc").rglob("*.py"))
    for folder in ("cambridge_input_matrix", "cambridge_mainline_inputs",
                   "cambridge_mechanism_stage_a", "stmary_pose_repair"):
        paths+=list((ROOT/"experiments"/folder).glob("*.py"))
    paths += [ROOT/"tools"/n for n in ("supplement_experiments.py","supplement_classic.py","figure_cache_reader.py")]
    paths += [ROOT/"tests/test_cambridge_mainline_inputs.py",
              ROOT/"experiments/cambridge_input_matrix/kernel_adaptations.json"]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}

def protocol(root):
    p=read(Path(root)/"protocol.json")
    if p["design"] != DESIGN or p["code_sha256"] != code_hashes():
        raise ValueError("frozen protocol/code changed; new version required")
    return p

def inherited(mode, selection, scene):
    if mode=="pure_gs" and selection=="native_compact":
        parent=REPAIR if scene=="StMarysChurch" else STAGE
        base=parent/"maps"/"new_native" if scene=="StMarysChurch" else parent/"maps"/scene/"new_native"
        info=read(base/"selection.json")
        ev=parent/"evaluations"/"new_native" if scene=="StMarysChurch" else parent/"evaluations"/scene/"new_native"
    elif scene=="StMarysChurch":
        if mode!="pure_gs" or selection!="full": return None
        parent=BASE; info=read(BASE/"protocol.json")["full"]["anygsloc"]
        ev=BASE/"evaluations/anygsloc"
    else:
        if selection!="full": return None
        parent=MATRIX
        info=read(MATRIX/"maps"/scene/mode/"anygsloc/summary.json")["full"]
        ev=MATRIX/"evaluations"/scene/mode/"anygsloc/full"
    att=ev/("input_matrix_attestation.json" if parent==MATRIX else "attestation.json")
    return {"map":info, "evaluation":str(ev), "attestation":str(att),
            "source_protocol":str(parent/"protocol.json"),
            "snapshot":str(parent/"query_inputs"/(scene+".json"))}

def prepare(root):
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    M.check_kernel_adaptations()
    old=read(MATRIX/"protocol.json"); base=read(BASE/"protocol.json")
    scenes={}
    for scene in SCENES:
        s=copy.deepcopy(base["scenes"][scene] if scene=="StMarysChurch" else old["scenes"][scene])
        s["mapping_image_sha256"]=old["scenes"][scene]["mapping_image_sha256"]
        s["inherited"]={m:{sel:inherited(m,sel,scene) for sel in DESIGN["selections"]} for m in MODES}
        sources={}
        for mode in ("raw_only","hybrid"):
            folder=MATRIX/"maps"/scene/mode/"anygsloc"
            sources[mode]={"cache":str(MATRIX/"observations"/scene/(mode+".pt")),
                "cache_sha256":read(MATRIX/"observations"/scene/"summary.json")["cache_sha256"][mode],
                "construction":str(folder/"construction.pt"),
                "full":read(folder/"summary.json")["full"]}
        s["old_sources"]=sources
        scenes[scene]=s
    write_new(root/"protocol.json",{"design":DESIGN,"scenes":scenes,"code_sha256":code_hashes(),
        "source_protocols":{str(p):ref(p) for p in (MATRIX/"protocol.json",STAGE/"protocol.json",
                                                   REPAIR/"protocol.json",BASE/"protocol.json")},
        "repaired_pure_cache":read(REPAIR/"new_observations/summary.json")["cache"]})
    print("PREPARED",root,flush=True)

def snapshot(root, scene, s):
    dataset=ColmapDataset(s["dataset"],load_masks=True)
    if any(dataset.valid_mask(c) is None for c in dataset.split("test")):
        raise ValueError("query masks missing")
    return M.evaluation_snapshot(root,scene,dataset,s)

def compare_snapshot(current, previous):
    for k in ("mapping_registry","calibration","mask_sha256","queries"):
        if current[k]!=previous[k]: raise ValueError("query snapshot differs: "+k)

def check_eval(folder, s, map_sha, reference_manifest):
    manifest=read(Path(folder)/"manifest.json")
    if manifest["map_sha256"]!=map_sha: raise ValueError("evaluation map mismatch")
    for k in ("dataset","images","image_domain","split","seed","queries","calibration","config"):
        if manifest[k]!=reference_manifest[k]: raise ValueError("evaluator config differs: "+k)
    rows=read(Path(folder)/"results.json")
    if [r["image_name"] for r in rows]!=s["test_names"]: raise ValueError("incomplete/reordered query results")
    return {n:ref(Path(folder)/n) for n in ("results.json","manifest.json","summary.json")}

def cell_path(root,scene,mode,selection):
    return Path(root)/"cells"/scene/mode/(selection+".json")

def audit_scene(root,scene):
    root=Path(root);p=protocol(root);s=p["scenes"][scene]
    target=root/"audit"/(scene+".json")
    if target.exists(): raise FileExistsError(target)
    dataset=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(dataset.split("mapping"),s["registry"])
    if [c.image_name for c in cameras]!=s["mapping_names"]: raise ValueError("mapping order differs")
    if set(s["mapping_names"]) & set(s["test_names"]): raise ValueError("split leakage")
    for c in cameras:
        if c.image_path.resolve() != (dataset.root/"processed"/c.image_name).resolve():
            raise ValueError("wrong processed mapping image")
        if sha256(c.image_path)!=s["mapping_image_sha256"][c.image_name]:
            raise ValueError("mapping RGB changed")
    prior=ref(s["prior"]["ply"],s["prior"]["ply_sha256"])
    if s["prior"]["gaussian_type"]!="3dgs" or s["prior"]["sh_degree"]!=3 or s["prior"]["background"]!="black":
        raise ValueError("wrong frozen prior")
    snap=snapshot(root,scene,s); current=read(root/"query_inputs"/(scene+".json"))
    reference=read(Path(s["inherited"]["pure_gs"]["full"]["evaluation"])/"manifest.json")
    reuse_count=0
    for mode in MODES:
        for selection in DESIGN["selections"]:
            entry=s["inherited"][mode][selection]
            if entry is None: continue
            att=read(entry["attestation"])
            if att["protocol_sha256"]!=sha256(entry["source_protocol"]):
                raise ValueError("source protocol attestation changed")
            previous=read(entry["snapshot"])
            if att["query_inputs_sha256"]!=sha256(entry["snapshot"]):
                raise ValueError("source query snapshot changed")
            compare_snapshot(current,previous)
            mi=ref(entry["map"]["path"],entry["map"]["sha256"])
            if att["map_sha256"]!=mi["sha256"]: raise ValueError("source attestation map differs")
            fs=check_eval(entry["evaluation"],s,mi["sha256"],reference)
            for name,r in fs.items():
                expected=att.get("output_sha256",{}).get(name)
                if expected is None: expected=att["files"][name]["sha256"]
                if r["sha256"]!=expected: raise ValueError("source output hash mismatch")
            mi["count"]=entry["map"]["count"]
            write_new(cell_path(root,scene,mode,selection),{
                "scene":scene,"input_mode":mode,"selection":selection,"reused":True,
                "protocol_sha256":sha256(root/"protocol.json"),"query_inputs_sha256":snap,
                "map":mi,"files":fs,"source_attestation":ref(entry["attestation"]),
                "source_snapshot":ref(entry["snapshot"])})
            reuse_count+=1
    sources={}
    for mode in ("raw_only","hybrid"):
        old=s["old_sources"][mode]
        if scene=="StMarysChurch" and mode=="hybrid": continue
        rec={"cache":ref(old["cache"],old["cache_sha256"])}
        if scene!="StMarysChurch":
            rec["construction"]=ref(old["construction"])
            rec["full"]=ref(old["full"]["path"],old["full"]["sha256"])
            rec["full"]["count"]=old["full"]["count"]
        sources[mode]=rec
    if scene=="StMarysChurch":
        src=p["repaired_pure_cache"]; sources["repaired_pure_cache"]=ref(src["path"],src["sha256"])
    write_new(target,{"complete":True,"scene":scene,"protocol_sha256":sha256(root/"protocol.json"),
        "mapping_count":len(cameras),"query_count":len(s["test_names"]),
        "mapping_rgb_exact":True,"prior":prior,"sources":sources,"reused_cells":reuse_count})
    print("AUDITED",scene,"reused",reuse_count,flush=True)

def audit(root,scene):
    p=protocol(root);a=read(Path(root)/"audit"/(scene+".json"))
    if not a["complete"] or a["protocol_sha256"]!=sha256(Path(root)/"protocol.json"):
        raise ValueError("preflight audit missing or stale")
    return p,a

def require_review(root):
    p=protocol(root);r=read(Path(root)/"independent_review.json")
    if r.get("approved") is not True or r.get("protocol_sha256")!=sha256(Path(root)/"protocol.json"):
        raise ValueError("independent implementation review not approved")
    if r.get("code_sha256")!=p["code_sha256"]: raise ValueError("reviewed source changed")

def assert_gpu(device):
    if device!="cuda:1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"","0,1,2"):
        raise ValueError("this experiment is restricted to physical GPU1 without remapping")

def load(r):
    return torch.load(checked(r),map_location="cpu",weights_only=False)

def rebind_raw(original, scene, cameras, protocol_sha):
    if source_contract(original)!="raw_only" or original["mapping_names"]!=scene["mapping_names"]:
        raise ValueError("raw input source/registry differs")
    records={}
    for camera in cameras:
        src=original["queries"][camera.image_name]
        if not torch.equal(torch.as_tensor(src["native_K"]),camera_intrinsic(camera)):
            raise ValueError("raw intrinsic changed")
        if not torch.equal(torch.as_tensor(src["native_input_hw"]),torch.tensor([camera.height,camera.width])):
            raise ValueError("raw image size changed")
        r=dict(src);r["pose_w2c"]=torch.from_numpy(camera.pose_w2c).float()
        records[camera.image_name]=r
    return M.payload("raw_only",records,scene,protocol_sha)

def rebind_stmary(root):
    root=Path(root);p,a=audit(root,"StMarysChurch");s=p["scenes"]["StMarysChurch"]
    data=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(data.split("mapping"),s["registry"])
    raw=rebind_raw(load(a["sources"]["raw_only"]["cache"]),s,cameras,sha256(root/"protocol.json"))
    target=root/"observations/StMarysChurch/raw_only.pt"
    save_new(target,raw)
    write_new(target.with_suffix(".json"),{"cache":ref(target),"source":a["sources"]["raw_only"]["cache"],
        "mapping_only":True,"RGB_K_size_order_verified":True,"pose_source":"repaired_v2_registry"})
    print("REBOUND_RAW_STMARY",flush=True)

@torch.inference_mode()
def extract_stmary_hybrid(root,device):
    root=Path(root);require_review(root);assert_gpu(device)
    p,a=audit(root,"StMarysChurch");s=p["scenes"]["StMarysChurch"]
    if not (root/"observations/StMarysChurch/raw_only.pt").exists(): rebind_stmary(root)
    raw=load(read(root/"observations/StMarysChurch/raw_only.json")["cache"])
    gs=load(a["sources"]["repaired_pure_cache"])
    if source_contract(gs)!="pure_gs" or gs["mapping_names"]!=s["mapping_names"]:
        raise ValueError("repaired render cache scope differs")
    A.equal_geometry(raw["queries"],gs["queries"],s["mapping_names"])
    from anygsloc.priors.models import GaussianModel3D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.mapping.render_certificate import CertificateThresholds, certify_v7_render, extreme_distortion_row_mask
    from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
    dataset=ColmapDataset(s["dataset"],load_masks=False)
    cameras=resolve_virtual_camera_registry(dataset.split("mapping"),s["registry"])
    records={};thresholds=CertificateThresholds();dev=torch.device(device)
    with torch.cuda.device(dev):
        model=GaussianModel3D(3,device=dev);model.load_ply(s["prior"]["ply"],loc_feature_dim=0);model=model.to(dev).eval()
        for i,camera in enumerate(cameras):
            r=dict(raw["queries"][camera.image_name]);xy=r["native_keypoints"].to(dev)
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
            r.update(v2_valid=cert["row_valid"].cpu(),
                native_depth_at_keypoints=sample_raster_at_grid_uv(depth.squeeze(),xy).cpu(),
                native_alpha_at_keypoints=sample_raster_at_grid_uv(alpha.squeeze(),xy).cpu())
            records[camera.image_name]=M.merge_camera(r,gs["queries"][camera.image_name])
            if (i+1)%100==0: print("HYBRID_SAMPLE",i+1,len(cameras),flush=True)
    checked(a["prior"],rehash=True)
    path=root/"observations/StMarysChurch/hybrid.pt"
    save_new(path,M.payload("hybrid",records,s,sha256(root/"protocol.json")))
    write_new(path.with_suffix(".json"),{"cache":ref(path),"prior":a["prior"],
        "raw":read(root/"observations/StMarysChurch/raw_only.json")["cache"],
        "render":a["sources"]["repaired_pure_cache"],"mapping_views":len(records),
        "rows":sum(len(r["native_keypoints"]) for r in records.values()),"no_refill":True})
    print("HYBRID_CACHE_COMPLETE",flush=True)

def source_inputs(root,scene,mode):
    p,a=audit(root,scene)
    if mode not in ("raw_only","hybrid"):raise ValueError("pure GS already complete")
    if scene=="StMarysChurch":
        cache=read(Path(root)/"observations"/scene/(mode+".json"))["cache"]
        summary=read(Path(root)/"maps"/scene/mode/"full/summary.json")
        return cache,summary["construction"],summary["full"]
    r=a["sources"][mode]
    return r["cache"],r["construction"],r["full"]

def build_stmary(root,mode,device):
    root=Path(root);require_review(root);assert_gpu(device)
    p,a=audit(root,"StMarysChurch")
    cache=read(root/"observations/StMarysChurch"/(mode+".json"))["cache"];data=load(cache)
    if source_contract(data)!=mode or data["protocol_sha256"]!=sha256(root/"protocol.json"):
        raise ValueError("new cache mode/protocol differs")
    start=time.perf_counter()
    xyz,features,evidence=M.construct_any(data,device=device,workers=2)
    folder=root/"maps/StMarysChurch"/mode/"full"
    info=M.save_map(folder/"deployment_map.pt",xyz,features,np.arange(len(xyz)),{
        "protocol_sha256":sha256(root/"protocol.json"),"input_mode":mode,"method":"anygsloc",
        "uses_source_mapping_rgb":True,"uses_test_queries":False,"cache_sha256":cache["sha256"]})
    save_new(folder/"construction.pt",evidence)
    mi=ref(info["path"],info["sha256"]);mi["count"]=info["count"]
    write_new(folder/"summary.json",{"full":mi,"construction":ref(folder/"construction.pt"),
        "cache":cache,"wall_seconds":time.perf_counter()-start})
    print("FULL_BUILT","StMarysChurch",mode,len(xyz),flush=True)

def select(root,scene,mode):
    root=Path(root);require_review(root)
    cache_ref,construction_ref,full_ref=source_inputs(root,scene,mode)
    original=load(cache_ref);evidence=load(construction_ref);full=load(full_ref)
    if source_contract(original)!=mode:raise ValueError("wrong source cache mode")
    state=adapt_selector_state(evidence,original,full,lineage={
        "protocol_sha256":sha256(root/"protocol.json"),"cache_sha256":cache_ref["sha256"],
        "construction_sha256":construction_ref["sha256"],"source_full_sha256":full_ref["sha256"]})
    folder=root/"maps"/scene/mode/"native_compact";start=time.perf_counter()
    save_new(folder/"selector_state.pt",state)
    del evidence,full;gc.collect()
    rows,trace=select_compact_rows_v2(state,original["queries"])
    if len(torch.unique(rows))!=len(rows) or not len(rows):raise ValueError("invalid native subset")
    info=M.save_map(folder/"deployment_map.pt",state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),rows.numpy(),
        {**state["provenance"],"selection":"native_compact","natural_N":len(rows)})
    save_new(folder/"selector_trace.pt",trace)
    mi=ref(info["path"],info["sha256"]);mi["count"]=info["count"]
    write_new(folder/"summary.json",{"map":mi,"source_full":full_ref,"cache":cache_ref,"construction":construction_ref,
        "state":ref(folder/"selector_state.pt"),"trace":ref(folder/"selector_trace.pt"),
        "source_rows":ref(folder/"deployment_map.source_rows.npy"),"selection_seconds":time.perf_counter()-start})
    print("COMPACT_BUILT",scene,mode,len(rows),flush=True)

def evaluate(root,scene,mode,selection,device):
    from anygsloc.evaluation.runner import main as run
    root=Path(root);require_review(root);assert_gpu(device);p=protocol(root);s=p["scenes"][scene]
    if s["inherited"][mode][selection] is not None:raise ValueError("refusing duplicate inherited evaluation")
    folder=root/"maps"/scene/mode/selection;summary=read(folder/"summary.json")
    mi=summary["full" if selection=="full" else "map"];checked(mi,rehash=True)
    before=snapshot(root,scene,s)
    out=root/"evaluations"/scene/mode/selection
    if out.exists():raise FileExistsError(out)
    run(["--dataset",s["dataset"],"--map",mi["path"],"--output",str(out),
         "--image-domain","rectified","--images","processed","--mask-policy","explicit_valid_mask",
         "--rectification-contract","opencv_same_k","--seed","2026","--device",device])
    if snapshot(root,scene,s)!=before:raise ValueError("query snapshot changed during evaluation")
    reference=read(Path(s["inherited"]["pure_gs"]["full"]["evaluation"])/"manifest.json")
    fs=check_eval(out,s,mi["sha256"],reference)
    write_new(cell_path(root,scene,mode,selection),{
        "scene":scene,"input_mode":mode,"selection":selection,"reused":False,
        "protocol_sha256":sha256(root/"protocol.json"),"query_inputs_sha256":before,"map":mi,"files":fs})
    print("EVALUATED",scene,mode,selection,flush=True)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);results={}
    for mode in MODES:
        for selection in DESIGN["selections"]:
            scenes={}
            for scene in SCENES:
                c=read(cell_path(root,scene,mode,selection))
                if (c["scene"],c["input_mode"],c["selection"])!=(scene,mode,selection):
                    raise ValueError("cell identity differs")
                if c["protocol_sha256"]!=sha256(root/"protocol.json") or c["query_inputs_sha256"]!=sha256(root/"query_inputs"/(scene+".json")):
                    raise ValueError("cell protocol/snapshot differs")
                checked(c["map"],rehash=True)
                for r in c["files"].values(): checked(r,rehash=True)
                rows=read(c["files"]["results.json"]["path"])
                if [r["image_name"] for r in rows]!=p["scenes"][scene]["test_names"]:raise ValueError("result list differs")
                v=summarize(rows,read(c["files"]["summary.json"]["path"])["loop_wall_seconds"])
                v.update(map_landmarks=c["map"]["count"],map_bytes=c["map"]["bytes"],reused=c["reused"])
                scenes[scene]=v
            keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent",
                  "p90_te_cm","failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
            results[mode+"/"+selection]={"per_scene":scenes,"scene_equal":{
                k:float(np.mean([float("inf") if v[k] is None else v[k] for v in scenes.values()])) for k in keys}}
    result=json_safe({"complete":True,"design":DESIGN,"results":results,
        "protocol_sha256":sha256(root/"protocol.json"),"cells":30,"reused":18,"new":12})
    write_new(root/"report.json",result)
    print("REPORT",json.dumps({k:v["scene_equal"] for k,v in results.items()}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("prepare","audit","rebind","extract","build","select","evaluate","report"))
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--scene",choices=SCENES)
    ap.add_argument("--mode",choices=("raw_only","hybrid"));ap.add_argument("--selection",choices=DESIGN["selections"])
    ap.add_argument("--device",default="cuda:1");args=ap.parse_args()
    torch.set_num_threads(2)
    if args.command=="prepare":prepare(args.root)
    elif args.command=="audit":audit_scene(args.root,args.scene)
    elif args.command=="rebind":rebind_stmary(args.root)
    elif args.command=="extract":extract_stmary_hybrid(args.root,args.device)
    elif args.command=="build":build_stmary(args.root,args.mode,args.device)
    elif args.command=="select":select(args.root,args.scene,args.mode)
    elif args.command=="evaluate":evaluate(args.root,args.scene,args.mode,args.selection,args.device)
    elif args.command=="report":report(args.root)
if __name__=="__main__":main()

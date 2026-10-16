"""Bounded Stage A: frontend/selector diagnosis; no rendering, mapping or training."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from anygsloc.artifacts import sha256
from anygsloc.mapping.anchor_map import merge_projective_candidates, materialize_projective_anchor_map
from anygsloc.mapping.contracts import ANCHOR_CANDIDATE_SCHEMA
from anygsloc.mapping.v2_filtered_observations import remap_candidate_rows_to_source
from anygsloc.mapping.compact_v2 import select_compact_rows_v2
from experiments.cambridge_input_matrix.driver import (
    ROOT, read, write_new, save_new, evaluation_snapshot, assert_unique_track_cameras)
from supplement_experiments import save_map, spatial_rows
from figure_cache_reader import read_cache

MATRIX = Path("/mnt/pool/sqy/anygsloc_input_matrix_vanilla3d_20261011_v3")
OLD = Path("/mnt/pool/sqy/anygsloc_three_gpu_20260910")
QUERY = Path("/mnt/pool/sqy/anygsloc_query_semantic_on_20261011_0615")
BUDGET = {"ShopFacade":12014, "OldHospital":9224, "KingsCollege":7735, "GreatCourt":44631}
CONFIGS = ("old_native", "old_spatial_B", "new_native", "new_spatial_B", "new_spatial_N")
DESIGN = {
    "schema":"cambridge_mechanism_stage_a_v1", "scene_order":list(BUDGET),
    "excluded":"StMarysChurch: unresolved per-camera quaternion/translation inconsistency in prior and GT",
    "scope":"four-scene mechanism development, not an independent five-scene benchmark",
    "prior":"same frozen vanilla_3dgs per scene", "input_mode":"pure_gs",
    "seed":2026, "query_semantics":True, "mapping_semantics":False,
    "old_frontend":"alpha>=0.05 local radius4 score gate before NMS/top2048, then V2",
    "new_frontend":"unmasked RGB NMS/top2048, then V2",
    "constructors":"existing mainline maps only; original kernels unchanged",
    "configs":list(CONFIGS), "budget_B":BUDGET,
    "native":"unchanged compact_v2, adaptive N; never truncate to B",
    "spatial":"unchanged 0.5m spatial_rows; exact B or new native N",
    "comparisons":{
        "frontend_fixed_budget":["old_spatial_B","new_spatial_B"],
        "old_selector_equal_budget":["old_native","old_spatial_B"],
        "new_selector_equal_budget":["new_native","new_spatial_N"],
        "native_operating_points":["old_native","new_native"]},
    "timing":"parallel unwarmed diagnostic telemetry, not a latency benchmark",
    "test_data_boundary":"mapping consumes only whitelist camera/observation records; shared metadata loader may parse test poses",
}

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def code_hashes():
    paths = list((ROOT/"anygsloc").rglob("*.py"))
    paths += list((ROOT/"experiments/cambridge_mechanism_stage_a").glob("*.py"))
    paths += [ROOT/"tools/supplement_experiments.py", ROOT/"tools/figure_cache_reader.py", ROOT/"experiments/cambridge_input_matrix/driver.py", ROOT/"tests/test_cambridge_mechanism_stage_a.py"]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}

def protocol(root):
    p=read(Path(root)/"protocol.json")
    if p["design"] != DESIGN or p["code_sha256"] != code_hashes():
        raise ValueError("frozen implementation or protocol changed")
    return p

def checked(path, expected=None):
    path=Path(path); actual=sha256(path)
    if expected is not None and actual != expected:
        raise ValueError("source hash mismatch: "+str(path))
    return {"path":str(path.resolve()),"sha256":actual,"bytes":path.stat().st_size}

def prepare(root):
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    matrix=read(MATRIX/"protocol.json")
    q=read(QUERY/"protocol.json")
    hashes=read(QUERY/"source_code_sha256_after.json")
    # Import no old code; verify that the evaluator's current source is unchanged.
    for rel,h in hashes.items():
        if rel.startswith("anygsloc/") and sha256(ROOT/rel)!=h:
            raise ValueError("query-on evaluator source changed: "+rel)
    scenes={}
    for scene,b in BUDGET.items():
        s=copy.deepcopy(matrix["scenes"][scene])
        producer=OLD/"runs"/(scene+"_vanilla_3dgs")
        manifest=read(producer/"anygsloc_scene_manifest.json")
        if manifest["prior"]["gaussian_type"]!="3dgs" or manifest["prior"]["ply_sha256"]!=s["prior"]["ply_sha256"]:
            raise ValueError("old/new prior mismatch")
        t=next(t for t in q["tasks"] if t["scene"]==scene and t["prior"]=="vanilla_3dgs")
        old_eval=Path(t["source_result"]).parent if t["reused"] else Path(t["output"])
        old_full=producer/"projective_map/projective_anchor_map.pt"
        refs={
            "old_cache":(producer/"observations/rendered_rgb_feature_cache.pt",manifest["results"]["observations"]["output_sha256"]),
            "old_full":(old_full,manifest["results"]["projective_map"]["artifact_sha256"][str(old_full)]),
            "old_native":(Path(t["map"]),t["map_sha256"]),
            "new_cache":(MATRIX/"observations"/scene/"pure_gs.pt",read(MATRIX/"observations"/scene/"summary.json")["cache_sha256"]["pure_gs"]),
            "new_construction":(MATRIX/"maps"/scene/"pure_gs/anygsloc/construction.pt",None),
            "new_full":(MATRIX/"maps"/scene/"pure_gs/anygsloc/full/deployment_map.pt",read(MATRIX/"maps"/scene/"pure_gs/anygsloc/summary.json")["full"]["sha256"]),
        }
        # Large source hashes are verified once per scene in preflight.
        s.update(B=b, references={k:{"path":str(v[0]),"expected_sha256":v[1]} for k,v in refs.items()},
                 producer_manifest=checked(producer/"anygsloc_scene_manifest.json"),
                 historical_evaluation={n:checked(old_eval/n) for n in ("results.json","manifest.json","summary.json")},
                 historical_rgb_hash_limitation="old query-on has no historical per-RGB SHA; fresh Stage-A reevaluation required")
        scenes[scene]=s
    from .parity import source_audit
    write_new(root/"protocol.json",{"design":DESIGN,"scenes":scenes,"code_sha256":code_hashes(),"historical_kernel_source_audit":source_audit(),
        "source_protocols":{"matrix":checked(MATRIX/"protocol.json"),"query_on":checked(QUERY/"protocol.json")}})

def equal_geometry(a,b,names):
    if list(a)!=names or list(b)!=names: raise ValueError("mapping name/order differs")
    for name in names:
        for key in ("pose_w2c","native_K","native_input_hw"):
            if not torch.equal(torch.as_tensor(a[name][key]),torch.as_tensor(b[name][key])):
                raise ValueError("camera geometry differs: "+name+" "+key)

def adapt_new_state(evidence, original, deployed):
    """Preserve actual quality/covariance; recover immutable source row identities."""
    if original["input_mode"]!="pure_gs" or original["uses_source_mapping_rgb"] or original["uses_test_queries"]:
        raise ValueError("Stage A requires source-free mapping observations")
    names=original["mapping_names"]
    filtered=evidence["filtered_observations"]["queries"]
    source_rows=[torch.nonzero(original["queries"][n]["v2_valid"],as_tuple=False).reshape(-1) for n in names]
    for n,rows in zip(names,source_rows):
        for key in ("native_keypoints","native_descriptors","native_scores"):
            if not torch.equal(filtered[n][key],original["queries"][n][key][rows]):
                raise ValueError("filtered/source row mismatch: "+n+" "+key)
    parts=[]
    for part in evidence["parts"]:
        if part.get("uses_test_queries") is not False or part.get("uses_source_mapping_rgb") is not False:
            raise ValueError("candidate provenance outside source-free contract")
        p=dict(part); p["schema"]=ANCHOR_CANDIDATE_SCHEMA
        parts.append(remap_candidate_rows_to_source(p,source_rows))
    candidates=merge_projective_candidates(parts)
    state=materialize_projective_anchor_map(candidates,lineage={"stage_a_adapter":"actual candidates, no numerical changes"})
    for key in ("anchor_xyz","anchor_features"):
        if not torch.equal(state[key],deployed[key]): raise ValueError("adapter changed deployed "+key)
    expected_quality=torch.cat([p["identity_reliability"]*p["geometry_reliability"] for p in evidence["parts"]]).clamp(0,1)
    expected_cov=torch.cat([p["anchor_position_covariance"] for p in evidence["parts"]])
    if not torch.equal(state["anchor_matchability"],expected_quality) or not torch.equal(state["anchor_position_covariance"],expected_cov):
        raise ValueError("quality/covariance changed")
    assert_unique_track_cameras(state["projective_anchor_observations"])
    if state["v6_mapping_query_names"]!=names: raise ValueError("candidate name mismatch")
    return state

def preflight(root, scene):
    root=Path(root); p=protocol(root); s=p["scenes"][scene]
    out=root/"validation"/(scene+".json")
    if out.exists(): raise FileExistsError(out)
    sources={k:checked(v["path"],v["expected_sha256"]) for k,v in s["references"].items()}
    oldcache=read_cache(sources["old_cache"]["path"]); newcache=load(sources["new_cache"]["path"])
    old_geometry={n:{k:torch.from_numpy(r[k].numpy()) for k in ("pose_w2c","native_K","native_input_hw")} for n,r in oldcache["queries"].items()}
    equal_geometry(old_geometry,newcache["queries"],s["mapping_names"])
    if set(newcache["mapping_names"]) & set(s["test_names"]): raise ValueError("mapping/test overlap")
    cfg=oldcache["configuration"]
    gate=cfg["render_valid_observation_extraction"]
    if cfg["gaussian_type"]!="3dgs" or gate["score_gate_stage"]!="before_native_nms_and_topk":
        raise ValueError("wrong historical feature cache")
    evidence=load(sources["new_construction"]["path"]); deployed=load(sources["new_full"]["path"])
    state=adapt_new_state(evidence,newcache,deployed)
    old=load(sources["old_full"]["path"]); native=load(sources["old_native"]["path"])
    if len(native["anchor_xyz"])!=s["B"]: raise ValueError("old native B drift")
    if old["v6_mapping_query_names"]!=s["mapping_names"]: raise ValueError("old map name mismatch")
    rows_path=Path(sources["old_native"]["path"]).with_name("deployment_map.source_rows.npy")
    rows=np.load(rows_path)
    for k in ("anchor_xyz","anchor_features"):
        if not torch.equal(old[k][rows],native[k]): raise ValueError("old native is not exact old full subset")
    # State has no feature caches or test poses, and retains all authentic selector evidence.
    path=root/"states"/scene/"new_full_state.pt"; save_new(path,state)
    write_new(out,{"complete":True,"source_hashes":sources,"old_native_source_rows":checked(rows_path),
        "state":checked(path),"mapping_count":len(s["mapping_names"]),
        "camera_geometry_exact":True,"source_row_remap_exact":True,
        "quality_covariance_exact":True,"new_full_count":len(state["anchor_xyz"]),
        "old_full_count":len(old["anchor_xyz"]),"old_native_count":len(native["anchor_xyz"]),
        "cuda_initialized":torch.cuda.is_initialized()})

def source_after_preflight(root,scene,key):
    a=read(Path(root)/"validation"/(scene+".json"))["source_hashes"][key]
    # Verify on use. This is bounded file IO, not a second provenance audit.
    checked(a["path"],a["sha256"])
    return a["path"]

def save_selection(root,scene,name,state,rows,extra):
    out=Path(root)/"maps"/scene/name/"deployment_map.pt"
    info=save_map(out,state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),np.asarray(rows),
        {"protocol_sha256":sha256(Path(root)/"protocol.json"),"uses_test_queries":False,
         "uses_source_mapping_rgb":False,"stage_a_configuration":name,**extra})
    write_new(out.parent/"selection.json",info)
    return info

def select(root,scene,kind):
    root=Path(root);p=protocol(root);s=p["scenes"][scene]
    audit=read(root/"validation"/(scene+".json"))
    if kind=="old":
        state=load(source_after_preflight(root,scene,"old_full"))
        save_selection(root,scene,"old_spatial_B",state,spatial_rows(state["anchor_xyz"].numpy(),s["B"]),{"B":s["B"]})
        # No need to serialize an identical copy of the old-native map.
        ref=audit["source_hashes"]["old_native"]
        write_new(root/"maps"/scene/"old_native"/"selection.json",
                  {"path":ref["path"],"sha256":ref["sha256"],"bytes":ref["bytes"],"count":s["B"],"historical_map_reused":True})
        return
    state=load(audit["state"]["path"])
    checked(audit["state"]["path"],audit["state"]["sha256"])
    if kind=="new_spatial_B":
        save_selection(root,scene,kind,state,spatial_rows(state["anchor_xyz"].numpy(),s["B"]),{"B":s["B"]});return
    cache=load(source_after_preflight(root,scene,"new_cache"))["queries"]
    started=time.perf_counter()
    rows,trace=select_compact_rows_v2(state,cache)
    elapsed=time.perf_counter()-started
    n=int(rows.numel())
    save_selection(root,scene,"new_native",state,rows.numpy(),{"adaptive_N":n,"old_B":s["B"]})
    save_new(root/"selection"/scene/"native_trace.pt",trace)
    save_selection(root,scene,"new_spatial_N",state,spatial_rows(state["anchor_xyz"].numpy(),n),{"adaptive_N":n,"matched_to":"new_native"})
    write_new(root/"selection"/scene/"native_summary.json",{"N":n,"B":s["B"],"wall_seconds":elapsed,
        "trace":checked(root/"selection"/scene/"native_trace.pt")})

def evaluate(root,scene,config,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root);s=p["scenes"][scene]
    dataset=ColmapDataset(s["dataset"],load_masks=True)
    names=[c.image_name for c in dataset.split("test")]
    if names!=s["test_names"] or any(dataset.valid_mask(c) is None for c in dataset.split("test")):
        raise ValueError("incomplete semantic-on query input")
    snap=evaluation_snapshot(root,scene,dataset,s)
    current=read(root/"query_inputs"/(scene+".json")); previous=read(MATRIX/"query_inputs"/(scene+".json"))
    if any(current[k]!=previous[k] for k in ("mapping_registry","calibration","mask_sha256","queries")):
        raise ValueError("query bytes/cameras differ from previous matrix")
    sel=read(root/"maps"/scene/config/"selection.json")
    map_path=Path(sel.get("path",root/"maps"/scene/config/"deployment_map.pt"))
    checked(map_path,sel["sha256"])
    output=root/"evaluations"/scene/config
    run(["--dataset",s["dataset"],"--map",str(map_path),"--output",str(output),
         "--image-domain","rectified","--images","processed","--mask-policy","explicit_valid_mask",
         "--rectification-contract","opencv_same_k","--seed","2026","--device",device])
    if evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:
        raise ValueError("query snapshot changed")
    manifest=read(output/"manifest.json")
    original=read(s["historical_evaluation"]["manifest.json"]["path"])
    for k in ("dataset","images","image_domain","split","seed","queries","calibration","config"):
        if manifest[k]!=original[k]: raise ValueError("historical evaluation condition mismatch: "+k)
    write_new(output/"attestation.json",{"protocol_sha256":sha256(root/"protocol.json"),
        "query_inputs_sha256":snap,"map_sha256":sel["sha256"],"fresh_evaluation":True,"scene":scene,"configuration":config,
        "files":{n:checked(output/n) for n in ("results.json","summary.json","manifest.json")}})

def verify_evaluation(folder, scene, config, p, protocol_sha, snapshot_sha, selection):
    folder=Path(folder);a=read(folder/"attestation.json")
    if (a["protocol_sha256"]!=protocol_sha or a["query_inputs_sha256"]!=snapshot_sha or
        a["map_sha256"]!=selection["sha256"] or a.get("fresh_evaluation") is not True or
        a.get("scene")!=scene or a.get("configuration")!=config):
        raise ValueError("evaluation identity/configuration/snapshot mismatch")
    for name,f in a["files"].items():
        if Path(f["path"]).resolve()!=(folder/name).resolve():
            raise ValueError("attested file belongs to a different cell")
        checked(f["path"],f["sha256"])
    manifest=read(folder/"manifest.json")
    reference=read(p["scenes"][scene]["historical_evaluation"]["manifest.json"]["path"])
    if manifest["map_sha256"]!=selection["sha256"]:
        raise ValueError("manifest map mismatch")
    for k in ("dataset","images","image_domain","split","seed","queries","calibration","config"):
        if manifest[k]!=reference[k]:raise ValueError("manifest condition mismatch: "+k)
    if manifest.get("rectification_contract")!="opencv_same_k":
        raise ValueError("rectification contract mismatch")
    return a

def scene_mean(values):
    """Retain failed/non-finite scene summaries in the macro denominator."""
    return float(np.mean([float("inf") if v is None else v for v in values]))

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);results={}
    for config in CONFIGS:
        scenes={}
        for scene in BUDGET:
            folder=root/"evaluations"/scene/config
            sel=read(root/"maps"/scene/config/"selection.json")
            verify_evaluation(folder,scene,config,p,sha256(root/"protocol.json"),
                              sha256(root/"query_inputs"/(scene+".json")),sel)
            rows=read(folder/"results.json")
            if [r["image_name"] for r in rows]!=p["scenes"][scene]["test_names"]: raise ValueError("missing queries")
            summary=summarize(rows,read(folder/"summary.json")["loop_wall_seconds"])
            sel=read(root/"maps"/scene/config/"selection.json")
            summary.update(map_landmarks=sel["count"],map_bytes=sel["bytes"])
            scenes[scene]=summary
        keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent",
              "p90_te_cm","failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
        results[config]={"per_scene":scenes,"scene_equal":{k:scene_mean([v[k] for v in scenes.values()]) for k in keys}}
    write_new(root/"report.json",json_safe({"complete":True,"scope":DESIGN["scope"],"results":results}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","preflight","select","evaluate","report"])
    p.add_argument("--root",type=Path,required=True);p.add_argument("--scene",choices=list(BUDGET))
    p.add_argument("--kind",choices=["old","new_spatial_B","new_native"])
    p.add_argument("--config",choices=CONFIGS);p.add_argument("--device",default="cuda:0")
    args=p.parse_args();torch.set_num_threads(4)
    if args.command=="prepare": prepare(args.root)
    elif args.command=="preflight":preflight(args.root,args.scene)
    elif args.command=="select":select(args.root,args.scene,args.kind)
    elif args.command=="evaluate":evaluate(args.root,args.scene,args.config,args.device)
    else:report(args.root)
if __name__=="__main__":main()


"""Five-scene compact pure-GS descriptor diagnosis; StMary data/prior repaired v2."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from experiments.stmary_pose_repair import mechanism as S
from .replay import pack_selected,descriptor_variants,tensor

A=S.A;M=S.M
REPAIRED=Path("/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v2/mechanism")
SCENES=(*A.BUDGET,S.SCENE)
ARMS=("mean_all","reproj4_robust")
DESIGN={"schema":"compact_puregs_descriptor_mechanism_v1","scene_order":list(SCENES),
 "data_version":"original four scenes plus independently accepted StMary pose/prior repair v2",
 "anchor":"old-gated native compact; frozen XYZ, selected source rows, CSR, covariance, quality and N",
 "arms":{"mean_all":"normalize every original support descriptor, unweighted mean, normalize",
         "reproj4_robust":"fixed-XYZ finite reprojection <=4px and positive depth; original robust fusion on retained support"},
 "filter":{"maximum_error_px":4.0,"minimum_distinct_cameras":3,"pixel_center_offset":0.5},
 "fallback":"fewer than 3 retained supports or norm<1e-12: original native descriptor; retain landmark",
 "replay_gate":"all selected baseline descriptors must be bitwise reproducible from original support/confidence",
 "query":"unchanged explicit semantic-valid mask, rectified RGB, seed2026, exact baseline snapshot",
 "scope":"two descriptor package interventions, not a single trim ablation, no XYZ fit or selector rerun",
 "maximum_new_evaluations":10,"timing":"shared/unwarmed diagnostic telemetry, not formal latency"}
def hashes():
    out=S.code_hashes()
    for p in list(Path(__file__).parent.glob("*.py"))+[A.ROOT/"tests/test_cambridge_descriptor_mechanism.py"]:
        out[str(p.resolve().relative_to(A.ROOT))]=S.sha256(p)
    return out

def prepare(root):
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    ap=A.protocol(S.A_ROOT);sp=S.protocol(REPAIRED)
    review=A.read(REPAIRED/"independent_final_review.json")
    if review.get("complete") is not True or review.get("verdict")!="PASS":raise ValueError("repaired baseline has no final PASS")
    report=S.evidence(REPAIRED/"report.json",review["report"]["sha256"])
    scenes={}
    for scene in SCENES:
        if scene==S.SCENE:
            info=copy.deepcopy(sp["scenes"][scene]);budget=A.read(REPAIRED/"budget.json")
            refs=budget["source"];native=S.checked_selection(REPAIRED,"old_native")
            producer=Path(refs["manifest"]["path"]);full=refs["full"];cache=refs["cache"]
            baseline=REPAIRED/"evaluations/old_native";snapshot=REPAIRED/"query_inputs"/(scene+".json")
        else:
            info=copy.deepcopy(ap["scenes"][scene]);audit=A.read(S.A_ROOT/"validation"/(scene+".json"))
            full=audit["source_hashes"]["old_full"];cache=audit["source_hashes"]["old_cache"]
            native=A.read(S.A_ROOT/"maps"/scene/"old_native/selection.json")
            producer=Path(info["producer_manifest"]["path"])
            baseline=S.A_ROOT/"evaluations"/scene/"old_native";snapshot=S.A_ROOT/"query_inputs"/(scene+".json")
        manifest=A.read(producer);association=Path(full["path"]).parent/"association_graph.pt"
        graph_sha=manifest["results"]["projective_map"]["artifact_sha256"][str(association)]
        row_path=Path(native["path"]).with_suffix(".source_rows.npy")
        info.update(inputs={"full":full,"cache":cache,"association":{"path":str(association),"sha256":graph_sha},
            "native":native,"rows":S.evidence(row_path),"producer":S.evidence(producer)},
            baseline_snapshot=S.evidence(snapshot),
            historical_evaluation={n:S.evidence(baseline/n) for n in ("manifest.json","results.json","summary.json","attestation.json")},
            native_count=native["count"])
        scenes[scene]=info
    A.write_new(root/"protocol.json",{"design":DESIGN,"scenes":scenes,"code_sha256":hashes(),
        "baseline_report":report,"baseline_review":S.evidence(REPAIRED/"independent_final_review.json"),
        "base_protocols":[S.evidence(S.A_ROOT/"protocol.json"),S.evidence(REPAIRED/"protocol.json")]})

def protocol(root):
    p=A.read(Path(root)/"protocol.json")
    if p["design"]!=DESIGN or p["code_sha256"]!=hashes():raise ValueError("descriptor experiment source/design changed")
    for ref in [p["baseline_report"],p["baseline_review"],*p["base_protocols"]]:S.unchanged(ref,hash_content=True)
    return p

def tensor_sha(x):
    x=torch.as_tensor(x).contiguous().cpu()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()

def summarize_stats(stats):
    support=stats["original_support"];kept=stats["retained_support"]
    return {"original_observations":int(support.sum()),"retained_observations":int(kept.sum()),
       "zero_support_points":int((kept==0).sum()),"insufficient_support_points":int((kept<3).sum()),
       "mean_zero_fallback_points":int(stats["mean_zero_fallback"].sum()),
       "reproj_fallback_points":int(stats["reproj_fallback"].sum()),
       "reproj_zero_fallback_points":int(stats["reproj_zero_fallback"].sum()),
       "reproj_fallback_fraction":float(stats["reproj_fallback"].float().mean())}

def preflight(root,scene):
    root=Path(root);p=protocol(root);s=p["scenes"][scene];out=root/"preflight"/scene
    if out.exists():raise FileExistsError(out)
    started=time.perf_counter();refs={k:S.evidence(v["path"],v["sha256"]) for k,v in s["inputs"].items()}
    full=A.load(refs["full"]["path"]);native=A.load(refs["native"]["path"]);rows=np.load(refs["rows"]["path"])
    if len(rows)!=s["native_count"]:raise ValueError("native count changed")
    for k in ("anchor_xyz","anchor_features"):
        if not torch.equal(full[k][rows],native[k]):raise ValueError("native not exact immutable full subset: "+k)
    prov=full["provenance"]
    if prov["v8_observation_cache_sha256"]!=refs["cache"]["sha256"] or prov["v8_association_graph_sha256"]!=refs["association"]["sha256"]:
        raise ValueError("full/cache/association lineage mismatch")
    cache=S.read_cache(refs["cache"]["path"]);assoc=A.load(refs["association"]["path"])
    if assoc["input_sha256"]["observation_cache"]!=refs["cache"]["sha256"]:raise ValueError("association/cache source differs")
    packed=pack_selected(full,cache,assoc,rows,s["mapping_names"],s["test_names"])
    geometry={e["name"]:e for e in s["registry"]["entries"]}
    for q,name in enumerate(s["mapping_names"]):
        e=geometry[name]
        if not torch.equal(packed["K"][q],torch.tensor(e["intrinsic"])) or not torch.equal(packed["poses"][q],torch.tensor(e["pose_w2c"])):
            raise ValueError("mapping camera differs from frozen registry")
    variants,replay,stats=descriptor_variants(full,packed)
    replay.update(seconds=time.perf_counter()-started,source_hashes=refs,scene=scene,
                  observations=len(packed["query"]),cuda_initialized=torch.cuda.is_initialized())
    A.write_new(out/"replay.json",replay)
    if not replay["bitwise_equal"]:raise ValueError("all-support robust descriptor replay is not bitwise exact; stop before deployment")
    invariants={"rows":torch.as_tensor(rows),"anchor_xyz":native["anchor_xyz"],
       "anchor_position_covariance":full["anchor_position_covariance"][rows],
       "anchor_matchability":full["anchor_matchability"][rows],
       "observation_offsets":packed["offsets"],"query_indices":packed["query"],"keypoint_indices":packed["keypoint"],
       "view_bins":packed["bins"]}
    S.atomic_save_new(invariants,out/"fixed_evidence.pt");S.atomic_save_new(stats,out/"support_diagnostics.pt")
    selection={}
    for arm,features in variants.items():
        changed=full["anchor_features"].clone();changed[rows]=features
        info=S.save_map(root/"maps"/scene/arm/"deployment_map.pt",full["anchor_xyz"].numpy(),changed.numpy(),rows,
            {"experiment":DESIGN["schema"],"protocol_sha256":S.sha256(root/"protocol.json"),"scene":scene,"arm":arm,
             "uses_source_mapping_rgb":False,"uses_test_queries":False,"native_map_sha256":refs["native"]["sha256"],
             "fixed_evidence_sha256":S.sha256(out/"fixed_evidence.pt"),"changed_field":"anchor_features only"})
        if not torch.equal(A.load(info["path"])["anchor_xyz"],native["anchor_xyz"]):raise ValueError("descriptor intervention changed XYZ")
        if S.sha256(Path(info["path"]).with_suffix(".source_rows.npy"))!=refs["rows"]["sha256"]:
            raise ValueError("descriptor intervention changed source-row identities")
        A.write_new(Path(info["path"]).parent/"selection.json",info);selection[arm]=info
    protocol(root)
    A.write_new(out/"summary.json",{"complete":True,"protocol_sha256":S.sha256(root/"protocol.json"),"replay":replay,
       "native_count":len(rows),"quality_covariance_sha256":{k:tensor_sha(invariants[k]) for k in ("anchor_position_covariance","anchor_matchability")},
       "fixed_evidence":S.evidence(out/"fixed_evidence.pt"),"support_diagnostics":S.evidence(out/"support_diagnostics.pt"),
       "support":summarize_stats(stats),"maps":selection,"wall_seconds":time.perf_counter()-started})

def checked_map(root,scene,arm):
    root=Path(root);p=protocol(root);a=A.read(root/"preflight"/scene/"summary.json")
    if not a["complete"] or a["protocol_sha256"]!=S.sha256(root/"protocol.json") or not a["replay"]["bitwise_equal"]:
        raise ValueError("CPU preflight not accepted")
    ref=a["maps"][arm];path=root/"maps"/scene/arm/"deployment_map.pt"
    if Path(ref["path"]).resolve()!=path.resolve():raise ValueError("variant path differs")
    A.checked(path,ref["sha256"]);S.unchanged(a["fixed_evidence"],True)
    data=A.load(path);fixed=A.load(a["fixed_evidence"]["path"])
    if len(data["anchor_xyz"])!=p["scenes"][scene]["native_count"] or not torch.equal(data["anchor_xyz"],fixed["anchor_xyz"]):
        raise ValueError("native XYZ/count changed")
    A.checked(path.with_suffix(".source_rows.npy"),p["scenes"][scene]["inputs"]["rows"]["sha256"])
    return ref

def evaluate(root,scene,arm,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root);s=p["scenes"][scene];ref=checked_map(root,scene,arm)
    if torch.device(device).type!="cuda" or torch.device(device).index is None:raise ValueError("explicit CUDA device required")
    output=root/"evaluations"/scene/arm
    if output.exists():raise FileExistsError(output)
    dataset=ColmapDataset(s["dataset"],load_masks=True)
    if [c.image_name for c in dataset.split("test")]!=s["test_names"] or any(dataset.valid_mask(c) is None for c in dataset.split("test")):
        raise ValueError("semantic-on query registry incomplete")
    snap=M.evaluation_snapshot(root,scene,dataset,s);current=A.read(root/"query_inputs"/(scene+".json"))
    old=A.read(S.unchanged(s["baseline_snapshot"],True))
    if any(current[k]!=old[k] for k in ("mapping_registry","calibration","mask_sha256","queries")):
        raise ValueError("query snapshot differs from native baseline")
    run(["--dataset",s["dataset"],"--map",ref["path"],"--output",str(output),"--image-domain","rectified",
         "--images","processed","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k",
         "--seed","2026","--device",device])
    if M.evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:raise ValueError("query inputs changed")
    A.write_new(output/"attestation.json",{"protocol_sha256":S.sha256(root/"protocol.json"),
       "query_inputs_sha256":snap,"map_sha256":ref["sha256"],"fresh_evaluation":True,"scene":scene,"configuration":arm,
       "files":{n:A.checked(output/n) for n in ("results.json","manifest.json","summary.json")}})
    A.verify_evaluation(output,scene,arm,p,S.sha256(root/"protocol.json"),snap,ref)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);results={}
    for arm in ("native_reference",*ARMS):
        per={}
        for scene in SCENES:
            s=p["scenes"][scene]
            if arm=="native_reference":
                folder=Path(s["historical_evaluation"]["results.json"]["path"]).parent
                for v in s["historical_evaluation"].values():S.unchanged(v,True)
                ref=s["inputs"]["native"]
            else:
                folder=root/"evaluations"/scene/arm;ref=checked_map(root,scene,arm)
                A.verify_evaluation(folder,scene,arm,p,S.sha256(root/"protocol.json"),S.sha256(root/"query_inputs"/(scene+".json")),ref)
            rows=A.read(folder/"results.json")
            if [r["image_name"] for r in rows]!=s["test_names"]:raise ValueError("incomplete query results")
            v=summarize(rows,A.read(folder/"summary.json")["loop_wall_seconds"])
            v.update(map_landmarks=ref["count"],map_bytes=ref["bytes"]);per[scene]=v
        keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent","p90_te_cm",
              "failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
        results[arm]={"per_scene":per,"scene_equal":{k:A.scene_mean([v[k] for v in per.values()]) for k in keys}}
    A.write_new(root/"report.json",json_safe({"complete":True,"design":DESIGN,"results":results}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","preflight","evaluate","report"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--scene",choices=SCENES);p.add_argument("--arm",choices=ARMS);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=="prepare":prepare(a.root)
    elif a.stage=="preflight":preflight(a.root,a.scene)
    elif a.stage=="evaluate":evaluate(a.root,a.scene,a.arm,a.device)
    else:report(a.root)
if __name__=="__main__":main()

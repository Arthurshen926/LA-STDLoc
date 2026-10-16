"""Bounded B1: existing classical RGB-track maps at Stage A's frozen point budgets."""
from __future__ import annotations
import argparse, copy
from pathlib import Path
import numpy as np
import torch
from experiments.cambridge_mechanism_stage_a import driver as A
from experiments.cambridge_input_matrix.driver import evaluation_snapshot
from anygsloc.artifacts import sha256
from supplement_experiments import save_map, spatial_rows

CONFIG = "classic_spatial_B"
DESIGN = {"schema":"cambridge_mechanism_b1_v1", "scope":A.DESIGN["scope"],
          "scene_order":list(A.BUDGET), "B":A.BUDGET, "input":"existing pure_gs classic full",
          "selection":"unchanged spatial_rows at frozen old-native B",
          "comparison":"same-selector architecture contrast against A.new_spatial_B; operational contrast against A.new_native also changes selector",
          "seed":2026, "query_semantics":True, "construct_or_render":False,
          "StMarysChurch":"quarantined; no training or repair"}
A_ROOT = Path("/mnt/pool/sqy/anygsloc_mechanism_stage_a_20261011_v2")

def code_hashes():
    out=A.code_hashes()
    for p in (Path(__file__).resolve(), A.ROOT/"tests/test_cambridge_mechanism_b1.py"):
        out[str(p.relative_to(A.ROOT))]=sha256(p)
    return out

def check_source(state, expected_cache):
    p=state["provenance"]
    if (p.get("input_mode")!="pure_gs" or p.get("method")!="classic" or
        p.get("uses_source_mapping_rgb") is not False or p.get("uses_test_queries") is not False or
        p.get("cache_sha256")!=expected_cache):
        raise ValueError("classic source provenance mismatch")

def same_snapshot(current, reference):
    for k in ("mapping_registry","calibration","mask_sha256","queries"):
        if current[k]!=reference[k]:
            raise ValueError("B1 query snapshot differs: "+k)

def prepare(root):
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    ap=A.protocol(A_ROOT)
    if A.read(A_ROOT/"report.json").get("complete") is not True: raise ValueError("Stage A incomplete")
    scenes=copy.deepcopy(ap["scenes"])
    for scene,s in scenes.items():
        base=A.MATRIX/"maps"/scene/"pure_gs/classic"
        source=base/"full/deployment_map.pt"
        info=A.read(base/"summary.json")["full"]
        s["classic_full"]=A.checked(source,info["sha256"])
        state=A.load(source)
        check_source(state,s["references"]["new_cache"]["expected_sha256"])
        if len(state["anchor_xyz"])<s["B"]: raise ValueError("insufficient classic points for frozen B")
        if state["provenance"]["protocol_sha256"]!=ap["source_protocols"]["matrix"]["sha256"]:
            raise ValueError("classic prior/observation protocol mismatch")
        s["A_snapshot"]=A.checked(A_ROOT/"query_inputs"/(scene+".json"))
    A.write_new(root/"protocol.json",{"design":DESIGN,"scenes":scenes,"code_sha256":code_hashes(),
          "A_protocol":A.checked(A_ROOT/"protocol.json"),"A_report":A.checked(A_ROOT/"report.json")})

def protocol(root):
    p=A.read(Path(root)/"protocol.json")
    if p["design"]!=DESIGN or p["code_sha256"]!=code_hashes(): raise ValueError("frozen B1 implementation changed")
    A.checked(p["A_protocol"]["path"],p["A_protocol"]["sha256"])
    A.checked(p["A_report"]["path"],p["A_report"]["sha256"])
    return p

def select(root,scene):
    root=Path(root);p=protocol(root);s=p["scenes"][scene]
    src=s["classic_full"];A.checked(src["path"],src["sha256"]);state=A.load(src["path"])
    check_source(state,s["references"]["new_cache"]["expected_sha256"])
    rows=spatial_rows(state["anchor_xyz"].numpy(),s["B"])
    target=root/"maps"/scene/CONFIG/"deployment_map.pt"
    info=save_map(target,state["anchor_xyz"].numpy(),state["anchor_features"].numpy(),rows,
                  dict(state["provenance"],B=s["B"],stage_b1_protocol_sha256=sha256(root/"protocol.json"),
                       source_full_sha256=src["sha256"],selection=DESIGN["selection"]))
    check=A.load(target)
    if not torch.equal(check["anchor_xyz"],state["anchor_xyz"][rows]) or not torch.equal(check["anchor_features"],state["anchor_features"][rows]):
        raise ValueError("selection changed numerical data")
    A.write_new(target.parent/"selection.json",info)

def evaluate(root,scene,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root);s=p["scenes"][scene]
    dataset=ColmapDataset(s["dataset"],load_masks=True)
    cameras=dataset.split("test")
    if [c.image_name for c in cameras]!=s["test_names"] or any(dataset.valid_mask(c) is None for c in cameras):
        raise ValueError("query registry or masks incomplete")
    A.checked(s["A_snapshot"]["path"],s["A_snapshot"]["sha256"])
    snap=evaluation_snapshot(root,scene,dataset,s)
    same_snapshot(A.read(root/"query_inputs"/(scene+".json")),A.read(s["A_snapshot"]["path"]))
    sel=A.read(root/"maps"/scene/CONFIG/"selection.json");A.checked(sel["path"],sel["sha256"])
    output=root/"evaluations"/scene/CONFIG
    run(["--dataset",s["dataset"],"--map",sel["path"],"--output",str(output),
         "--image-domain","rectified","--images","processed","--mask-policy","explicit_valid_mask",
         "--rectification-contract","opencv_same_k","--seed","2026","--device",device])
    if evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:
        raise ValueError("query snapshot changed")
    manifest=A.read(output/"manifest.json")
    ref=A.read(s["historical_evaluation"]["manifest.json"]["path"])
    for k in ("dataset","images","image_domain","split","seed","queries","calibration","config"):
        if manifest[k]!=ref[k]: raise ValueError("B1 evaluator configuration changed: "+k)
    A.write_new(output/"attestation.json",{"protocol_sha256":sha256(root/"protocol.json"),
        "query_inputs_sha256":snap,"map_sha256":sel["sha256"],"fresh_evaluation":True,"scene":scene,
        "configuration":CONFIG,"files":{n:A.checked(output/n) for n in ("results.json","summary.json","manifest.json")}})
    A.verify_evaluation(output,scene,CONFIG,p,sha256(root/"protocol.json"),snap,sel)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);scenes={}
    for scene in A.BUDGET:
        folder=root/"evaluations"/scene/CONFIG;sel=A.read(root/"maps"/scene/CONFIG/"selection.json")
        A.verify_evaluation(folder,scene,CONFIG,p,sha256(root/"protocol.json"),sha256(root/"query_inputs"/(scene+".json")),sel)
        rows=A.read(folder/"results.json")
        if [r["image_name"] for r in rows]!=p["scenes"][scene]["test_names"]: raise ValueError("missing B1 queries")
        v=summarize(rows,A.read(folder/"summary.json")["loop_wall_seconds"])
        v.update(map_landmarks=sel["count"],map_bytes=sel["bytes"]);scenes[scene]=v
    keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent","p90_te_cm",
          "failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
    baseline=A.read(p["A_report"]["path"])["results"]
    A.write_new(root/"report.json",json_safe({"complete":True,"scope":DESIGN["scope"],
       "classic_spatial_B":{"per_scene":scenes,"scene_equal":{k:A.scene_mean([s[k] for s in scenes.values()]) for k in keys}},
       "frozen_A_comparisons":{k:baseline[k] for k in ("new_native","new_spatial_B")},
       "causal_limit":DESIGN["comparison"]}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","select","evaluate","report"])
    p.add_argument("--root",type=Path,required=True);p.add_argument("--scene",choices=list(A.BUDGET));p.add_argument("--device",default="cuda:0")
    a=p.parse_args();torch.set_num_threads(4)
    if a.command=="prepare":prepare(a.root)
    elif a.command=="select":select(a.root,a.scene)
    elif a.command=="evaluate":evaluate(a.root,a.scene,a.device)
    else:report(a.root)
if __name__=="__main__":main()

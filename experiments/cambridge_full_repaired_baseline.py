"""Only two repaired-StMary full-map evaluations; reuse verified four-scene full results."""
from __future__ import annotations
import argparse,copy,time
from pathlib import Path
import torch
from anygsloc.artifacts import validate
from experiments.stmary_pose_repair import mechanism as S
A=S.A;M=S.M
REPAIRED=Path("/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v2/mechanism")
METHODS=("anygsloc","classic")
SCENES=(*A.BUDGET,S.SCENE)
DESIGN={"schema":"repaired_puregs_unmasked_full_baseline_v1","scene_order":list(SCENES),
 "input":"pureGS, vanilla3DGS, mapping SP2048 unmasked; no original RGB",
 "data":"original four scenes plus StMary repaired v2 camera/prior/GT",
 "maps":"existing complete full maps; no native/spatial selection or export",
 "new_evaluations":"StMary full anygsloc and classic only",
 "query":"original rectified semantic-on runner, seed2026, 2048 SP and scene calibration",
 "aggregation":"equal-weight mean of five per-scene metrics, all failures retained",
 "timing":"parallel unwarmed telemetry; not a formal benchmark"}

def hashes():
    h=S.code_hashes();p=Path(__file__).resolve();h[str(p.relative_to(A.ROOT))]=S.sha256(p);return h

def same_snapshot(a,b):
    keys=("mapping_registry","calibration","mask_sha256","queries")
    if any(a[k]!=b[k] for k in keys):raise ValueError("baseline query snapshot differs")

def old_cell(scene,method,p,rehash_queries=False):
    root=A.MATRIX;s=p["scenes"][scene];folder=root/"evaluations"/scene/"pure_gs"/method/"full"
    a=A.read(folder/"input_matrix_attestation.json")
    if (a["protocol_sha256"]!=S.sha256(root/"protocol.json") or a["query_semantics"] is not True or
        a["mask_sha256"]!=s["mask_sha256"] or a["query_inputs_sha256"]!=S.sha256(root/"query_inputs"/(scene+".json"))):
        raise ValueError("old full evaluation protocol differs")
    refs={n:S.evidence(folder/n,h) for n,h in a["output_sha256"].items()}
    refs["attestation"]=S.evidence(folder/"input_matrix_attestation.json")
    ref=A.read(root/"maps"/scene/"pure_gs"/method/"summary.json")["full"]
    state=validate(A.load(A.checked(ref["path"],ref["sha256"])["path"]))
    if len(state["anchor_xyz"])!=ref["count"]:raise ValueError("old full count differs")
    manifest=A.read(folder/"manifest.json")
    if manifest["map_sha256"]!=ref["sha256"] or manifest["seed"]!=2026 or manifest["config"]["mask_policy"]!="explicit_valid_mask":
        raise ValueError("old full evaluator differs")
    if manifest["queries"]!=s["test_names"] or manifest["rectification_contract"]!="opencv_same_k":
        raise ValueError("old full query identities differ")
    if rehash_queries:
        from anygsloc.data.datasets import ColmapDataset
        if not (root/"query_inputs"/(scene+".json")).is_file():raise ValueError("old snapshot missing")
        M.evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)
    return {"map":ref,"evaluation":refs,"snapshot":S.evidence(root/"query_inputs"/(scene+".json"))}

def prepare(root):
    from anygsloc.data.datasets import ColmapDataset
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    old=M.protocol(A.MATRIX);repaired=S.protocol(REPAIRED)
    review=A.read(REPAIRED/"independent_final_review.json")
    if review.get("complete") is not True or review.get("verdict")!="PASS":raise ValueError("repaired baseline not independently accepted")
    source_report=S.evidence(REPAIRED/"report.json",review["report"]["sha256"])
    reuse={scene:{method:old_cell(scene,method,old,rehash_queries=method=="anygsloc") for method in METHODS} for scene in A.BUDGET}
    full={}
    for method in METHODS:
        summary,state,cache=S.bound_full(REPAIRED,method);validate(state)
        full[method]=summary["full"]
        if len(state["anchor_xyz"])!=full[method]["count"]:raise ValueError("repaired full count differs")
    s=copy.deepcopy(repaired["scenes"][S.SCENE])
    evidence=[S.evidence(A.MATRIX/"protocol.json"),S.evidence(REPAIRED/"protocol.json"),
              S.evidence(REPAIRED/"independent_final_review.json"),source_report]
    A.write_new(root/"protocol.json",{"design":DESIGN,"code_sha256":hashes(),"scenes":{S.SCENE:s},
        "reused":reuse,"full":full,"evidence":evidence,
        "reference_native_same_frontend":A.read(source_report["path"])["results"]["new_native"]})
    snapshot=M.evaluation_snapshot(root,S.SCENE,ColmapDataset(s["dataset"],load_masks=False),s)
    same_snapshot(A.read(root/"query_inputs"/(S.SCENE+".json")),A.read(REPAIRED/"query_inputs"/(S.SCENE+".json")))
    A.write_new(root/"preflight.json",{"complete":True,"protocol_sha256":S.sha256(root/"protocol.json"),
        "full_maps_directly_valid":True,"export_or_selection_performed":False,
        "query_inputs_sha256":snapshot,"cuda_initialized":torch.cuda.is_initialized()})

def protocol(root):
    root=Path(root);p=A.read(root/"protocol.json")
    if p["design"]!=DESIGN or p["code_sha256"]!=hashes():raise ValueError("full baseline code/design changed")
    for ref in p["evidence"]:S.unchanged(ref,True)
    ready=A.read(root/"preflight.json")
    if not ready["complete"] or ready["protocol_sha256"]!=S.sha256(root/"protocol.json"):raise ValueError("preflight incomplete")
    return p

def evaluate(root,method,device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.runner import main as run
    root=Path(root);p=protocol(root);s=p["scenes"][S.SCENE]
    if method not in METHODS:raise ValueError("method outside two approved cells")
    if torch.device(device).type!="cuda" or torch.device(device).index is None:raise ValueError("explicit CUDA device required")
    ref=p["full"][method];A.checked(ref["path"],ref["sha256"]);output=root/"evaluations"/method
    if output.exists():raise FileExistsError(output)
    dataset=ColmapDataset(s["dataset"],load_masks=True)
    if [c.image_name for c in dataset.split("test")]!=s["test_names"] or any(dataset.valid_mask(c) is None for c in dataset.split("test")):
        raise ValueError("query-on inputs incomplete")
    snap=M.evaluation_snapshot(root,S.SCENE,dataset,s)
    same_snapshot(A.read(root/"query_inputs"/(S.SCENE+".json")),A.read(REPAIRED/"query_inputs"/(S.SCENE+".json")))
    run(["--dataset",s["dataset"],"--map",ref["path"],"--output",str(output),"--image-domain","rectified",
         "--images","processed","--mask-policy","explicit_valid_mask","--rectification-contract","opencv_same_k",
         "--seed","2026","--device",device])
    if M.evaluation_snapshot(root,S.SCENE,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:raise ValueError("query inputs changed")
    protocol(root)
    A.write_new(output/"attestation.json",{"protocol_sha256":S.sha256(root/"protocol.json"),
       "query_inputs_sha256":snap,"map_sha256":ref["sha256"],"fresh_evaluation":True,"scene":S.SCENE,"configuration":method,
       "files":{n:A.checked(output/n) for n in ("results.json","manifest.json","summary.json")}})
    A.verify_evaluation(output,S.SCENE,method,p,S.sha256(root/"protocol.json"),snap,ref)

def report(root):
    from anygsloc.evaluation.runner import summarize,json_safe
    root=Path(root);p=protocol(root);result={}
    keys=("median_te_cm","median_ae_deg","recall_5cm_5deg_percent","recall_15cm_5deg_percent",
          "p90_te_cm","failed_query_count","catastrophic_100cm_count","map_landmarks","map_bytes")
    for method in METHODS:
        per={}
        for scene in SCENES:
            if scene==S.SCENE:
                folder=root/"evaluations"/method;ref=p["full"][method]
                A.verify_evaluation(folder,scene,method,p,S.sha256(root/"protocol.json"),
                    S.sha256(root/"query_inputs"/(scene+".json")),ref)
                names=p["scenes"][scene]["test_names"]
            else:
                cell=p["reused"][scene][method];ref=cell["map"]
                for v in cell["evaluation"].values():S.unchanged(v,True)
                S.unchanged(cell["snapshot"],True)
                folder=Path(cell["evaluation"]["results.json"]["path"]).parent
                names=A.read(folder/"manifest.json")["queries"]
            rows=A.read(folder/"results.json")
            if [r["image_name"] for r in rows]!=names:raise ValueError("missing full-map queries")
            v=summarize(rows,A.read(folder/"summary.json")["loop_wall_seconds"])
            v.update(map_landmarks=ref["count"],map_bytes=ref["bytes"]);per[scene]=v
        result[method]={"per_scene":per,"scene_equal":{k:A.scene_mean([v[k] for v in per.values()]) for k in keys}}
    A.write_new(root/"report.json",json_safe({"complete":True,"design":DESIGN,"results":result,
        "native_same_unmasked_reference":p["reference_native_same_frontend"]}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","evaluate","report"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--method",choices=METHODS);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=="prepare":prepare(a.root)
    elif a.stage=="evaluate":evaluate(a.root,a.method,a.device)
    else:report(a.root)
if __name__=="__main__":main()

"""CPU preparation for frozen 5-scene native-initialization, query-semantic-on evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

BASE=Path("/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915")
BENCH=Path("/mnt/pool/sqy/anygsloc_cambridge_deployment_benchmark_20261011_v1/protocol.json")
SCENES=("ShopFacade","OldHospital","KingsCollege","GreatCourt","StMarysChurch")
METHODS=("STDLoc","ULFLoc")
ROOT=Path("/mnt/pool/sqy/anygsloc_native_baseline_stmary_repair_20261012_v1")
REPO=Path(__file__).resolve().parents[2]
PROTOCOL_NAME="protocol_v2.json"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def ref(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def write_new(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(value,f,indent=2)

def verify(r):
    if sha(r["path"])!=r["sha256"]:raise ValueError("changed input: "+r["path"])

def replace_value(argv,flag,value):
    if argv.count(flag)!=1:raise ValueError("ambiguous argument "+flag)
    a=list(argv);a[a.index(flag)+1]=str(value);return a

def asset_names(method):
    common=["cfg_args","point_cloud/iteration_30000/point_cloud.ply"]
    if method=="STDLoc":return common+["detector/sampled_idx.pkl","detector/30000_detector.pth"]
    if method=="ULFLoc":return common+["test/keypoints_sampled_idx.pkl","test/keypoints_features.pkl"]
    raise ValueError(method)

def expected_cell(p, key, protocol_path):
    c=p["cells"][key];scene=p["scenes"][c["scene"]]
    out=Path(protocol_path).parent/"cells"/key;view=out/"model"
    argv=c["original_job"]["command"][2:]
    for flag,v in [("--source",scene["source"]),("--receipt",out/"receipt.json"),("-s",scene["source"]),("-m",view),("--images",Path(scene["source"])/"processed")]:
        argv=replace_value(argv,flag,v)
    argv=argv[:argv.index("--")]+["--run-tag","query_semantic_on"]+argv[argv.index("--"):]
    repo=Path(argv[argv.index("--repo")+1])
    return dict(wrapper_argv=argv,source=scene["source"],receipt=str(out/"receipt.json"),model_view=str(view),
                native_output=str(out/"native_output" if c["method"]=="STDLoc" else view/"query_semantic_on"),
                method=c["method"],scene=c["scene"],mapping_semantics=c["mapping_semantics"],query_semantics=True,
                seed=2026,query_names=scene["query_names"],env=c["original_job"]["env"],
                executable=c["original_job"]["command"][0])

def validate_contract(c,p,protocol_path):
    expected=expected_cell(p,c["case"],protocol_path)
    for field,value in expected.items():
        if c.get(field)!=value:raise ValueError("contract differs from frozen native recipe: "+field)
    original=p["cells"][c["case"]];method=original["method"];argv=expected["wrapper_argv"]
    repo=Path(argv[argv.index("--repo")+1])
    paths={"wrapper":Path(original["original_job"]["command"][1]),
           "native_entry":repo/("stdloc.py" if method=="STDLoc" else "ulfloc.py"),
           "native_config":Path(argv[argv.index("--cfg")+1])}
    for label,path in paths.items():
        if c[label]["path"]!=str(path) or c[label]["sha256"]!=p["code_sha256"][str(path)]:
            raise ValueError("contract native dependency differs from protocol")
    adapter=str(REPO/"experiments/native_baseline_repair/query_semantic_adapter.py")
    if c["adapter_sha256"]!=p["code_sha256"][adapter]:raise ValueError("adapter detached from protocol")
    scene=p["scenes"][original["scene"]]
    if c["scene_inputs"]!=scene["inputs"] or c["query_snapshot"]!=scene["reference_snapshot"]:
        raise ValueError("query snapshot differs from protocol")
    if set(c["model_assets"])!=set(asset_names(method)):raise ValueError("missing native inference asset")
    model=Path(original["model_source"])
    for rel,r in c["model_assets"].items():
        if r["path"]!=str(model/rel):raise ValueError("model asset source differs from frozen cell")
    if original["scene"]=="StMarysChurch":
        ap=Path(p["training_protocol"]["path"]).parent/"acceptance"/(c["case"].removesuffix("_eval")+"_train.json")
        if not c.get("training_acceptance") or c["training_acceptance"]["path"]!=str(ap):
            raise ValueError("repaired training acceptance detached")
        verify(c["training_acceptance"])
        verify_accepted_model(json.loads(ap.read_text()),c["case"],method,c["mapping_semantics"],model,c["model_assets"],p["training_protocol"])
    elif c.get("training_acceptance") is not None:raise ValueError("old-scene source contract changed")

def verify_accepted_model(a, key, method, semantic, model, assets, training_ref):
    required=dict(complete=True,case=key.removesuffix("_eval")+"_train",method=method,semantics=semantic,seed=2026,iterations=30000)
    for field,value in required.items():
        if a.get(field)!=value:raise ValueError("training acceptance identity mismatch: "+field)
    if a.get("training_protocol",{}).get("sha256")!=training_ref["sha256"]:
        raise ValueError("accepted training protocol mismatch")
    verify(a["training_protocol"])
    field_map={"cfg_args":"cfg_args","point_cloud/iteration_30000/point_cloud.ply":"ply"}
    field_map.update({"test/keypoints_sampled_idx.pkl":"indices","test/keypoints_features.pkl":"features"} if method=="ULFLoc"
                     else {"detector/sampled_idx.pkl":"indices","detector/30000_detector.pth":"detector"})
    for rel,value in assets.items():
        accepted=a["artifacts"][field_map[rel]]
        if Path(accepted["path"]).resolve()!=Path(model/rel).resolve():
            raise ValueError("acceptance bound to a different model asset")
        if accepted["sha256"]!=value["sha256"] or accepted["bytes"]!=value["bytes"]:
            raise ValueError("model no longer equals its accepted training product")
        verify(accepted)

def historical_jobs():
    found={}
    # gpu0 is the original full 120-job plan (20 evaluations); gpu1 contains two later delegated duplicates.
    for path in [BASE/"gpu0/jobs.json"]:
        for job in json.loads(path.read_text()):
            key=job["id"]
            if key in found and found[key]!=job:raise ValueError("conflicting historical jobs")
            found[key]=job
    return found

def prepare(r):
    target=r/"query_semantic_on"
    if (target/PROTOCOL_NAME).exists():raise FileExistsError(target/PROTOCOL_NAME)
    bench=json.loads(BENCH.read_text());train=json.loads((r/"training_protocol_v4.json").read_text())
    scenes={}
    for scene in SCENES:
        old=bench["scenes"][scene];snap=old["reference_snapshot"];verify(snap)
        q=json.loads(Path(snap["path"]).read_text());source=(r/"cambridge_data"/scene if scene=="StMarysChurch" else BASE/"cambridge_data"/scene)
        names=sorted(x["name"] for x in q["queries"])
        if len(names)!=len(set(names)):raise ValueError("duplicate query")
        inputs={p:ref(source/p) for p in ["dataset_train.txt","dataset_test.txt","sparse/0/images.bin","sparse/0/cameras.bin","processed/masks.pkl"]}
        if inputs["processed/masks.pkl"]["sha256"]!=q["mask_sha256"]:raise ValueError("Any/native mask input mismatch")
        for item in q["queries"]:
            if sha(source/"processed"/item["name"])!=item["image_sha256"]:raise ValueError("Any/native query RGB mismatch")
        actual=sorted(line.split()[0] for line in (source/"dataset_test.txt").read_text().splitlines()[3:] if line.strip())
        if names!=actual:raise ValueError("native test list differs from frozen Any query names")
        import numpy as np
        sys.path.insert(0,str(BASE/"code"))
        from anygsloc.data.colmap import read_extrinsics_binary, qvec2rotmat
        ext={v.name:v for v in read_extrinsics_binary(str(source/"sparse/0/images.bin")).values()}
        for item in q["queries"]:
            v=ext[item["name"]];pose=np.eye(4,dtype=np.float32);pose[:3,:3]=qvec2rotmat(v.qvec);pose[:3,3]=v.tvec
            if not np.array_equal(pose,np.asarray(item["pose_w2c"],dtype=np.float32)):
                raise ValueError("native/Any query GT differs")
        scenes[scene]=dict(source=str(source),query_names=names,inputs=inputs,reference_snapshot=snap,
                           query_rgb_count=len(names),query_rgb_hashes_verified=True)
    code=dict(train["native_code_sha256"])
    for p in [Path(__file__),REPO/"experiments/native_baseline_repair/query_semantic_adapter.py",
              REPO/"experiments/native_baseline_repair/launch_query.py",
              REPO/"tests/test_native_baseline_query_semantics.py"]:
        code[str(p)]=sha(p)
    # The wrapper imports its original frozen Any data helpers through original PYTHONPATH.
    for rel in ["anygsloc/data/colmap.py","anygsloc/data/geometric_validity.py"]:
        p=BASE/"code"/rel;code[str(p)]=sha(p)
    for method in METHODS:
        for rel in [f"configs/{method.lower()}_cambridge.yaml","encoders/sp_encoder/weights/superpoint_v1.pth"]:
            path=BASE/"baselines"/method/rel;code[str(path)]=sha(path)
    for p,h in code.items():
        if sha(p)!=h:raise ValueError("frozen code mismatch")
    jobs=historical_jobs();cells={}
    for scene in SCENES:
        for method in METHODS:
            for semantic in ("off","on"):
                key=f"{method}_{scene}_mapping_semantics_{semantic}_eval"
                j=jobs[key];model=(r/"models" if scene=="StMarysChurch" else BASE/"baseline_models")/key.removesuffix("_eval")
                cells[key]=dict(scene=scene,method=method,mapping_semantics=semantic,original_job=j,model_source=str(model),
                                status="requires accepted repaired training" if scene=="StMarysChurch" else "historical asset reuse")
    value=dict(schema="native_baseline_query_on_protocol_v1",created_unix=time.time(),seed=2026,query_semantics=True,
               initialization="user-selected native SfM initialization; test-supported and mixed-track initial geometry retained and disclosed",
               scope="5 scenes x STDLoc/ULFLoc x mapping semantics off/on; all query semantics on",
               method_variant=dict(STDLoc="native features, selector, pose solver and dense refinement; added external RGB query mask preprocessing",
                                   ULFLoc="native query-mask branch enabled; native features, solver and dense refinement"),
               scientific_native_configuration="original method YAML and original model parameters; no forced common feature/PnP budgets",
               timing="unwarmed telemetry only; native localize includes native GT diagnostics, excludes RGB/mask preprocessing and segmentation network",
               scenes=scenes,cells=cells,code_sha256=code,original_jobs=ref(BASE/"gpu0/jobs.json"),any_query_reference=ref(BENCH),training_protocol=ref(r/"training_protocol_v4.json"))
    write_new(target/PROTOCOL_NAME,value);print(json.dumps(dict(path=str(target/PROTOCOL_NAME),sha256=sha(target/PROTOCOL_NAME),cells=len(cells))))

def cell(r,key):
    root=r/"query_semantic_on";pp=root/PROTOCOL_NAME;p=json.loads(pp.read_text());c=p["cells"][key]
    for path,h in p["code_sha256"].items():
        if sha(path)!=h:raise ValueError("code changed: "+path)
    scene=p["scenes"][c["scene"]]
    for item in scene["inputs"].values():verify(item)
    verify(scene["reference_snapshot"])
    model=Path(c["model_source"])
    acceptance=None
    assets={rel:ref(model/rel) for rel in asset_names(c["method"])}
    if c["scene"]=="StMarysChurch":
        ap=r/"acceptance"/(key.removesuffix("_eval")+"_train.json")
        a=json.loads(ap.read_text())
        verify_accepted_model(a,key,c["method"],c["mapping_semantics"],model,assets,p["training_protocol"])
        acceptance=ref(ap)
    out=root/"cells"/key
    if out.exists():raise FileExistsError(out)
    model_view=out/"model";model_view.mkdir(parents=True)
    for rel,value in assets.items():
        target=model_view/rel;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(value["path"])
    argv=c["original_job"]["command"][2:]
    for flag,v in [("--source",scene["source"]),("--receipt",out/"receipt.json"),("-s",scene["source"]),("-m",model_view),("--images",Path(scene["source"])/"processed")]:
        argv=replace_value(argv,flag,v)
    argv=argv[:argv.index("--")]+["--run-tag","query_semantic_on"]+argv[argv.index("--"):]
    repo=Path(argv[argv.index("--repo")+1]);entry=repo/("stdloc.py" if c["method"]=="STDLoc" else "ulfloc.py")
    native_output=(out/"native_output" if c["method"]=="STDLoc" else model_view/"query_semantic_on")
    config=Path(argv[argv.index("--cfg")+1])
    contract=dict(schema="native_baseline_query_semantic_on_cell_v1",case=key,method=c["method"],scene=c["scene"],
                  mapping_semantics=c["mapping_semantics"],query_semantics=True,seed=2026,source=scene["source"],
                  query_names=scene["query_names"],adapter_sha256=sha(REPO/"experiments/native_baseline_repair/query_semantic_adapter.py"),
                  wrapper_argv=argv,wrapper=ref(BASE/"code/tools/baseline_protocol_v2.py"),native_entry=ref(entry),
                  receipt=str(out/"receipt.json"),native_output=str(native_output),model_view=str(model_view),
                  model_assets=assets,native_config=ref(config),scene_inputs=scene["inputs"],query_snapshot=scene["reference_snapshot"],
                  protocol=ref(pp),training_acceptance=acceptance,env=c["original_job"]["env"],executable=c["original_job"]["command"][0])
    validate_contract(contract,p,pp)
    write_new(out/"contract.json",contract);print(json.dumps(dict(path=str(out/"contract.json"),sha256=sha(out/"contract.json"))))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",type=Path,default=ROOT)
    sub=ap.add_subparsers(dest="action",required=True);sub.add_parser("prepare")
    ca=sub.add_parser("cell");ca.add_argument("--case",required=True)
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.root)
    else:cell(a.root,a.case)
if __name__=="__main__":main()

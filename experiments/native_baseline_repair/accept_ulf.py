"""CPU-only acceptance of a completed native ULF training and its deployment assets."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import pickle
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def ref(path):
    path=Path(path)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)

class CPUUnpickler(pickle.Unpickler):
    """Only trusted local native products; preserve tensor values and map storage to CPU."""
    def find_class(self,module,name):
        if module=="torch.storage" and name=="_load_from_bytes":
            return lambda b:torch.load(io.BytesIO(b),map_location="cpu",weights_only=False)
        return super().find_class(module,name)

def load_cpu(path):
    with Path(path).open("rb") as f:return CPUUnpickler(f).load()

def check_arrays(indices, features, point_count):
    idx=torch.as_tensor(indices)
    feat=torch.as_tensor(features)
    if idx.dtype not in (torch.int32,torch.int64) or idx.ndim!=1:
        raise ValueError("native landmark indices must be a one-dimensional integer array")
    if idx.numel()!=20000 or idx.unique().numel()!=idx.numel():
        raise ValueError("expected 20000 unique native sampled landmarks")
    if idx.min()<0 or idx.max()>=point_count:raise ValueError("landmark outside native PLY")
    if tuple(feat.shape)!=(20000,256) or not torch.isfinite(feat).all():
        raise ValueError("native features must be finite 20000x256")
    return dict(landmarks=idx.numel(),feature_shape=list(feat.shape),feature_dtype=str(feat.dtype),
                feature_zero_rows=int((torch.linalg.vector_norm(feat.float(),dim=1)==0).sum()))

def parse_cfg(path):
    node=ast.parse(Path(path).read_text(),mode="eval").body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!="Namespace":
        raise ValueError("unexpected native cfg_args")
    return {k.arg:ast.literal_eval(k.value) for k in node.keywords}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--semantics",choices=["off","on"],required=True)
    ap.add_argument("--terminal-evidence",type=Path,required=True)
    a=ap.parse_args();r=a.root;key=f"ULFLoc_StMarysChurch_mapping_semantics_{a.semantics}_train"
    terminal=json.loads(a.terminal_evidence.read_text())
    if terminal.get("terminal") is not True or terminal.get("exit_code")!=0 or terminal.get("case")!=key:
        raise ValueError("native Runner job not proven successful and terminal")
    protocol_path=r/"training_protocol_v4.json";protocol=json.loads(protocol_path.read_text())
    for p,expected in protocol["native_code_sha256"].items():
        if sha(p)!=expected:raise ValueError("frozen training code changed: "+p)
    case=protocol["jobs"][key];model=Path(case["fresh_model_path"])
    receipt_path=Path(case["fresh_receipt_path"]);receipt=json.loads(receipt_path.read_text())
    for k,v in dict(status="complete",method="ULFLoc",phase="train",mapping_semantics=a.semantics,query_semantics=False,seed=2026,fusion_batch_size=1).items():
        if receipt.get(k)!=v:raise ValueError("native receipt mismatch: "+k)
    if receipt.get("postprocess_resume_ply") or receipt.get("postprocess_resume_indices"):
        raise ValueError("postprocessing-only run cannot satisfy full training")
    if [s["stage"] for s in receipt["build_stages"]]!=["keypoints_vote","feature_fusion"]:
        raise ValueError("native postprocessing stages missing")
    cfg=parse_cfg(model/"cfg_args")
    if cfg["source_path"]!=protocol["source"] or cfg["model_path"]!=str(model):
        raise ValueError("native source/model binding changed")
    ply=model/"point_cloud/iteration_30000/point_cloud.ply"
    from plyfile import PlyData
    vertices=PlyData.read(str(ply))["vertex"]
    if len(vertices)==0:raise ValueError("empty native PLY")
    for field in vertices.data.dtype.names:
        import numpy as np
        if not np.isfinite(vertices.data[field]).all():raise ValueError("non-finite PLY field "+field)
    idx=model/"test/keypoints_sampled_idx.pkl";feat=model/"test/keypoints_features.pkl"
    arrays=check_arrays(load_cpu(idx),load_cpu(feat),len(vertices))
    log=r/"launches"/key/"native.log"
    text=log.read_text(errors="replace")
    if "30000/30000" not in text:raise ValueError("missing complete native iteration progress")
    artifacts={k:ref(p) for k,p in dict(ply=ply,indices=idx,features=feat,cfg_args=model/"cfg_args",cameras=model/"cameras.json",native_receipt=receipt_path,launch=r/"launches"/key/"launch.json",log=log).items()}
    result=dict(schema="native_ulf_training_acceptance_v1",complete=True,case=key,method="ULFLoc",
                semantics=a.semantics,seed=2026,iterations=30000,point_count=len(vertices),**arrays,
                initialization="original native SfM including mixed and test-supported tracks; 419837 points",
                training_protocol=ref(protocol_path),terminal_evidence=ref(a.terminal_evidence),
                artifacts=artifacts,wall_seconds=receipt["wall_seconds"],build_stages=receipt["build_stages"],
                peak_rss_kib=receipt["peak_rss_kib"],peak_torch_gpu_bytes=receipt["peak_torch_gpu_bytes"],
                validation_device="cpu",gpu_validation=False)
    out=r/"acceptance"/f"{key}.json";out.parent.mkdir(exist_ok=True)
    with out.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(path=str(out),sha256=sha(out),point_count=len(vertices),**arrays,wall_seconds=receipt["wall_seconds"])))
if __name__=="__main__":main()

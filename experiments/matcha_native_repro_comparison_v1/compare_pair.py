"""CPU-only complete native U1/U2 difference inventory; no acceptance tolerance."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
import numpy as np
import torch
torch.set_num_threads(1)

V1 = Path("/root/STDLoc/experiments/matcha_exact_resume")
sys.path.insert(0, str(V1))
import exact_resume as er
import native_n8_launcher as original_launcher
CHUNK = 1_000_000
MAX_EXAMPLES = 16

def require(ok, text):
    if not ok: raise RuntimeError("native-AA-compare: " + text)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def scalar(value):
    if isinstance(value, np.generic):
        return {"type":type(value).__name__, "dtype":value.dtype.str, "bits":value.tobytes().hex()}
    if isinstance(value, float):
        return {"type":type(value).__name__, "repr":repr(value),
                "float64_bits":struct.pack("<d",float(value)).hex()}
    if isinstance(value, np.generic):
        return {"type":type(value).__name__, "dtype":value.dtype.str, "bits":value.tobytes().hex()}
    if value is None or type(value) in (str,bool,int): return value
    return {"type":type(value).__name__, "repr":repr(value)}

def array_summary(a, b):
    """Raw-byte element comparison; chunk slices also bound noncontiguous copies."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            return {"kind":"tensor_descriptor", "expected":[str(a.dtype),list(a.shape)],
                    "actual":[str(getattr(b,"dtype",type(b))),list(getattr(b,"shape",()))]}
        require(a.device.type == b.device.type == "cpu", "CPU tensors required")
        bfloat = a.dtype == torch.bfloat16
        x = a.detach().view(torch.int16).numpy().view(np.uint16) if bfloat else a.detach().numpy()
        y = b.detach().view(torch.int16).numpy().view(np.uint16) if bfloat else b.detach().numpy()
        dtype_name = str(a.dtype)
    else:
        if not isinstance(b,np.ndarray) or a.dtype != b.dtype or a.shape != b.shape:
            return {"kind":"array_descriptor", "expected":[str(a.dtype),list(a.shape)],
                    "actual":[str(getattr(b,"dtype",type(b))),list(getattr(b,"shape",()))]}
        x,y,bfloat,dtype_name = a,b,False,str(a.dtype)
    count, first = 0, []
    max_abs, max_ulp = 0., 0
    nfa,nfb,nonfinite_mismatches = 0,0,0
    floating = bfloat or x.dtype.kind == "f"
    for start in range(0,x.size,CHUNK):
        # ndarray.flat slices produce at most CHUNK elements even with noncontiguous input.
        ac = np.ascontiguousarray(x.flat[start:start+CHUNK])
        bc = np.ascontiguousarray(y.flat[start:start+CHUNK])
        ab = ac.view(np.uint8).reshape(-1,ac.dtype.itemsize)
        bb = bc.view(np.uint8).reshape(-1,bc.dtype.itemsize)
        mask = np.any(ab != bb,axis=1)
        count += int(np.count_nonzero(mask))
        if len(first) < MAX_EXAMPLES:
            for i in np.flatnonzero(mask)[:MAX_EXAMPLES-len(first)]:
                first.append({"flat_index":start+int(i), "expected_bits":ab[i].tobytes().hex(),
                              "actual_bits":bb[i].tobytes().hex()})
        if floating:
            if bfloat:
                av=(ac.astype(np.uint32)<<16).view(np.float32)
                bv=(bc.astype(np.uint32)<<16).view(np.float32)
                raw_a,raw_b,bits=ac,bc,16
            else:
                av,bv=ac,bc
                bits=ac.dtype.itemsize*8
                raw_a=ac.view(np.dtype("u"+str(ac.dtype.itemsize)))
                raw_b=bc.view(np.dtype("u"+str(bc.dtype.itemsize)))
            fa,fb=np.isfinite(av),np.isfinite(bv)
            nfa += int(np.count_nonzero(~fa));nfb += int(np.count_nonzero(~fb))
            nonfinite_mismatches += int(np.count_nonzero(mask & ~(fa & fb)))
            finite=fa & fb
            if np.any(finite):
                differences=np.abs(av[finite].astype(np.float64)-bv[finite].astype(np.float64))
                max_abs=max(max_abs,float(np.max(differences)))
                sign=np.array(1 << (bits-1),dtype=raw_a.dtype)
                oa=np.where((raw_a & sign)!=0,~raw_a,raw_a | sign)
                ob=np.where((raw_b & sign)!=0,~raw_b,raw_b | sign)
                distance=np.maximum(oa[finite],ob[finite])-np.minimum(oa[finite],ob[finite])
                max_ulp=max(max_ulp,int(np.max(distance)))
    if count == 0: return None
    result={"kind":"tensor_values" if isinstance(a,torch.Tensor) else "array_values",
            "dtype":dtype_name,"shape":list(x.shape),"elements":int(x.size),
            "differing_elements":count,"first_differences":first}
    if floating:
        result.update(max_finite_absolute_difference=max_abs if math.isfinite(max_abs) else None,
                      absolute_difference_overflow=not math.isfinite(max_abs),max_finite_ulp_distance=max_ulp,
                      expected_nonfinite=nfa,actual_nonfinite=nfb,nonfinite_mismatches=nonfinite_mismatches)
    return result

def differences(a, b, path="$"):
    output=[]
    def visit(x,y,p):
        if isinstance(x,(torch.Tensor,np.ndarray)):
            value=array_summary(x,y)
            if value is not None: output.append(dict(path=p,**value))
            return
        if type(x)!=type(y):
            output.append({"path":p,"kind":"type","expected":scalar(x),"actual":scalar(y)})
        elif isinstance(x,dict):
            for key in sorted(set(x)|set(y),key=repr):
                kp=p+"["+repr(key)+"]"
                if key not in x or key not in y:
                    output.append({"path":kp,"kind":"missing_key","expected_present":key in x,"actual_present":key in y})
                else: visit(x[key],y[key],kp)
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y): output.append({"path":p,"kind":"length","expected":len(x),"actual":len(y)})
            for i in range(min(len(x),len(y))): visit(x[i],y[i],p+"["+str(i)+"]")
        elif isinstance(x,np.generic):
            if x.dtype!=y.dtype or x.tobytes()!=y.tobytes():
                output.append({"path":p,"kind":"numpy_scalar","expected":scalar(x),"actual":scalar(y)})
        elif isinstance(x,float):
            if struct.pack("<d",x)!=struct.pack("<d",y):
                output.append({"path":p,"kind":"scalar_bits","expected":scalar(x),"actual":scalar(y)})
        elif x != y:
            output.append({"path":p,"kind":"scalar","expected":scalar(x),"actual":scalar(y)})
    visit(a,b,path)
    return output

def scalar_loss_diagnostics(a,b):
    rows=[]
    for step in range(1,9):
        ac=a[step-1]["loss_components"];bc=b[step-1]["loss_components"]
        for name in sorted(set(ac)&set(bc)):
            x,y=ac[name],bc[name]
            if not (isinstance(x,dict) and isinstance(y,dict) and x.get("dtype")==y.get("dtype")):continue
            dtype={"torch.float16":np.float16,"torch.float32":np.float32,"torch.float64":np.float64}.get(x.get("dtype"))
            if dtype is None or "bits" not in x or "bits" not in y:continue
            xa=np.frombuffer(bytes.fromhex(x["bits"]),dtype=dtype);ya=np.frombuffer(bytes.fromhex(y["bits"]),dtype=dtype)
            row=array_summary(xa,ya)
            if row: rows.append({"iteration":step,"component":name,**row})
    return rows

def normalized_runtime(proof):
    value=json.loads(json.dumps(proof))
    value["context"].pop("pid",None)
    value["context"].pop("allocated_bytes",None)
    return value

def registration():
    path=Path("/root/STDLoc/experiments/matcha_native_repro_diagnostic_v1/native_aa_launcher.py")
    require(sha(path)=="988e299f59629cfe2047428470c75464c11e07db6c7c5e4b40f4c0dd7b1a1e40","registered launcher changed")
    ms=importlib.util.spec_from_file_location("registered_native_aa_launcher",path)
    module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    return module

def validate_run_headers(spec,result,complete,before,after,replicate):
    registered=registration()
    registered.validate_spec(spec)
    require(result["schema"]=="matcha_native_aa_result_v1" and
            result["native_horizon"]==30000 and result["comparison_executed"] is False and
            result["production_continuation_allowed"] is False,"launcher completion schema")
    require(complete["schema"]=="matcha_native_aa_child_v1" and complete["replicate"]==replicate and
            complete["native_horizon"]==30000 and complete["observed_terminal"]==8 and
            complete["guard_used"] is False and complete["comparison_executed"] is False and
            complete["production_continuation_allowed"] is False,"child completion identity")
    for proof in (before,after):
        registered.validate_runtime_flags(proof)
        require(proof["original_helper_path"]==str(V1/"exact_resume.py"),"wrong observer helper")

def validate_terminal_receipt(path,roots):
    terminal=json.loads(Path(path).read_text())
    require(terminal["schema"]=="matcha_native_pair_runner_receipt_v1" and set(terminal["runs"])=={"u1","u2"},
            "official pair terminal receipt missing")
    job_ids=[]
    for name,root in roots.items():
        item=terminal["runs"][name]
        require(item["status"]=="completed" and item["exit_code"]==0 and item["job_id"].startswith("wc_job_") and
                item["root"]==str(root.resolve()) and item["launcher_result_sha256"]==sha(root/"launcher_result.json"),
                "official Runner terminal receipt does not match")
        job_ids.append(item["job_id"])
    require(len(set(job_ids))==2,"pair must be two distinct official jobs")
    return terminal

def read_run(root, expected_replicate):
    root=Path(root).resolve()
    evidence={}
    def read(rel):
        p=root/rel;evidence[str(p)]=sha(p)
        return json.loads(p.read_text())
    launch=read("launch.json");result=read("launcher_result.json");complete=read("native_aa_child_complete.json")
    require(result["verdict"]=="NATIVE_N8_COMPLETE_PENDING_PAIR_COMPARISON" and result["terminal"]==8 and
            result["replicate"]==expected_replicate and not result["production_continuation_allowed"],"incomplete launcher")
    require(complete["observed_terminal"]==8 and not complete["guard_used"],"incomplete/guarded child")
    spec=launch["spec"]
    require(original_launcher.spec_digest(spec)==launch["spec_body_sha256"],"launch spec body differs")
    require(spec["replicate"]==expected_replicate and spec["deadline_seconds"]==1200 and
            Path(spec["output_root"]).resolve()==root,"run/spec identity")
    for item in spec["references"].values():
        require(sha(item["path"])==item["sha256"],"changed referenced evidence")
    for path,digest in spec["frozen_files"].items():require(sha(path)==digest,"changed source")
    binding=json.loads(Path(spec["references"]["binding"]["path"]).read_text())
    review=json.loads(Path(spec["references"]["source_review"]["path"]).read_text())
    require(review["schema"]=="matcha_exact_launcher_review_v1" and review["verdict"]=="PASS" and
            review["spec_body_sha256"]==launch["spec_body_sha256"] and
            review["frozen_files"]==spec["frozen_files"] and
            review["approval_scope"]=="native_u1_u2_only_n8_no_guard_no_promotion","independent review differs")
    for path,digest in binding["files"].items():require(sha(path)==digest,"changed original input/source")
    before=read("runtime_before.json");after=read("runtime_after.json")
    validate_run_headers(spec,result,complete,before,after,expected_replicate)
    for proof in (before,after):
        ctx=proof["context"]
        require(ctx["actual_driverapi_context_uuid"]==spec["uuid"] and ctx["visible_count"]==1 and
                ctx["logical_gpu"]==0 and not proof["diagnostic_guard_imported"] and
                not proof["autocast_enabled"] and not proof["cpu_autocast_enabled"],"invalid runtime control")
    trace_path=root/"state/trace_from_1.jsonl"; evidence[str(trace_path)]=sha(trace_path)
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    require([(r["iteration"],"exact_digest" in r) for r in trace]==
            [(k,False) for k in range(1,9)]+[(8,True)],"missing/extra/reordered trace")
    projection_path=Path("/root/STDLoc/experiments/matcha_exact_resume_v2/project_a_reference.py")
    require(sha(projection_path)=="644fc452589babcf90a4bbc579ebd30a062d3c89912766e5ad28c72b589f147c","trace schema validator changed")
    ms=importlib.util.spec_from_file_location("trace_schema",projection_path)
    module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    module.validate_projected_records({module.validate_record(r):r for r in trace})
    endpoint_info=read("state/step_00008/manifest.json");static_info=read("state/static/manifest.json")
    require((root/"state/static/state.pt").is_file() and
            (root/"state/static/state.pt").stat().st_size==static_info["bytes"],"static payload absent/truncated")
    state_path=root/"state/step_00008/state.pt"
    require(state_path.stat().st_size==endpoint_info["bytes"] and sha(state_path)==endpoint_info["sha256"],"corrupt endpoint payload")
    state=torch.load(state_path,map_location="cpu",weights_only=False)
    require(state["schema"]==er.SCHEMA and state["completed_iteration"]==8 and state["next_iteration"]==9 and
            state["binding"]["manifest"]==binding and state["phase"]==er.PHASE,"endpoint binding/boundary")
    require(state["binding"]["runtime"]==after["runtime"],"endpoint and observed native runtime differ")
    require(state["static_ref"]["sha256"]==static_info["sha256"] and
            state["static_ref"]["bytes"]==static_info["bytes"] and
            Path(state["static_ref"]["path"]).resolve()==root/"state/static","static reference mismatch")
    comparison={k:state[k] for k in er.COMPARISON_FIELDS}
    digest=er.semantic_digest(comparison)
    require(digest==trace[-1]["exact_digest"],"endpoint raw content differs from producer full digest")
    evidence[str(state_path)]=endpoint_info["sha256"]
    return {"root":str(root),"evidence":evidence,"before":before,"after":after,"trace":trace,
            "state":state,"comparison":comparison,"digest":digest,"static":static_info,"spec":spec}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--u1",required=True,type=Path);parser.add_argument("--u2",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--terminal-receipt",required=True,type=Path)
    args=parser.parse_args()
    require(not args.output.exists(),"refuse report overwrite")
    require(not torch.cuda.is_initialized() and torch.cuda.device_count()==0,"CPU-only process required")
    started=time.monotonic()
    terminal=validate_terminal_receipt(args.terminal_receipt,{"u1":args.u1,"u2":args.u2})
    a=read_run(args.u1,"u1");b=read_run(args.u2,"u2")
    require(a["spec"]["references"]["binding"]==b["spec"]["references"]["binding"],"pair binding differs")
    runtime=[]
    for phase in ("before","after"):
        runtime+=differences(normalized_runtime(a[phase]),normalized_runtime(b[phase]),"$runtime."+phase)
    runtime+=differences(a["state"]["binding"]["runtime"],b["state"]["binding"]["runtime"],"$endpoint_runtime")
    light=differences(a["trace"][:-1],b["trace"][:-1],"$light")
    full=differences(a["comparison"],b["comparison"],"$endpoint")
    require(not (a["digest"]!=b["digest"] and not full),"digest differs but inventory is empty")
    static_equal=a["static"]==b["static"] and a["state"]["static_digest"]==b["state"]["static_digest"]
    verdict="INVALID_RUNTIME_CONTROL" if runtime else ("PASS_EXACT_NATIVE_PAIR" if not light and not full and static_equal else "FAIL_EXACT_NATIVE_PAIR")
    report={"schema":"matcha_native_aa_pair_difference_v1","verdict":verdict,"tolerance":0,
        "production_continuation_allowed":False,"light_differences":light,"runtime_differences":runtime,
        "official_runner_receipt":{"path":str(args.terminal_receipt),"sha256":sha(args.terminal_receipt)},
        "scalar_loss_diagnostics":scalar_loss_diagnostics(a["trace"],b["trace"]),
        "full_endpoint_differences":full,"endpoint_digests":[a["digest"],b["digest"]],
        "static_producer_archives_and_endpoint_digest_equal":static_equal,
        "static_payload_reread":False,"static_deep_localization_required":not static_equal,
        "inputs":[{"root":x["root"],"sha256":x["evidence"]} for x in (a,b)],
        "seconds":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "cuda_initialized":torch.cuda.is_initialized(),"comparator_sha256":sha(__file__),
        "limits":"No floating tolerance, no automatic retry/continuation; all mismatched paths enumerated with bounded examples."}
    with args.output.open("x") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({"path":str(args.output),"sha256":sha(args.output),"verdict":verdict,
        "light_difference_count":len(light),"full_difference_count":len(full),"runtime_difference_count":len(runtime),
        "seconds":report["seconds"],"peak_rss_bytes":report["peak_rss_bytes"]}))
if __name__=="__main__":main()

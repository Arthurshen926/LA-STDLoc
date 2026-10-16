"""Standalone semantic-on Cambridge Full/Compact benchmark; no scientific kernel edits."""
from __future__ import annotations
import argparse, os, platform, resource, subprocess
import threading, time
from pathlib import Path
import numpy as np
import torch
from anygsloc.artifacts import validate, sha256
from anygsloc.data.datasets import ColmapDataset
from anygsloc.evaluation.metrics import pose_error
from anygsloc.evaluation.runner import summarize, json_safe
from anygsloc.online.runtime import Localizer
from experiments import cambridge_full_repaired_baseline as F
S=F.S; A=F.A; M=F.M
BASE=Path("/mnt/pool/sqy/anygsloc_full_repaired_baseline_20261011_v1")
CONFIGS=("full","compact")
TIMINGS=("runtime_ms","frame_ms","rgb_decode_preprocess_ms","mask_compose_ms")
DESIGN={
 "schema":"cambridge_semantic_on_deployment_benchmark_v1",
 "scenes":list(F.SCENES),"configurations":list(CONFIGS),"warmup_per_round":50,"rounds":3,
 "query_semantics":True,"keypoint_count":2048,"seed":2026,"cpu_threads":4,
 "frontend":"pureGS unmasked; repaired StMary v2; Full and native Compact from same full pools",
 "mask_warning":"PRECOMPUTED MASKS. Semantic network inference is NOT included.",
 "mask_io":"masks.pkl disk load belongs to initialization; per-frame in-memory channel conversion/resize/AND is mask_compose_ms",
 "core":"unchanged Localizer.runtime_ms: image/mask H2D, mask application, SuperPoint, global matching, host transfer, PnP",
 "frame":"RGB decode/preprocess + CPU mask composition + complete localize; synchronized; excludes metrics, serialization and initialization",
 "features":"SuperPoint executes for every warmup and measured frame; no feature cache",
 "storage":"actual deployment_map.pt bytes only; shared feature-network weights and offline Gaussian/cache assets excluded",
 "isolation":"single physical GPU UUID, serial cells; external compute process checks every second; host CPU/disk conditions recorded",
 "aggregation":"each scene uses all three rounds; pooled query distribution and five-scene-equal mean reported separately",
}
def write(path,value): A.write_new(Path(path),json_safe(value))
def hashes():
    out=F.hashes()
    for rel in ("experiments/cambridge_deployment_benchmark.py","tests/test_cambridge_deployment_benchmark.py"):
        out[rel]=sha256(A.ROOT/rel)
    return out
def check_ref(ref):
    if sha256(ref["path"])!=ref["sha256"] or Path(ref["path"]).stat().st_size!=ref["bytes"]:
        raise ValueError("frozen artifact changed: "+ref["path"])
def check_release(release,protocol_sha,now=None):
    now=time.time() if now is None else now
    if (release.get("approved") is not True or release.get("scope")!="cambridge_full_compact_formal_timing"
        or release.get("protocol_sha256")!=protocol_sha or not str(release.get("gpu_uuid","")).startswith("GPU-")
        or not release.get("approval_reference")
        or not release.get("valid_from_unix",float("inf"))<=now<=release.get("valid_until_unix",-1)):
        raise ValueError("missing, expired or mismatched parent GPU release")
    if release["valid_until_unix"]-release["valid_from_unix"]>7200:
        raise ValueError("GPU release window exceeds two hours")
    return release["gpu_uuid"]
def verify_subset(full,compact,rows):
    rows=np.asarray(rows)
    if rows.ndim!=1 or rows.dtype.kind not in "iu" or len(rows)!=len(compact["anchor_xyz"]):
        raise ValueError("Compact source rows malformed")
    if len(np.unique(rows))!=len(rows) or (rows<0).any() or (rows>=len(full["anchor_xyz"])).any():
        raise ValueError("Compact source rows invalid")
    for key in ("anchor_xyz","anchor_features"):
        if not torch.equal(full[key][rows],compact[key]): raise ValueError("Compact not exact Full subset: "+key)

def prepare(root):
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    baseline=F.protocol(BASE)
    review=A.read(BASE/"independent_final_review.json")
    if (review.get("complete") is not True or review.get("verdict")!="PASS" or
        review.get("protocol_sha256")!=sha256(BASE/"protocol.json") or
        review.get("report_sha256")!=sha256(BASE/"report.json")):
        raise ValueError("Full baseline independent review incomplete")
    scenes={}; refs=[S.evidence(BASE/x) for x in ("protocol.json","report.json","independent_final_review.json")]
    for scene in F.SCENES:
        repaired=scene==S.SCENE
        sr=F.REPAIRED if repaired else A.MATRIX
        scene_info=A.read(sr/"protocol.json")["scenes"][scene]
        full=baseline["full"]["anygsloc"] if repaired else baseline["reused"][scene]["anygsloc"]["map"]
        compact_dir=F.REPAIRED/"maps/new_native" if repaired else S.A_ROOT/"maps"/scene/"new_native"
        selection=compact_dir/"selection.json"; compact=A.read(selection)
        if Path(compact["path"]).resolve()!=(compact_dir/"deployment_map.pt").resolve():
            raise ValueError("Compact selection path detached")
        states={}
        for cfg,ref in (("full",full),("compact",compact)):
            check_ref(ref); state=validate(A.load(ref["path"]))
            if len(state["anchor_xyz"])!=ref["count"]: raise ValueError("map point count differs")
            if state["provenance"].get("uses_test_queries") is not False:
                raise ValueError("map test-query consumption contract absent")
            states[cfg]=state
        rows_path=compact_dir/"deployment_map.source_rows.npy"
        verify_subset(states["full"],states["compact"],np.load(rows_path,allow_pickle=False))
        native_root=F.REPAIRED if repaired else S.A_ROOT
        native_folder=native_root/"evaluations"/"new_native" if repaired else native_root/"evaluations"/scene/"new_native"
        att=A.verify_evaluation(native_folder,scene,"new_native",A.read(native_root/"protocol.json"),
             sha256(native_root/"protocol.json"),sha256(native_root/"query_inputs"/(scene+".json")),compact)
        if repaired:
            if states["compact"]["provenance"].get("source_full_sha256")!=full["sha256"]:
                raise ValueError("repaired Compact source Full hash differs")
        else:
            audit=native_root/"validation"/(scene+".json")
            if A.read(audit)["source_hashes"]["new_full"]["sha256"]!=full["sha256"]:
                raise ValueError("Compact adapter source Full hash differs")
            refs.append(S.evidence(audit))
        refs.extend([S.evidence(rows_path),S.evidence(native_folder/"attestation.json"),
                     S.evidence(native_root/"protocol.json"),*att["files"].values()])
        del states,state
        expected=baseline["reference_native_same_frontend"]["per_scene"][scene]
        if compact["count"]!=expected["map_landmarks"] or compact["bytes"]!=expected["map_bytes"]:
            raise ValueError("Compact differs from accepted five-scene reference")
        snap=BASE/"query_inputs"/(scene+".json") if repaired else A.MATRIX/"query_inputs"/(scene+".json")
        native_snap=(F.REPAIRED if repaired else S.A_ROOT)/"query_inputs"/(scene+".json")
        F.same_snapshot(A.read(snap),A.read(native_snap))
        refs.extend([S.evidence(selection),S.evidence(snap),S.evidence(native_snap)])
        scenes[scene]={"scene_info":scene_info,"maps":{"full":full,"compact":compact},
                       "reference_snapshot":S.evidence(snap),"calibration":A.read(snap)["calibration"]}
        if scenes[scene]["calibration"]["keypoint_count"]!=2048: raise ValueError("unexpected keypoint budget")
    write(root/"protocol.json",{"design":DESIGN,"code_sha256":hashes(),"scenes":scenes,"evidence":refs})
    for scene,item in scenes.items():
        data=ColmapDataset(item["scene_info"]["dataset"],load_masks=False)
        M.evaluation_snapshot(root,scene,data,item["scene_info"])
        F.same_snapshot(A.read(root/"query_inputs"/(scene+".json")),A.read(item["reference_snapshot"]["path"]))
    write(root/"preflight.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
         "query_snapshots":{s:sha256(root/"query_inputs"/(s+".json")) for s in F.SCENES},
         "cuda_initialized":torch.cuda.is_initialized(),"gpu_run_performed":False})
def protocol(root):
    root=Path(root); p=A.read(root/"protocol.json"); ready=A.read(root/"preflight.json")
    if p["design"]!=DESIGN or p["code_sha256"]!=hashes(): raise ValueError("benchmark source/design changed")
    if not ready["complete"] or ready["cuda_initialized"] or ready["protocol_sha256"]!=sha256(root/"protocol.json"):
        raise ValueError("CPU preflight invalid")
    for ref in p["evidence"]: check_ref(ref)
    for s,h in ready["query_snapshots"].items():
        if sha256(root/"query_inputs"/(s+".json"))!=h: raise ValueError("query snapshot changed")
    return p
def stats(rows):
    if not rows: raise ValueError("empty timing sample")
    out={}
    for key in TIMINGS:
        v=np.asarray([r[key] for r in rows],dtype=np.float64)
        if not np.isfinite(v).all() or (v<0).any(): raise ValueError("invalid timing")
        out[key]={"mean":float(v.mean()),"p50":float(np.percentile(v,50)),"p90":float(np.percentile(v,90)),
                  "p95":float(np.percentile(v,95)),"count":len(v)}
    return out
def benchmark_round(data,cameras,engine,*,warmup=50,clock=time.perf_counter,sync=lambda:None,guard=lambda:None,after_warmup=lambda:None):
    """CPU-testable orchestration. Scientific operations are only engine.localize."""
    if not cameras: raise ValueError("empty test split")
    warm_names=[]
    for i in range(warmup):
        guard(); cam=cameras[i%len(cameras)]; mask=data.valid_mask(cam)
        if mask is None: raise ValueError("semantic-on mask missing")
        engine.localize(data.load_image(cam),fov_x=cam.fov_x,fov_y=cam.fov_y,valid_mask=mask)
        warm_names.append(cam.image_name)
    sync(); after_warmup(); rows=[]; loop_start=clock()
    for cam in cameras:
        guard(); sync(); start=clock()
        image=data.load_image(cam); after_rgb=clock()
        mask=data.valid_mask(cam); after_mask=clock()
        if mask is None: raise ValueError("semantic-on mask missing")
        result=engine.localize(image,fov_x=cam.fov_x,fov_y=cam.fov_y,valid_mask=mask)
        sync(); end=clock()
        re,te=pose_error(result["pose"].pose_w2c,cam.pose_w2c) if result["success"] else (None,None)
        rows.append({"image_name":cam.image_name,"success":result["success"],"rotation_error_deg":re,
           "translation_error_cm":te,"pose_w2c":result["pose"].pose_w2c.tolist(),
           "runtime_ms":result["runtime_ms"],"frame_ms":(end-start)*1000,
           "rgb_decode_preprocess_ms":(after_rgb-start)*1000,"mask_compose_ms":(after_mask-after_rgb)*1000})
    guard()
    return {"warmup_names":warm_names,"queries":rows,"timing":stats(rows),
            "accuracy":summarize(rows,clock()-loop_start)}
def smi(query,kind):
    return subprocess.check_output(["nvidia-smi","--query-"+kind+"="+query,"--format=csv,noheader,nounits"],
                                   text=True,timeout=10).strip()
def hardware():
    gpu=smi("index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,driver_version","gpu")
    apps=smi("gpu_uuid,pid,process_name,used_memory","compute-apps")
    return {"unix":time.time(),"gpu_csv":gpu,"compute_apps_csv":apps,"loadavg":list(os.getloadavg()),
       "meminfo":Path("/proc/meminfo").read_text(),"diskstats":Path("/proc/diskstats").read_text(),
       "self_io":Path("/proc/self/io").read_text(),"cpu_affinity":sorted(os.sched_getaffinity(0))}
def target_pids(sample,uuid):
    return {int(row.split(",")[1].strip()) for row in sample["compute_apps_csv"].splitlines()
            if row.split(",")[0].strip()==uuid}
class Monitor:
    def __init__(self,uuid,owned_pids):
        self.uuid=uuid; self.owned=owned_pids; self.samples=[]; self.error=None
        self.stop=threading.Event(); self.thread=threading.Thread(target=self.work,daemon=True)
    def work(self):
        while not self.stop.is_set():
            try:
                s=hardware();self.samples.append(s)
                if target_pids(s,self.uuid)-self.owned: self.error="external compute process observed"
            except Exception as exc: self.error=repr(exc)
            self.stop.wait(1)
    def check(self):
        if self.error: raise RuntimeError("benchmark isolation lost: "+self.error)
    def __enter__(self): self.thread.start();return self
    def __exit__(self,*args): self.stop.set();self.thread.join(timeout=15)
def evaluate(root,scene,config,release_path):
    root=Path(root);p=protocol(root);rel=A.read(release_path)
    protocol_sha=sha256(root/"protocol.json");uuid=check_release(rel,protocol_sha)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=uuid:
        raise ValueError("launch with CUDA_VISIBLE_DEVICES set to the released single GPU UUID")
    if scene not in p["scenes"] or config not in CONFIGS: raise ValueError("unknown benchmark cell")
    # A shared root lock prevents simultaneous cells even on different visible GPUs.
    lock=root/"benchmark.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    out=root/"runs"/scene/config; monitor=None;created=False
    try:
        out.mkdir(parents=True,exist_ok=False);created=True
        before=hardware()
        if target_pids(before,uuid): raise RuntimeError("released GPU already has compute processes")
        if not any(row.split(",")[1].strip()==uuid for row in before["gpu_csv"].splitlines()):
            raise ValueError("released GPU UUID absent")
        item=p["scenes"][scene];ref=item["maps"][config];check_ref(ref)
        s=item["scene_info"];snap=sha256(root/"query_inputs"/(scene+".json"))
        M.evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)
        started=time.perf_counter(); data=ColmapDataset(s["dataset"],load_masks=True)
        dataset_seconds=time.perf_counter()-started
        cameras=data.split("test")
        if [c.image_name for c in cameras]!=s["test_names"]: raise ValueError("query order changed")
        cal=item["calibration"]
        engine=Localizer(ref["path"],keypoint_count=2048,reprojection_error_px=cal["reprojection_error_px"],
                         mask_policy="explicit_valid_mask",device="cuda:0",seed=2026)
        torch.cuda.synchronize();init_seconds=time.perf_counter()-started
        expected_config=A.read((BASE/"evaluations/anygsloc/manifest.json") if scene==S.SCENE else
                        (A.MATRIX/"evaluations"/scene/"pure_gs/anygsloc/full/manifest.json"))["config"]
        for k,v in expected_config.items():
            if k in engine.config and engine.config[k]!=v: raise ValueError("Localizer config differs: "+k)
        after=hardware();owned=target_pids(after,uuid)
        if len(owned)!=1: raise RuntimeError("cannot uniquely identify sole initialized GPU process")
        versions={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
                  "cudnn":torch.backends.cudnn.version(),"platform":platform.platform(),
                  "torch_threads":torch.get_num_threads(),"visible_gpu":os.environ["CUDA_VISIBLE_DEVICES"],
                  "namespace_pid":os.getpid(),"nvidia_smi_owned_pid":sorted(owned)}
        import cv2,poselib
        from anygsloc.features.superpoint import SUPERPOINT_WEIGHT_SHA256
        versions.update(cpu_model=next((v.split(":",1)[1].strip() for v in Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name")),"unknown"),
                        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                        opencv=cv2.__version__,poselib=getattr(poselib,"__version__","unavailable"),
                        poselib_module_sha256=sha256(poselib.__file__),superpoint_weights_sha256=SUPERPOINT_WEIGHT_SHA256)
        initialization_peak={"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()}
        write(out/"release_used.json",rel)
        write(out/"initialization.json",{"protocol_sha256":protocol_sha,"scene":scene,"configuration":config,
              "query_inputs_sha256":snap,"map":ref,"dataset_including_mask_archive_seconds":dataset_seconds,
              "model_map_device_seconds":init_seconds-dataset_seconds,"total_seconds":init_seconds,
              "before":before,"after":after,"versions":versions,"release":S.evidence(out/"release_used.json"),
              "cuda_peak_initialization":initialization_peak,
              "warning":DESIGN["mask_warning"]})
        rounds=[]
        monitor=Monitor(uuid,owned)
        with monitor:
            for repeat in range(3):
                torch.cuda.reset_peak_memory_stats();warmup_peak={}
                def guard():
                    monitor.check();check_release(rel,protocol_sha)
                def after_warmup():
                    warmup_peak.update(allocated_bytes=torch.cuda.max_memory_allocated(),
                                       reserved_bytes=torch.cuda.max_memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                result=benchmark_round(data,cameras,engine,warmup=50,sync=lambda:torch.cuda.synchronize(),
                                       guard=guard,after_warmup=after_warmup)
                result.update(round=repeat,warmup_cuda_peak=warmup_peak,peak_torch_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_torch_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                write(out/("round"+str(repeat)+".json"),result);rounds.append(result)
        monitor.check();last=hardware();monitor.samples.append(last)
        if target_pids(last,uuid)-owned:raise RuntimeError("external process at benchmark end")
        check_ref(ref)
        if M.evaluation_snapshot(root,scene,ColmapDataset(s["dataset"],load_masks=False),s)!=snap:
            raise ValueError("query inputs changed")
        protocol(root)
        write(out/"telemetry.json",monitor.samples)
        write(out/"summary.json",{"complete":True,"protocol_sha256":sha256(root/"protocol.json"),
              "query_inputs_sha256":snap,"map":ref,"scene":scene,"configuration":config,
              "timing":stats([q for r in rounds for q in r["queries"]]),"sample_count":sum(len(r["queries"]) for r in rounds),
              "round_files":{str(i):S.evidence(out/("round"+str(i)+".json")) for i in range(3)},
              "gpu_uuid":uuid,"isolation":"no external process observed in sampled intervals; host contention telemetry retained",
              "mask_warning":DESIGN["mask_warning"],"telemetry":S.evidence(out/"telemetry.json"),
              "initialization":S.evidence(out/"initialization.json"),"release":S.evidence(out/"release_used.json")})
    except Exception as exc:
        if created and not (out/"failure.json").exists():
            write(out/"failure.json",{"complete":False,"error":repr(exc),"telemetry":monitor.samples if monitor else []})
        raise
    finally: lock.unlink()
def checked_cell(root,scene,config,p):
    root=Path(root);out=root/"runs"/scene/config;s=A.read(out/"summary.json")
    expected_sha=sha256(root/"protocol.json");snap=sha256(root/"query_inputs"/(scene+".json"))
    if (s.get("complete") is not True or s.get("protocol_sha256")!=expected_sha or
        s.get("scene")!=scene or s.get("configuration")!=config or s.get("query_inputs_sha256")!=snap or
        s.get("map")!=p["scenes"][scene]["maps"][config] or (out/"failure.json").exists()):
        raise ValueError("benchmark cell identity incomplete or mismatched")
    for key,name in (("initialization","initialization.json"),("telemetry","telemetry.json"),("release","release_used.json")):
        if Path(s[key]["path"]).resolve()!=(out/name).resolve():raise ValueError("cell artifact path differs")
        check_ref(s[key])
    init=A.read(out/"initialization.json");rel=A.read(out/"release_used.json")
    for key in ("protocol_sha256","scene","configuration","query_inputs_sha256","map"):
        if init[key]!=s[key]:raise ValueError("initialization cell identity differs")
    if init["release"]!=s["release"]:raise ValueError("initialization release differs")
    if check_release(rel,expected_sha,init["before"]["unix"])!=s["gpu_uuid"]:
        raise ValueError("physical GPU release differs")
    if set(s["round_files"])!={"0","1","2"}:raise ValueError("round set differs")
    rows=[];names=p["scenes"][scene]["scene_info"]["test_names"]
    for i in range(3):
        ref=s["round_files"][str(i)];path=out/("round"+str(i)+".json")
        if Path(ref["path"]).resolve()!=path.resolve():raise ValueError("round artifact path differs")
        check_ref(ref);r=A.read(path)
        if r.get("round")!=i or [q["image_name"] for q in r["queries"]]!=names or r["warmup_names"]!=[names[j%len(names)] for j in range(50)]:
            raise ValueError("round query/warmup identities differ")
        rows.extend(r["queries"])
    if s["sample_count"]!=len(rows) or s["timing"]!=stats(rows):raise ValueError("stored timing aggregation differs")
    return s,rows

def report(root):
    root=Path(root);p=protocol(root);results={}
    for config in CONFIGS:
        per={};all_rows=[];uuids=set()
        for scene in F.SCENES:
            s,rows=checked_cell(root,scene,config,p)
            per[scene]={"timing":stats(rows),"map":s["map"]};all_rows.extend(rows);uuids.add(s["gpu_uuid"])
        results[config]={"per_scene":per,"pooled":stats(all_rows),
            "scene_equal_mean_ms":{k:float(np.mean([v["timing"][k]["mean"] for v in per.values()])) for k in TIMINGS},
            "total_map_bytes":sum(v["map"]["bytes"] for v in per.values()),
            "mean_map_bytes":float(np.mean([v["map"]["bytes"] for v in per.values()])),"gpu_uuids":sorted(uuids)}
    if len(set(u for x in results.values() for u in x["gpu_uuids"]))!=1:
        raise ValueError("Full and Compact must use same physical GPU")
    write(root/"report.json",{"complete":True,"design":DESIGN,"results":results})
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("stage",choices=("prepare","evaluate","report"))
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--scene",choices=F.SCENES)
    ap.add_argument("--configuration",choices=CONFIGS);ap.add_argument("--release",type=Path)
    args=ap.parse_args();torch.set_num_threads(4)
    if args.stage=="prepare":prepare(args.root)
    elif args.stage=="evaluate":evaluate(args.root,args.scene,args.configuration,args.release)
    else:report(args.root)
if __name__=="__main__":main()

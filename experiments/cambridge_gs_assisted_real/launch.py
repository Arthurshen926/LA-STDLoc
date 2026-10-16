"""One owned worker with an enforced 3000s deadline, stage lock and terminal receipt."""
import contextlib,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from .artifacts import identity,stage_path,stage_name,check_identity
REPO=Path(__file__).resolve().parents[2]
DEFAULT_ROOT=Path("/mnt/pool/sqy/anygsloc_gs_assisted_real_20261012_v1")
MIRROR=REPO/"outputs/anygsloc_gs_assisted_real_20261012_v1"
DEADLINE_SECONDS=3000
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reference(p):
    p=Path(p).resolve();s=p.stat()
    return dict(path=str(p),sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def create(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
def argument(name,default=None):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
def gate(root,stage):
    p=read(root/"protocol.json");r=read(root/"independent_review.json")
    approved=r.get("cpu_approved") if stage=="audit" else r.get("approved")
    if approved is not True or r.get("protocol_sha256")!=sha(root/"protocol.json") or r.get("code_sha256")!=p["code_sha256"]:
        raise ValueError("independent scientific review absent")
    for name,value in p["code_sha256"].items():
        if sha(REPO/name)!=value:raise ValueError("frozen source changed: "+name)
    release=read(root/"execution_release.json")
    if release.get("released") is not True or release.get("protocol_sha256")!=sha(root/"protocol.json"):
        raise ValueError("formal timing release absent")
    if stage in ("sample","build","evaluate"):
        from .resource import PHYSICAL_UUID
        if release.get("gpu_uuid")!=PHYSICAL_UUID:raise ValueError("GPU1 lease absent")
    return p

@contextlib.contextmanager
def exclusive_lock(name):
    p=MIRROR/"locks"/(name+".lock");p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield f.fileno()
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def worker(stage,root,scene,selection):
    if stage!="prepare":gate(root,stage)
    context=None
    if stage in ("sample","build","evaluate"):
        from .resource import initialize
        context=initialize()
    elif os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU stage must hide all GPUs")
    create(MIRROR/"launch_context"/(stage_name(scene,stage,selection)+".json"),dict(
        stage=stage,scene=scene,selection=selection,context=context,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}))
    from . import driver
    driver.main()

def supervise(command,log_path,seconds=DEADLINE_SECONDS,lock_fds=()):
    """Critical launch registration; repeated signals cannot interrupt owned-worker drain."""
    child=None;critical=False;pending=None;error=None;started=time.monotonic()
    old={}
    def interrupted(signum,frame):
        nonlocal pending
        if critical:pending=signum
        else:raise InterruptedError("supervisor signal "+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):
        old[sig]=signal.signal(sig,interrupted)
    log=None;returncode=None
    try:
        critical=True
        try:
            log=Path(log_path).open("x")
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(lock_fds))
        finally:
            critical=False
        if pending is not None:raise InterruptedError("pending supervisor signal "+str(pending))
        returncode=child.wait(timeout=max(.1,seconds-(time.monotonic()-started)))
    except BaseException as exc:
        error=repr(exc)
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            # A failed worker must not leave grandchildren that ignore TERM.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if returncode is None:returncode=child.returncode
        if log is not None:log.close()
        for sig,handler in old.items():signal.signal(sig,handler)
    return dict(returncode=returncode,error=error,worker_pid=None if child is None else child.pid,
        owned_worker_waited=child is not None and child.poll() is not None,
        wall_seconds=time.monotonic()-started,deadline_seconds=seconds)

def main():
    child_mode=len(sys.argv)>1 and sys.argv[1]=="--worker"
    if child_mode:sys.argv.pop(1)
    stage=sys.argv[1];root=Path(argument("--root",str(DEFAULT_ROOT))).resolve()
    scene=argument("--scene");selection=argument("--selection")
    allowed=("prepare","audit","sample","build","select","evaluate","report")
    if stage not in allowed:raise ValueError("unknown stage")
    if stage not in ("prepare","report") and scene not in ("ShopFacade","OldHospital","KingsCollege","GreatCourt","StMarysChurch"):
        raise ValueError("stage scene required")
    if stage=="evaluate" and selection not in ("full","native_compact"):raise ValueError("selection required")
    if stage!="evaluate" and selection is not None:raise ValueError("unexpected selection")
    if child_mode:
        worker(stage,root,scene,selection);return
    if stage!="prepare":gate(root,stage)
    name=stage_name(scene,stage,selection);receipt=root/"execution"/(name+".json")
    if receipt.exists():raise FileExistsError(receipt)
    with contextlib.ExitStack() as stack:
        lock_fds=[stack.enter_context(exclusive_lock(name))]
        if stage in ("sample","build","evaluate"):lock_fds.append(stack.enter_context(exclusive_lock("physical_gpu1")))
        log=MIRROR/"logs"/(name+".log");log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-m","experiments.cambridge_gs_assisted_real.launch","--worker",*sys.argv[1:]]
        started=time.time();print("START",name,flush=True)
        result=supervise(command,log,lock_fds=lock_fds)
        ok=result["returncode"]==0 and result["error"] is None
        summary=None
        if ok:
            try:
                path=stage_path(root,scene,stage,selection);value=read(path)
                if stage!="prepare":
                    check_identity(value,root,scene,stage,selection)
                    if value.get("complete") is not True:raise ValueError("child incomplete output")
                summary=reference(path)
            except BaseException as error:
                ok=False;result["error"]=repr(error)
        value=dict(**identity(root,scene,stage,selection),success=ok,summary=summary,
            started_unix=started,ended_unix=time.time(),argv=command,log=reference(log),**result)
        create(receipt,value)
        create(MIRROR/"receipts"/(name+".json"),value)
        if ok and stage!="prepare":create(MIRROR/"stages"/(name+".json"),dict(source=summary,data=read(summary["path"])))
        print("END",name,json.dumps(value),flush=True)
        if not ok:raise SystemExit(1)
if __name__=="__main__":main()


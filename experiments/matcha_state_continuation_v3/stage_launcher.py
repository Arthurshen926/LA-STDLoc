"""Two-stage original30k execution. Exact state transport, no trajectory equivalence claim."""
import argparse, json, os, runpy, signal, shutil, subprocess, sys, time
from pathlib import Path
import native_n8_launcher as base

ROOT = Path(__file__).resolve().parent
RUNS, PYTHON = base.RUNS, base.PYTHON
ASSIGNED = "GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1"
SERIES = "state_continuation_stmary_20261012_v3"
require, sha, write_new = base.require, base.sha, base.write_new
REQUIRED_FILES = base.REQUIRED_FILES + ("stage_launcher.py", "build_v3.py")
BASE_REFS = set(base.REFS) | {"budget"}
STAGE2_REFS = {"prefix_result", "prefix_terminal", "checkpoint_manifest", "static_manifest"}

def validate_spec(spec):
    require(set(spec) == {"schema","stage","physical_gpu","logical_gpu","visible_count","uuid",
        "deadline_seconds","forecast_seconds","output_root","references","frozen_files","environment"},
        "v3 spec fields")
    require(spec["schema"] == "matcha_state_continuation_launch_v3" and spec["stage"] in (1,2),
            "unregistered stage")
    require((spec["physical_gpu"],spec["logical_gpu"],spec["visible_count"],spec["uuid"]) ==
            (0,0,1,ASSIGNED), "unassigned GPU")
    require(spec["deadline_seconds"] == 3500 and 0 < spec["forecast_seconds"] < 3380, "budget")
    require(Path(spec["output_root"]) == RUNS/(SERIES+"_stage"+str(spec["stage"])), "unique fresh output")
    expected = BASE_REFS | (STAGE2_REFS if spec["stage"] == 2 else set())
    require(set(spec["references"]) == expected, "reference coverage")
    for item in spec["references"].values():
        require(set(item) == {"path","sha256"} and Path(item["path"]).is_absolute() and
                len(item["sha256"]) == 64, "reference format")
    require({str(ROOT/name) for name in REQUIRED_FILES} <= set(spec["frozen_files"]), "source coverage")
    require(all(Path(p).parent == ROOT for p in spec["frozen_files"]), "source outside isolated v3")
    return spec

def validate_budget(spec, budget):
    require(budget["schema"] == "matcha_state_continuation_budget_v3" and
            budget["stage"] == spec["stage"] and budget["forecast_seconds"] == spec["forecast_seconds"],
            "budget identity")
    components = budget["components_seconds"]
    require(components and all(isinstance(x,(int,float)) and 0 <= x < 3380 for x in components.values()),
            "invalid budget components")
    require(abs(sum(components.values())-budget["forecast_seconds"]) < 1e-8 and
            budget["forecast_seconds"] < 3380 and budget["hard_deadline_seconds"] == 3500,
            "budget sum or deadline")
    require(budget["prediction_not_guarantee"] is True and budget["no_trajectory_equivalence_claim"] is True,
            "misleading budget claim")
    if spec["stage"] == 1:
        require(components == {"observed_max_setup":530.0,
            "historical_prefix_with_10pct_margin":2169.0*1.10,
            "observed_max_full_capture":179.0, "observed_publication":24.0,
            "source_hash_finite_tail_and_extra_io_reserve":120.0}, "unregistered prefix budget")
    else:
        require(budget["measured_prefix_result_sha256"] == spec["references"]["prefix_result"]["sha256"] and
                budget["measured_checkpoint_manifest_sha256"] == spec["references"]["checkpoint_manifest"]["sha256"] and
                budget["measured_static_manifest_sha256"] == spec["references"]["static_manifest"]["sha256"],
                "stage2 budget lacks actual prefix assets")

def verify_frozen(spec, deadline):
    docs = base.verify_frozen(spec,deadline)
    require(docs["source_review"]["approval_scope"] == "state_exact_continuation_v3_stage"+str(spec["stage"]),
            "wrong source/execution review scope")
    validate_budget(spec,docs["budget"])
    if spec["stage"] == 2:
        receipt, result = docs["prefix_terminal"], docs["prefix_result"]
        require(receipt["schema"] == "matcha_v3_official_prefix_terminal" and
                receipt["status"] == "completed" and receipt["exit_code"] == 0 and
                receipt["job_id"].startswith("wc_job_") and
                receipt["launcher_result_sha256"] == spec["references"]["prefix_result"]["sha256"],
                "missing official successful prefix terminal")
        require(result["verdict"] == "PREFIX14999_COMPLETE" and result["terminal"] == 14999 and
                result["native_horizon"] == 30000 and not result["floating_trajectory_equivalence_claimed"],
                "prefix incomplete")
        prefix = RUNS/(SERIES+"_stage1")
        require(Path(spec["references"]["prefix_result"]["path"]) == prefix/"launcher_result.json" and
                Path(spec["references"]["checkpoint_manifest"]["path"]) == prefix/"state/step_14999/manifest.json" and
                Path(spec["references"]["static_manifest"]["path"]) == prefix/"state/static/manifest.json",
                "wrong source checkpoint")
    return docs

def native_argv(spec, plan, root):
    argv = list(plan["argv"])
    argv[0] = str(ROOT/("train_with_charts_observer.py" if spec["stage"] == 1 else "train_with_charts_exact.py"))
    argv[argv.index("-m")+1] = str(Path(root)/"model")
    argv += ["--exact_manifest",spec["references"]["binding"]["path"],
             "--exact_dir",str(Path(root)/"state")]
    if spec["stage"] == 1: return argv+["--exact_stop","14999","--exact_save","14999"]
    return argv+["--exact_stop","30000","--exact_resume",
                 str(Path(spec["references"]["checkpoint_manifest"]["path"]).parent)]

def runtime_proof(spec, torch):
    import exact_resume as er
    require(Path(er.__file__).resolve() == ROOT/"exact_resume.py", "wrong continuation helper")
    runtime = er.runtime_binding()
    expected={"threads":32,"interop":16,"deterministic":False,"warn_only":False,
      "cudnn_benchmark":False,"cudnn_deterministic":False,"cudnn_tf32":True,"matmul_tf32":False}
    require(all(runtime.get(k)==v for k,v in expected.items()), "native runtime flags changed")
    require(not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled(), "unexpected AMP")
    return {"context":base.context_proof(spec,torch),"runtime":runtime,
            "autocast_enabled":False,"cpu_autocast_enabled":False}

def validate_trace(records,stage):
    first,last=(1,14999) if stage==1 else (15000,30000)
    expected=[(k,False) for k in range(first,last+1)]
    if stage==1: expected.append((14999,True))
    require([(r["iteration"],"exact_digest" in r) for r in records] == expected,
            "missing/extra/duplicate/out-of-order trace")
    return first,last

def resource_preflight():
    values=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
    memory=int(values["MemAvailable"].split()[0])*1024
    disk=shutil.disk_usage(RUNS).free
    require(memory>=96*1024**3 and disk>=64*1024**3,"insufficient96GiB RAM/64GiB disk")
    return {"available_host_ram_bytes":memory,"free_disk_bytes":disk,
            "minimum_host_ram_bytes":96*1024**3,"minimum_disk_bytes":64*1024**3}

def child(spec_path,run_root):
    signal.pthread_sigmask(signal.SIG_UNBLOCK,base.TERMINATION_SIGNALS)
    spec=validate_spec(json.loads(Path(spec_path).read_text())); root=Path(run_root)
    require(root.resolve()==Path(spec["output_root"]).resolve(),"child output binding")
    owner=json.loads((root/"launch.json").read_text())
    require(owner["spec_sha256"]==sha(spec_path) and owner["parent_pid"]==os.getppid(),"unbound child")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==ASSIGNED,"child visibility")
    fd=int(os.environ.get("MATCHA_LEASE_FD","-1")); require(fd>=0,"inherited lease missing")
    actual,expected=os.fstat(fd),(RUNS/("."+ASSIGNED+".lease")).stat()
    require((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),"lease identity")
    docs=verify_frozen(spec,owner["started_monotonic"]+3380)
    import torch
    torch.cuda.set_device(0); implicit=torch.zeros(1).cuda()
    require(implicit.device==torch.device("cuda:0"),"implicit CUDA assignment"); del implicit
    write_new(root/"runtime_before.json",runtime_proof(spec,torch))
    sys.argv=native_argv(spec,docs["training_plan"],root); started=time.monotonic()
    try:
        runpy.run_path(sys.argv[0],run_name="__main__")
    finally:
        write_new(root/"runtime_after.json",runtime_proof(spec,torch))
    first,terminal=(1,14999) if spec["stage"]==1 else (15000,30000)
    records=[json.loads(line) for line in (root/("state/trace_from_%d.jsonl"%first)).read_text().splitlines()]
    validate_trace(records,spec["stage"])
    perf=[json.loads(line) for line in (root/"state/performance.jsonl").read_text().splitlines()]
    require(perf[-1]["event"]=="stop" and perf[-1]["iteration"]==terminal,"truncated terminal")
    if spec["stage"]==1:
        require((root/"state/step_14999/manifest.json").is_file(),"missing checkpoint")
    else:
        proof=json.loads((root/"state/restore_bitwise_PASS.json").read_text())
        require(proof["verdict"]=="PASS" and proof["all_state_bitwise"] is True and
                proof["saved_full_digest"]==proof["restored_full_digest"] and proof["next_iteration"]==15000,
                "missing full exact restore")
        require((root/"model/point_cloud/iteration_30000/point_cloud.ply").is_file(),"native30000 model missing")
    verify_frozen(spec,owner["started_monotonic"]+3380)
    write_new(root/"stage_child_complete.json",{"schema":"matcha_state_child_complete_v3",
      "stage":spec["stage"],"native_horizon":30000,"first_iteration":first,"observed_terminal":terminal,
      "training_seconds":time.monotonic()-started,"floating_trajectory_equivalence_claimed":False,
      "state_restore_bitwise_passed":spec["stage"]==2})

def launch(spec_path):
    started=time.monotonic(); spec_path=Path(spec_path).resolve()
    spec=validate_spec(json.loads(spec_path.read_text())); deadline=started+3500
    base.check_interruption(); verify_frozen(spec,deadline-120)
    with base.gpu_lease(ASSIGNED) as lease:
        base.verify_assignment(spec); base.check_interruption()
        root=base.claim_output(spec["output_root"])
        write_new(root/"resource_preflight.json",resource_preflight())
        write_new(root/"launch.json",{"schema":"matcha_state_launch_v3","spec_sha256":sha(spec_path),
          "spec_body_sha256":base.spec_digest(spec),"parent_pid":os.getpid(),"spec":spec,
          "started_monotonic":started,"deadline_seconds":3500,"cleanup_reserve_seconds":120})
        env=os.environ.copy(); env.update(spec["environment"])
        env.update(CUDA_VISIBLE_DEVICES=ASSIGNED,ANYGSLOC_GPU_PHYSICAL="0",MATCHA_LEASE_FD=str(lease.fileno()))
        command=[PYTHON,str(Path(__file__).resolve()),"--child",str(spec_path),"--run-root",str(root)]
        process,verdict,terminal=None,"FAILED",None
        try:
            with (root/"stdout.log").open("x") as out,(root/"stderr.log").open("x") as err:
                base.check_interruption(); require(time.monotonic()<deadline-120,"preflight deadline")
                with base.blocked_termination_signals():
                    process=subprocess.Popen(command,cwd="/root/MAtCha",env=env,stdout=out,stderr=err,
                        start_new_session=True,pass_fds=(lease.fileno(),))
                code=base.wait_owned(process,deadline-120); require(code==0,"v3 child failed")
            base.check_interruption()
            complete=json.loads((root/"stage_child_complete.json").read_text()); terminal=complete["observed_terminal"]
            require(complete["stage"]==spec["stage"] and terminal==(14999 if spec["stage"]==1 else 30000),
                    "completion identity")
            for name in ("runtime_before.json","runtime_after.json"):
                proof=json.loads((root/name).read_text())["context"]
                base.validate_context(spec,proof["actual_driverapi_context_uuid"],proof["visible_count"],proof["logical_gpu"])
            verdict="PREFIX14999_COMPLETE" if spec["stage"]==1 else "ORIGINAL30000_COMPLETE_STATE_RESTORED"
        finally:
            with base.blocked_termination_signals():
                if process is not None: base.cleanup_owned_group(process)
                write_new(root/"launcher_result.json",{"schema":"matcha_state_result_v3","verdict":verdict,
                  "stage":spec["stage"],"elapsed_seconds":time.monotonic()-started,"native_horizon":30000,
                  "terminal":terminal,"floating_trajectory_equivalence_claimed":False,
                  "state_restore_bitwise_passed":verdict=="ORIGINAL30000_COMPLETE_STATE_RESTORED"})
        base.check_interruption(); require(time.monotonic()<deadline,"3500s total deadline")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",type=Path); parser.add_argument("--child",type=Path); parser.add_argument("--run-root",type=Path)
    args=parser.parse_args()
    if args.child:
        require(args.spec is None and args.run_root is not None,"invalid child"); child(args.child,args.run_root)
    else:
        require(args.spec is not None and args.run_root is None,"missing spec")
        for sig in base.TERMINATION_SIGNALS: signal.signal(sig,base.interruption)
        signal.setitimer(signal.ITIMER_REAL,3430)
        try: launch(args.spec)
        finally: signal.setitimer(signal.ITIMER_REAL,0)

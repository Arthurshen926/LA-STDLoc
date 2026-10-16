"""Finite serial GPU1 campaign. Existing artifacts are never overwritten."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from .driver import protocol, require_review, SCENES, write_new, ref

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    a=ap.parse_args();root=a.root;protocol(root);require_review(root)
    commands=[("rebind_stmary",["rebind"]),("extract_stmary_hybrid",["extract"]),
              ("build_stmary_raw",["build","--mode","raw_only"]),
              ("build_stmary_hybrid",["build","--mode","hybrid"])]
    for scene in SCENES:
        for mode in ("raw_only","hybrid"):
            commands.append((scene+"_"+mode+"_compact",["select","--scene",scene,"--mode",mode]))
            commands.append((scene+"_"+mode+"_eval_compact",["evaluate","--scene",scene,"--mode",mode,"--selection","native_compact"]))
            if scene=="StMarysChurch":
                commands.append((scene+"_"+mode+"_eval_full",["evaluate","--scene",scene,"--mode",mode,"--selection","full"]))
    commands.append(("report",["report"]))
    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    logdir=root/"campaign";logdir.mkdir(exist_ok=False)
    for name,args in commands:
        command=[sys.executable,"-m","experiments.cambridge_mainline_inputs.driver",*args,"--root",str(root),"--device","cuda:1"]
        started=time.time();print("START",name,flush=True)
        with (logdir/(name+".log")).open("x") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=5400)
        write_new(logdir/(name+".json"),{"name":name,"argv":command,"exit_code":result.returncode,
            "started_unix":started,"ended_unix":time.time(),"log":ref(logdir/(name+".log"))})
        print("END",name,"exit",result.returncode,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    write_new(root/"campaign_complete.json",{"complete":True,"commands":len(commands),"report":ref(root/"report.json")})
if __name__=="__main__":main()

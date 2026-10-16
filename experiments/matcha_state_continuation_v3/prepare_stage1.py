"""Create small binding/spec artifacts only; no input arrays, torch or GPU."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
R=Path("/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1")
PREP=R/"matcha_state_continuation_v3_preparation"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
    with Path(path).open("x") as f:json.dump(obj,f,indent=2,sort_keys=True)
def ref(p):return {"path":str(p),"sha256":sha(p)}
def main():
    PREP.mkdir(exist_ok=False)
    source={str(p):sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in (".py",".md")}
    old=R/"matcha_exact_resume_v1/binding.json"
    assert sha(old)=="0ccab9df17a023a1a81782cff46d4693496245889bcd5479096916dc6a419826"
    binding=json.loads(old.read_text())
    # Retain every non-engineering source/input entry. Old source artifacts stay untouched.
    removed={p:v for p,v in binding["files"].items() if str(ROOT.parent/"matcha_exact_resume")+"/" in p}
    binding["files"]={p:v for p,v in binding["files"].items() if p not in removed}
    binding["files"].update(source)
    write(PREP/"binding.json",binding)
    components={"observed_max_setup":530.0,"historical_prefix_with_10pct_margin":2169.0*1.10,
      "observed_max_full_capture":179.0,"observed_publication":24.0,"source_hash_finite_tail_and_extra_io_reserve":120.0}
    budget={"schema":"matcha_state_continuation_budget_v3","stage":1,"components_seconds":components,
      "forecast_seconds":sum(components.values()),"hard_deadline_seconds":3500,"prediction_not_guarantee":True,
      "no_trajectory_equivalence_claim":True,
      "formal_log":{"path":"/root/MAtCha/output_cambridge_full_retained_v2/StMarysChurch_n20_long_masked_retrain/logs/train_30000_driver.log",
                    "sha256":"d695c4472d21b730b8af937203d555e080b4cdd0ab43a557dfab16b79fd8104e"},
      "native_pair_report":ref(R/"matcha_native_repro_diagnostic_v1/pair_comparison.json"),
      "native_pair_review":ref(R/"matcha_native_repro_diagnostic_v1/independent_pair_result_review.json"),
      "observed_a":ref(R/"matcha_exact_runs/native_n8_trial_a_stmary_20261012_v1/independent_calibration_review.json")}
    write(PREP/"stage1_budget.json",budget)
    plan=R/"matcha_source_repair_v1/training_plan.json"
    spec={"schema":"matcha_state_continuation_launch_v3","stage":1,"physical_gpu":0,"logical_gpu":0,
      "visible_count":1,"uuid":"GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1","deadline_seconds":3500,
      "forecast_seconds":budget["forecast_seconds"],
      "output_root":str(R/"matcha_exact_runs/state_continuation_stmary_20261012_v3_stage1"),
      "references":{"training_plan":ref(plan),"binding":ref(PREP/"binding.json"),
        "input_review":ref(R/"matcha_source_repair_v1/independent_input_review.json"),
        "budget":ref(PREP/"stage1_budget.json"),
        "source_review":{"path":str(PREP/"stage1_source_execution_review.json"),"sha256":"0"*64}},
      "frozen_files":source,"environment":json.loads(plan.read_text())["env_candidate"]}
    write(PREP/"stage1_spec.json",spec)
    write(PREP/"binding_delta.json",{"removed_old_helper_entries":removed,"added_v3_files":source,
      "all_non_helper_bound_entries_preserved":True,"science_args_exactly_unchanged":True})
    print(json.dumps({"prep":str(PREP),"binding":sha(PREP/"binding.json"),"spec":sha(PREP/"stage1_spec.json"),"budget":sha(PREP/"stage1_budget.json"),"source":source},sort_keys=True))
if __name__=="__main__":main()

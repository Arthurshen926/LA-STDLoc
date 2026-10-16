"""Bounded tiny CPU suite. No production checkpoint or CUDA load."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
from pathlib import Path
import copy, importlib.util, inspect, json, textwrap, time, types, unittest, sys
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"matcha_exact_resume_v2"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
old=load("v2_contracts",OLD/"test_cpu_contracts.py")
sys.modules["test_cpu_contracts"]=old
er,torch=old.er,old.torch
import stage_launcher as launcher

class V3Contracts(unittest.TestCase):
    def test_finite_screen_preserves_state_rng_threads(self):
        state={"param":torch.tensor([1.,-0.]),"adam":torch.zeros(3),"accum":[0.0,None]}
        before=er.semantic_digest(state); rng=er.rng_state(); threads=torch.get_num_threads()
        er.finite_state(state)
        self.assertEqual(before,er.semantic_digest(state))
        self.assertTrue(er.equal(rng,er.rng_state())); self.assertEqual(threads,torch.get_num_threads())
    def test_nonfinite_state_and_scalar_rejected(self):
        for value in (float("nan"),float("inf"),float("-inf")):
            for state in (value,{"x":torch.tensor([value])}):
                with self.assertRaises(RuntimeError): er.finite_state(state)
            with self.assertRaises(RuntimeError): er.scalar_record(value)
    def test_restore_gate_and_sequence_fail_closed(self):
        obj=er.Controller.__new__(er.Controller)
        obj.restore_verified=False; obj.in_step=False; obj.next_expected=15000
        obj.binding={"runtime":{"threads":torch.get_num_threads(),"interop":torch.get_num_interop_threads()}}
        with self.assertRaises(RuntimeError): obj.before_step(15000)
        obj.restore_verified=True
        for k in (14999,15001):
            with self.assertRaises(RuntimeError): obj.before_step(k)
        obj.before_step(15000)
        with self.assertRaises(RuntimeError): obj.before_step(15000)
    def test_terminal_finite_check_keeps_gradients_and_optimizer(self):
        model,_,_,_,_=old.example()
        for g in model.optimizer.param_groups: g["params"][0].grad=torch.ones_like(g["params"][0])
        params=[p for g in model.optimizer.param_groups for p in g["params"]]
        before=er.semantic_digest(er.pack([params,[p.grad for p in params],model.optimizer.state_dict()]))
        er.finite_terminal_model(model)
        self.assertEqual(before,er.semantic_digest(er.pack([params,[p.grad for p in params],model.optimizer.state_dict()])))
    def test_trace_coverage_both_stages_and_negatives(self):
        for stage,(first,last) in ((1,(1,14999)),(2,(15000,30000))):
            records=[{"iteration":i} for i in range(first,last+1)]
            if stage==1: records.append({"iteration":14999,"exact_digest":"x"})
            self.assertEqual(launcher.validate_trace(records,stage),(first,last))
            for bad in (records[:-1],records+[records[-1]],records[1:]):
                with self.assertRaises(RuntimeError): launcher.validate_trace(bad,stage)
    def test_argv_no_early_stop_reference_or_extra_steps(self):
        plan={"argv":["native","-m","old","--iterations","30000"]}
        for stage in (1,2):
            spec={"stage":stage,"references":{"binding":{"path":"/binding"},
                  "checkpoint_manifest":{"path":"/prefix/state/step_14999/manifest.json"}}}
            argv=launcher.native_argv(spec,plan,"/rootout")
            self.assertEqual(argv[argv.index("--iterations")+1],"30000")
            self.assertEqual(argv[argv.index("--exact_stop")+1],"14999" if stage==1 else "30000")
            self.assertNotIn("--exact_reference_trace",argv);self.assertNotIn("--early_stop_iteration",argv)
            self.assertEqual("--exact_resume" in argv,stage==2)
    def test_v3_transport_functions_and_trainers_unchanged(self):
        import ast
        a,b=(ast.parse((p/"exact_resume.py").read_text()) for p in (OLD,ROOT))
        def functions(tree): return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        a,b=functions(a),functions(b)
        for name in ("pack","unpack","equal","semantic_digest","atomic_save","read_atomic","capture_model",
                     "restore_model","validate_model","validate_checkpoint","snapshot_static","rng_state","restore_rng"):
            self.assertEqual(a[name],b[name],name)
        for name in ("train_with_charts_exact.py","train_with_charts_observer.py","diagnostic_cpu.py"):
            self.assertEqual((ROOT/name).read_bytes(),(OLD/name).read_bytes())
    def test_unique_scope_and_budget_negative(self):
        R="/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1"
        prep=Path(R)/"matcha_state_continuation_v3_preparation"
        spec=json.loads((prep/"stage1_spec.json").read_text()); budget=json.loads((prep/"stage1_budget.json").read_text())
        launcher.validate_spec(spec);launcher.validate_budget(spec,budget)
        for key,value in (("stage",3),("physical_gpu",1),("uuid","GPU-wrong"),("forecast_seconds",3380),
                          ("deadline_seconds",3600),("output_root",spec["output_root"]+"reuse")):
            bad=copy.deepcopy(spec);bad[key]=value
            with self.assertRaises(RuntimeError):launcher.validate_spec(bad)
        bad=copy.deepcopy(budget);bad["components_seconds"]["observed_max_setup"]=1
        with self.assertRaises(RuntimeError):launcher.validate_budget(spec,bad)

# Keep the previously audited full controller tiny-state roundtrip, with authentic boundary
# labels14999/15000 only; this fixture does not run14999 training steps.
source=textwrap.dedent(inspect.getsource(old.Contracts.test_controller_full_boundary_static_and_restore_micro))
source=source.replace("exact_stop=16,","exact_stop=14999,")
source=source.replace('exact_save=[], exact_windows=["16:17"],','exact_save=[14999], exact_windows=[], exact_reference_trace=None, exact_continue_gate=None, exact_reference_terminal=None, exact_prefix_spec=None,')
source=source.replace("iteration=16,","iteration=14999,")
source=source.replace('controller.observe_view(a)','controller.next_expected = 14999\n        controller.before_step(14999)\n        controller.observe_view(a)')
source=source.replace("baseline/step_00016","baseline/step_14999")
source=source.replace("args.exact_stop = 17","args.exact_stop = 30000\n        args.exact_save = []")
source=source.replace('b["first_iter"], 16','b["first_iter"], 14999')
source=source.replace("restore.trace.close()","self.assertTrue((root / 'branch/restore_bitwise_PASS.json').is_file())\n        restore.before_step(15000)\n        restore.trace.close()\n        restore.performance.close()")
scope=dict(vars(old))
exec(compile(source,str(__file__),"exec"),scope)
V3Contracts.test_controller_full_boundary_static_and_restore_micro=scope["test_controller_full_boundary_static_and_restore_micro"]

def main():
    start=time.monotonic(); suite=unittest.TestSuite()
    names=["test_exact_model_adam_rng_roundtrip_and_next_step","test_original_30000_scheduler_at_authentic_events",
      "test_atomic_readback_no_rng_consumption","test_tensor_bitwise_stride_and_nan",
      "test_none_empty_order_and_phase_are_preserved","test_native_sampling_refill_and_mutable_cadence_replay",
      "test_reject_corrupt_or_incompatible_state_before_step","test_reject_changed_config",
      "test_reject_gradient_and_wrong_parameter_binding","test_reject_truncated_payload",
      "test_native_parser_science_recipe_without_gpu","test_stale_overlapping_file_binding_rejected",
      "test_generated_copies_match_fixed_hash_builders","test_missing_loss_component_rejected_and_signed_zero_exact",
      "test_original_source_frozen_and_no_cuda"]
    for name in names:suite.addTest(old.Contracts(name))
    for name,path in (("signal_contracts",OLD/"test_launcher_cpu.py"),("guard_contracts",OLD/"test_diagnostic_cpu.py")):
        module=load(name,path); suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(V3Contracts))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"schema":"matcha_v3_cpu_contracts","tests_run":result.testsRun,"failures":len(result.failures),
      "errors":len(result.errors),"cuda_initialized":torch.cuda.is_initialized(),"seconds":time.monotonic()-start,
      "production_data_loaded":False,"only_tiny_cpu_fixtures":True}
    out=ROOT/"cpu_validation_v2.json"
    with out.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report)); raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
if __name__=="__main__":main()

"""Stage2-specific tiny JSON unit contracts; source/real artifacts are verified separately after14999."""
import copy,json,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
V3=Path("/root/STDLoc/experiments/matcha_state_continuation_v3")
sys.path.insert(0,str(V3))
import stage_launcher as launch
class Stage2Contracts(unittest.TestCase):
    def fixture(self):
        P=Path("/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1/matcha_state_continuation_v3_preparation")
        s=json.loads((P/"stage1_spec.json").read_text());s["stage"]=2;s["forecast_seconds"]=100.
        s["output_root"]=str(launch.RUNS/(launch.SERIES+"_stage2"))
        prefix=launch.RUNS/(launch.SERIES+"_stage1")
        for key,path in {"prefix_result":prefix/"launcher_result.json","prefix_terminal":P/"official.json",
          "checkpoint_manifest":prefix/"state/step_14999/manifest.json","static_manifest":prefix/"state/static/manifest.json"}.items():
            s["references"][key]={"path":str(path),"sha256":"a"*64}
        b={"schema":"matcha_state_continuation_budget_v3","stage":2,"forecast_seconds":100.,
          "components_seconds":{"fixture_not_actual":100.},"hard_deadline_seconds":3500,
          "prediction_not_guarantee":True,"no_trajectory_equivalence_claim":True,
          "measured_prefix_result_sha256":"a"*64,"measured_checkpoint_manifest_sha256":"a"*64,
          "measured_static_manifest_sha256":"a"*64}
        docs={"source_review":{"approval_scope":"state_exact_continuation_v3_stage2"},"budget":b,
          "prefix_terminal":{"schema":"matcha_v3_official_prefix_terminal","status":"completed",
            "exit_code":0,"job_id":"wc_job_fixture","launcher_result_sha256":"a"*64},
          "prefix_result":{"verdict":"PREFIX14999_COMPLETE","terminal":14999,"native_horizon":30000,
             "floating_trajectory_equivalence_claimed":False}}
        return s,docs
    def verify(self,s,d):
        # Only stagespecific logic is unit-tested here; already-audited base verifier is deliberately stubbed.
        with patch.object(launch.base,"verify_frozen",return_value=d):launch.verify_frozen(s,time.monotonic()+5)
    def test_positive_small_stage2_contract(self):
        s,d=self.fixture();launch.validate_spec(s);self.verify(s,d)
    def test_no_official_success_rejected(self):
        for key,value in (("status","running"),("exit_code",1),("launcher_result_sha256","b"*64),
                          ("job_id","unofficial")):
            s,d=self.fixture();d["prefix_terminal"][key]=value
            with self.assertRaises(RuntimeError):self.verify(s,d)
    def test_wrong_prefix_and_claim_rejected(self):
        for key,value in (("verdict","FAILED"),("terminal",14998),("native_horizon",29999),
                          ("floating_trajectory_equivalence_claimed",True)):
            s,d=self.fixture();d["prefix_result"][key]=value
            with self.assertRaises(RuntimeError):self.verify(s,d)
    def test_wrong_checkpoint_or_static_reference_rejected(self):
        for key in ("checkpoint_manifest","static_manifest","prefix_result"):
            s,d=self.fixture();s["references"][key]["path"]+="_other"
            with self.assertRaises(RuntimeError):self.verify(s,d)
    def test_missing_actual_asset_budget_binding_rejected(self):
        for key in ("measured_prefix_result_sha256","measured_checkpoint_manifest_sha256",
                    "measured_static_manifest_sha256"):
            s,d=self.fixture();d["budget"][key]="b"*64
            with self.assertRaises(RuntimeError):self.verify(s,d)
    def test_unapproved_scope_or_insufficient_budget_rejected(self):
        s,d=self.fixture();d["source_review"]["approval_scope"]="state_exact_continuation_v3_stage1"
        with self.assertRaises(RuntimeError):self.verify(s,d)
        for value in (3380.,3500.,float("nan"),-1.):
            s,d=self.fixture();s["forecast_seconds"]=value
            with self.assertRaises(RuntimeError):launch.validate_spec(s)
if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Stage2Contracts))
    print(json.dumps({"tests_run":result.testsRun,"success":result.wasSuccessful(),
      "torch_imported":"torch" in sys.modules,"actual_checkpoint_read":False,
      "stub_scope":"base.verify_frozen only; actual whole-spec validation still mandatory"}))
    raise SystemExit(0 if result.wasSuccessful() and "torch" not in sys.modules else 1)

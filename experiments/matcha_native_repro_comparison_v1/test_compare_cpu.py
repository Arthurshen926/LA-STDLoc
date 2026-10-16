import copy
import json
import math
from pathlib import Path
import struct
import unittest
import tempfile
import numpy as np
import torch
import compare_pair as c

class PairContracts(unittest.TestCase):
    def test_float32_ulp_and_all_paths(self):
        a=torch.tensor([1.,2.,3.])
        b=a.clone(); b[1]=torch.nextafter(b[1],torch.tensor(float("inf")))
        result=c.differences({"model":a,"scalar":2,"list":[1,2]}, {"model":b,"scalar":3,"list":[1,3]})
        self.assertEqual(len(result),3)
        t=next(r for r in result if r["kind"]=="tensor_values")
        self.assertEqual(t["differing_elements"],1)
        self.assertEqual(t["max_finite_ulp_distance"],1)
        self.assertEqual(t["first_differences"][0]["flat_index"],1)
    def test_nan_payload_signed_zero_are_not_equal(self):
        a=np.array([0x7fc00001,0x00000000],dtype=np.uint32).view(np.float32)
        b=np.array([0x7fc00002,0x80000000],dtype=np.uint32).view(np.float32)
        r=c.array_summary(a,b)
        self.assertEqual(r["differing_elements"],2)
        self.assertEqual(r["nonfinite_mismatches"],1)
        self.assertEqual(r["max_finite_ulp_distance"],1)
        self.assertTrue(c.differences(0.,-0.))
    def test_bfloat_integer_and_noncontiguous_chunks(self):
        a=torch.tensor([1.,2.],dtype=torch.bfloat16)
        b=torch.tensor([1.,3.],dtype=torch.bfloat16)
        self.assertEqual(c.array_summary(a,b)["differing_elements"],1)
        old=c.CHUNK;c.CHUNK=3
        try:
            x=np.arange(40,dtype=np.int64).reshape(8,5).T
            y=x.copy();y[1,3]=-1;y[4,7]=-2
            r=c.array_summary(x,y)
            self.assertEqual(r["differing_elements"],2)
            self.assertEqual([z["flat_index"] for z in r["first_differences"]],[11,39])
        finally:c.CHUNK=old
    def test_missing_type_shape_are_explicit(self):
        r=c.differences({"a":torch.zeros(3),"extra":1},{"a":torch.zeros(4),"new":2})
        self.assertEqual(len(r),3)
        self.assertTrue(c.differences([1],(1,)))
    def test_examples_bounded_but_all_differences_counted(self):
        r=c.array_summary(np.zeros(2000,dtype=np.float32),np.ones(2000,dtype=np.float32))
        self.assertEqual(r["differing_elements"],2000)
        self.assertEqual(len(r["first_differences"]),16)
    def test_runtime_ignores_only_pid_and_allocated(self):
        a={"context":{"pid":1,"allocated_bytes":10,"uuid":"one"},"runtime":{"threads":32}}
        b=copy.deepcopy(a);b["context"].update(pid=2,allocated_bytes=1000)
        self.assertFalse(c.differences(c.normalized_runtime(a),c.normalized_runtime(b)))
        b["runtime"]["threads"]=1
        self.assertTrue(c.differences(c.normalized_runtime(a),c.normalized_runtime(b)))
    def test_float64_overflow_is_json_safe_and_zero_tol(self):
        a=np.array([np.finfo(np.float64).max]);b=-a
        r=c.array_summary(a,b)
        self.assertTrue(r["absolute_difference_overflow"])
        json.dumps(r,allow_nan=False)
    def test_common_wrong_native_flags_or_child_identity_rejected(self):
        u=c.registration()
        spec={"schema":"matcha_native_aa_launch_v1","stage":"native_aa_n8","pair_id":u.PAIR,
            "replicate":"u1","physical_gpu":0,"logical_gpu":0,"visible_count":1,"uuid":u.ASSIGNED,
            "deadline_seconds":1200,"forecast_seconds":900,"output_root":str(u.RUNS/(u.PAIR+"_u1")),
            "references":{k:{"path":"/fixture/"+k,"sha256":"0"*64} for k in u.base.REFS},
            "frozen_files":{str(p):"0"*64 for p in u.SOURCE_FILES},"environment":{}}
        result={"schema":"matcha_native_aa_result_v1","native_horizon":30000,
            "comparison_executed":False,"production_continuation_allowed":False}
        complete={"schema":"matcha_native_aa_child_v1","replicate":"u1","native_horizon":30000,
            "observed_terminal":8,"guard_used":False,"comparison_executed":False,"production_continuation_allowed":False}
        proof={"runtime":{"threads":32,"interop":16,"deterministic":False,"warn_only":False,
            "cudnn_benchmark":False,"cudnn_deterministic":False,"cudnn_tf32":True,"matmul_tf32":False},
            "autocast_enabled":False,"cpu_autocast_enabled":False,"diagnostic_guard_imported":False,
            "original_helper_path":str(c.V1/"exact_resume.py")}
        c.validate_run_headers(spec,result,complete,proof,proof,"u1")
        for key,value in (("replicate","u2"),("native_horizon",8),("comparison_executed",True),
                          ("production_continuation_allowed",True)):
            bad=dict(complete);bad[key]=value
            with self.assertRaises(RuntimeError):c.validate_run_headers(spec,result,bad,proof,proof,"u1")
        bad=copy.deepcopy(proof);bad["runtime"]["threads"]=1
        with self.assertRaises(RuntimeError):c.validate_run_headers(spec,result,complete,bad,bad,"u1")
    def test_terminal_jobs_must_be_distinct_and_complete_before_payload_load(self):
        with tempfile.TemporaryDirectory() as directory:
            roots={k:Path(directory)/k for k in ("u1","u2")}
            for root in roots.values():root.mkdir();(root/"launcher_result.json").write_text("{}")
            data={"schema":"matcha_native_pair_runner_receipt_v1","runs":{k:{"status":"completed",
                "exit_code":0,"job_id":"wc_job_"+k,"root":str(root),
                "launcher_result_sha256":c.sha(root/"launcher_result.json")} for k,root in roots.items()}}
            path=Path(directory)/"terminal.json"
            path.write_text(json.dumps(data));c.validate_terminal_receipt(path,roots)
            for key,value in (("job_id","wc_job_u1"),("exit_code",1),("status","running")):
                bad=copy.deepcopy(data);bad["runs"]["u2"][key]=value
                path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):c.validate_terminal_receipt(path,roots)

    def test_equal_returns_no_inventory_and_no_cuda(self):
        values={"t":torch.tensor([1.,2.]),"n":np.arange(4,dtype=np.uint8),"rng":(None,[],0)}
        self.assertFalse(c.differences(values,copy.deepcopy(values)))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main(verbosity=2)

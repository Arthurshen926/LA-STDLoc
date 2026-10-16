import unittest
import tempfile
from pathlib import Path
from experiments.native_baseline_repair.query_report import (summarize, SCENES, metrics,
    check_cell_bindings, check_receipt, actual_deployment_bytes, ref, RESULT_VERIFIER_SHA)

def fixture():
    cells = []
    for i, scene in enumerate(SCENES):
        # Unequal query counts ensure macro statistics are not pooled.
        te = [float(i + 1)] * (i + 1); re = [0.1] * len(te)
        cells.append(dict(scene=scene, method="STDLoc", mapping_semantics="off",
            deployment_bytes=100 + i,
            metrics={s: metrics(te, re) for s in ("sparse", "dense")},
            errors={s: {"te_cm": te, "re_deg": re} for s in ("sparse", "dense")}))
    return cells

class ReportTest(unittest.TestCase):
    def test_equal_scene_not_pooled(self):
        r = summarize(fixture())
        self.assertEqual(r["dense"]["equal_scene"]["median_te_cm"], 3.0)
        self.assertEqual(r["dense"]["pooled"]["median_te_cm"], 4.0)
        self.assertEqual(r["dense"]["query_count"], 15)
    def test_storage_total_mean(self):
        r = summarize(fixture())["storage"]
        self.assertEqual(r["total_bytes"], 510)
        self.assertEqual(r["mean_bytes"], 102)
        self.assertEqual(r["total_decimal_MB"], 0.00051)
    def test_missing_scene_rejected(self):
        with self.assertRaises(ValueError): summarize(fixture()[:-1])
    def test_duplicate_scene_rejected(self):
        x = fixture(); x[-1]["scene"] = x[0]["scene"]
        with self.assertRaises(ValueError): summarize(x)
    def test_cross_condition_rejected(self):
        x = fixture(); x[-1]["mapping_semantics"] = "on"
        with self.assertRaises(ValueError): summarize(x)
    def test_failure_outlier_retained(self):
        x = fixture(); x[0]["errors"]["dense"]["te_cm"] = [10001.]
        x[0]["metrics"]["dense"] = metrics([10001.], [0.1])
        r = summarize(x)["dense"]
        self.assertEqual(r["gt_1m_count"], 1)
        self.assertGreater(r["equal_scene"]["median_te_cm"], 2000)
class BindingTest(unittest.TestCase):
    def fixture(self):
        p=Path("/owned/cell")
        c=dict(case="cell",receipt=str(p/"receipt.json"),native_output=str(p/"native_output"),
               method="STDLoc",mapping_semantics="off")
        code=dict(path="/repo/query_results.py",sha256=RESULT_VERIFIER_SHA,bytes=123)
        paths=dict(contract=p/"contract.json",terminal_evidence=p/"terminal.json",
            native_receipt=p/"receipt.json",results=p/"native_output/results.json",
            native_log=p/"launch/native.log",launch=p/"launch/launch.json")
        v={k:dict(path=str(x),sha256="same-content-hash",bytes=10) for k,x in paths.items()}
        v["verifier"]=dict(code)
        receipt=dict(status="complete",method="STDLoc",phase="evaluate",
            mapping_semantics="off",query_semantics=True,seed=2026,warmup_queries=0,
            native_output=c["native_output"],timings=[{}]*103)
        return p,c,code,v,receipt
    def test_correct_bindings(self):
        p,c,code,v,r=self.fixture()
        check_cell_bindings(v,c,p,code);check_receipt(r,c,103)
    def test_cross_cell_paths_rejected(self):
        for key in ("results","native_log","native_receipt","launch","terminal_evidence","contract"):
            p,c,code,v,r=self.fixture()
            v[key]["path"]=v[key]["path"].replace("/owned/cell/","/owned/other/")
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_cell_bindings(v,c,p,code)
    def test_actual_verifier_drift_rejected(self):
        p,c,code,v,r=self.fixture()
        code["sha256"]="changed"
        with self.assertRaises(ValueError):check_cell_bindings(v,c,p,code)
    def test_receipt_condition_rejected(self):
        for key,value in (("method","ULFLoc"),("mapping_semantics","on"),("seed",2025),
            ("query_semantics",False),("warmup_queries",50),("status","running"),
            ("native_output","/other")):
            p,c,code,v,r=self.fixture();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_receipt(r,c,103)
    def test_cell_identity_rejected(self):
        p,c,code,v,r=self.fixture();c["case"]="other"
        with self.assertRaises(ValueError):check_cell_bindings(v,c,p,code)
    def test_actual_storage_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"asset";p.write_bytes(b"1234567")
            a={"native":ref(p)}
            self.assertEqual(actual_deployment_bytes(a),7)
            a["native"]["bytes"]=999
            with self.assertRaises(ValueError):actual_deployment_bytes(a)
    def test_missing_query_timing_rejected(self):
        p,c,code,v,r=self.fixture();r["timings"].pop()
        with self.assertRaises(ValueError):check_receipt(r,c,103)
if __name__ == "__main__": unittest.main()

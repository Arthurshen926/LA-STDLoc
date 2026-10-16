"""Persist the new thin launcher microtests without modifying A/B sources or evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import native_aa_launcher as u
import test_native_aa_cpu
import test_launcher_cpu
root = Path(__file__).resolve().parent
started = time.monotonic()
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in
                           (test_native_aa_cpu, test_launcher_cpu))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {"schema":"matcha_native_aa_micro_cpu_v1", "passed":result.wasSuccessful(),
    "tests_run":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
    "seconds":time.monotonic()-started,"torch_imported": "torch" in sys.modules,
    "GPU_launched":False,"production_input_read":False,
    "new_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")},
    "original_files":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in u.SOURCE_FILES if p.parent==u.V1}}
with (root/"cpu_validation.json").open("w") as f: json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps(report,sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and "torch" not in sys.modules else 1)

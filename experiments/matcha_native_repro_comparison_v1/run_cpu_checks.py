import hashlib
import json
from pathlib import Path
import unittest
import time
import test_compare_cpu
import torch
root=Path(__file__).resolve().parent
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_compare_cpu))
report={"schema":"matcha_native_pair_comparator_micro_v1","passed":result.wasSuccessful(),
    "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
    "seconds":time.monotonic()-started,"cuda_initialized":torch.cuda.is_initialized(),
    "production_state_read":False,"maximum_fixture_elements":2000,
    "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}}
(root/"cpu_validation.json").write_text(json.dumps(report,sort_keys=True,indent=2))
print(json.dumps(report,sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)

import pytest
from experiments.cambridge_mainline_ablations import t6_serial_resume as R
def test_scope():
 assert len(R.CELLS)==10 and len(set(R.CELLS))==10
 assert all(m in ("filter_off_fused","filter_off_single") for m,s in R.CELLS)
def test_exact_identity():
 a=dict(scene="ShopFacade",map_sha256="m",files={n:{} for n in ("results.json","manifest.json","summary.json")})
 R.identity(a,dict(scene="ShopFacade",map_sha256="m"))
 with pytest.raises(ValueError):R.identity(a,dict(scene="wrong"))
 with pytest.raises(ValueError):R.identity(a,dict(map_sha256="wrong"))
 a["files"]={}
 with pytest.raises(ValueError):R.identity(a,{})
def test_cpu_parent_requires_empty_visibility(monkeypatch):
 monkeypatch.setattr(R.torch.cuda,"is_initialized",lambda:False)
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
 with pytest.raises(ValueError):R.cpu_parent()
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");R.cpu_parent()
 monkeypatch.setattr(R.torch.cuda,"is_initialized",lambda:True)
 with pytest.raises(ValueError):R.cpu_parent()
def test_vacancy_waits_without_accepting_unknown_pid(monkeypatch):
 seq=iter([{2316},set()])
 monkeypatch.setattr(R.S,"gpu_process_ids",lambda:next(seq))
 monkeypatch.setattr(R.time,"sleep",lambda _:None)
 assert R.wait_empty()==[[2316],[]]
def test_vacancy_timeout_rejects_foreign(monkeypatch):
 monkeypatch.setattr(R.S,"gpu_process_ids",lambda:{2316})
 with pytest.raises(RuntimeError,match="no PID assumed owned"):R.wait_empty(timeout=0)
def test_scheduler_bound_and_child_gpu_explicit():
 assert "experiments/cambridge_mainline_ablations/accuracy_scheduler.py" in R.hashes()
 assert R.S.worker_env()["CUDA_VISIBLE_DEVICES"]=="0"

import copy,pytest
from experiments.cambridge_mainline_ablations import t5_serial_resume as R
def fixture():
 expected=dict(protocol_sha256="p",map_sha256="m",query_inputs_sha256="q",scene="ShopFacade",configuration="ray")
 return dict(expected,files={n:{} for n in ("results.json","manifest.json","summary.json")}),expected
def test_exact_scope():
 assert len(R.REMAINING)==12 and len(set(R.REMAINING))==12
 assert len(R.REUSED)==3 and not set(R.REMAINING)&set(R.REUSED)
 assert all(s!="ShopFacade" for m,s in R.REMAINING)
 assert [m for m,s in R.REUSED]==list(R.T.CONFIGS)
def test_identity_accepts_exact():
 a,e=fixture();R.identity(a,e)
@pytest.mark.parametrize("key",["protocol_sha256","map_sha256","query_inputs_sha256","scene","configuration"])
def test_identity_rejects_wrong_binding(key):
 a,e=fixture();a[key]="wrong"
 with pytest.raises(ValueError,match="identity differs"):R.identity(a,e)
@pytest.mark.parametrize("files",[{},{"results.json":{}},{"results.json":{},"manifest.json":{},"summary.json":{},"extra":{}}])
def test_rejects_missing_or_extra_files(files):
 a,e=fixture();a["files"]=files
 with pytest.raises(ValueError,match="file set differs"):R.identity(a,e)
def test_cpu_parent_checks_vacancy_and_waits_for_child(monkeypatch):
 calls=[]
 monkeypatch.setattr(R.S,"current_admission",lambda concurrency:calls.append(("gate",concurrency)))
 monkeypatch.setattr(R.subprocess,"run",lambda command,**kw:calls.append(("child",command,kw)))
 R.execute_cell("ray","OldHospital")
 assert calls[0]==("gate",1)
 assert calls[1][1]==R.S.query_command("T5",R.SOURCE,"ray","OldHospital")
 assert calls[1][2]["check"] is True and calls[1][2]["timeout"]==7200
 assert calls[1][2]["env"]["CUDA_VISIBLE_DEVICES"]=="0"
 with pytest.raises(ValueError,match="outside fixed"):R.execute_cell("ray","ShopFacade")
def test_scheduler_dependency_is_bound():
 assert "experiments/cambridge_mainline_ablations/accuracy_scheduler.py" in R.code_hashes()


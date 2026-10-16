"""CPU-only input availability contracts; no GPU or filesystem production outputs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.cambridge_mainline_inputs.adapter import adapt_selector_state, source_contract, validate_csr
from experiments.cambridge_mainline_inputs.driver import rebind_raw, assert_gpu, compare_snapshot, DESIGN
from experiments.cambridge_input_matrix.driver import construct_any
from experiments.cambridge_mechanism_stage_a.driver import adapt_new_state
from anygsloc.mapping.compact_v2 import select_compact_rows_v2

spec=importlib.util.spec_from_file_location("input_fixture",Path(__file__).with_name("test_cambridge_input_matrix.py"))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
torch.set_num_threads(1)

def example(mode="pure_gs"):
    p,_=fixture.synthetic(mode)
    if mode=="hybrid":
        for r in p["queries"].values():
            r["row_source"][:4]=0
            r["source_row"][4:]-=4
    if mode!="raw_only":
        for r in p["queries"].values():r["v2_valid"][1]=False
    xyz,f,e=construct_any(p,device="cpu",workers=1)
    return p,e,{"anchor_xyz":torch.from_numpy(xyz),"anchor_features":torch.from_numpy(f)}

@pytest.mark.parametrize("mode",DESIGN["modes"])
def test_true_quality_covariance_and_scope(mode):
    p,e,d=example(mode);s=adapt_selector_state(e,p,d)
    assert torch.equal(s["anchor_xyz"],d["anchor_xyz"])
    assert torch.equal(s["anchor_features"],d["anchor_features"])
    assert torch.equal(s["anchor_matchability"],torch.cat([v["identity_reliability"]*v["geometry_reliability"] for v in e["parts"]]).clamp(0,1))
    assert torch.equal(s["anchor_position_covariance"],torch.cat([v["anchor_position_covariance"] for v in e["parts"]]))
    assert s["provenance"]["uses_source_mapping_rgb"] is (mode!="pure_gs")
    assert s["provenance"]["uses_test_queries"] is False
    assert "test_names" not in s and "filtered_observations" not in s
    assert not torch.cuda.is_initialized()

def test_pure_adapter_numeric_equivalence_to_accepted_stage_a():
    p,e,d=example();a=adapt_new_state(e,p,d);b=adapt_selector_state(e,p,d)
    for k in ("anchor_ids","anchor_xyz","anchor_features","anchor_matchability","anchor_position_covariance","v6_mapping_query_bins"):
        assert torch.equal(a[k],b[k])
    for k in ("observation_offsets","query_indices","keypoint_indices"):
        assert torch.equal(a["projective_anchor_observations"][k],b["projective_anchor_observations"][k])

def test_hybrid_source_local_rows_never_become_global_cache_rows():
    p,e,d=example("hybrid");s=adapt_selector_state(e,p,d)
    rows=s["projective_anchor_observations"]["keypoint_indices"]
    assert 1 not in rows and int(rows.max())>=4
    assert len(next(iter(p["queries"].values()))["source_row"].unique())==4

@pytest.mark.parametrize("mode",DESIGN["modes"])
@pytest.mark.parametrize("where",["original","filtered","part"])
def test_false_source_labels_rejected(mode,where):
    p,e,d=example(mode)
    data={"original":p,"filtered":e["filtered_observations"],"part":e["parts"][0]}[where]
    data["uses_source_mapping_rgb"]=mode=="pure_gs"
    with pytest.raises(ValueError,match="provenance"):adapt_selector_state(e,p,d)

@pytest.mark.parametrize("where",["original","filtered","part"])
def test_test_input_provenance_rejected(where):
    p,e,d=example("hybrid")
    {"original":p,"filtered":e["filtered_observations"],"part":e["parts"][0]}[where]["uses_test_queries"]=True
    with pytest.raises(ValueError,match="test-input"):adapt_selector_state(e,p,d)

@pytest.mark.parametrize("kind",["offset","negative","oob","camera_duplicate"])
def test_malformed_csr_rejected(kind):
    p,e,d=example();c=e["parts"][0]["projective_anchor_observations"]
    if kind=="offset":c["observation_offsets"][-1]-=1
    elif kind=="negative":c["keypoint_indices"][0]=-1
    elif kind=="oob":c["keypoint_indices"][0]=100000
    else:c["query_indices"][1]=c["query_indices"][0]
    with pytest.raises(ValueError):adapt_selector_state(e,p,d)

def test_filtered_rows_and_full_changes_rejected():
    p,e,d=example("hybrid");bad=copy.deepcopy(e)
    next(iter(bad["filtered_observations"]["queries"].values()))["native_scores"][0]+=1
    with pytest.raises(ValueError,match="filtered/source"):adapt_selector_state(bad,p,d)
    d["anchor_xyz"][0,0]+=1
    with pytest.raises(ValueError,match="deployed Full"):adapt_selector_state(e,p,d)

def test_raw_gaussian_module_availability_rejected():
    p,e,d=example("raw_only");e["v2_enabled"]=True
    with pytest.raises(ValueError,match="V2"):adapt_selector_state(e,p,d)
    e["v2_enabled"]=False;e["completion_reason"]=None
    with pytest.raises(ValueError,match="completion"):adapt_selector_state(e,p,d)

def test_native_selector_same_kernel_for_all_modes_and_deterministic():
    for mode in DESIGN["modes"]:
        p,e,d=example(mode);state=adapt_selector_state(e,p,d)
        a,_=select_compact_rows_v2(state,p["queries"])
        b,_=select_compact_rows_v2(state,p["queries"])
        assert torch.equal(a,b) and len(a)>0 and int(a.max())<len(state["anchor_ids"])
        changed=copy.deepcopy(p);changed["test_names"]=["different_test.png"];changed["irrelevant_test_poses"]=[99.]
        other=adapt_selector_state(e,changed,d)
        c,_=select_compact_rows_v2(other,changed["queries"])
        assert torch.equal(a,c)

def test_gpu_other_than_physical_one_rejected(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    assert_gpu("cuda:1")
    for v in ("cpu","cuda","cuda:0","cuda:2"):
        with pytest.raises(ValueError):assert_gpu(v)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    with pytest.raises(ValueError):assert_gpu("cuda:1")

def test_snapshot_mismatch_rejected():
    a={"mapping_registry":[],"calibration":{},"mask_sha256":"m","queries":[]}
    compare_snapshot(a,copy.deepcopy(a))
    for k in a:
        b=copy.deepcopy(a);b[k]="changed"
        with pytest.raises(ValueError):compare_snapshot(a,b)

def test_rebind_only_pose_changes():
    p,_=fixture.synthetic("raw_only");cams=[]
    for n,r in p["queries"].items():
        K=r["native_K"].numpy()
        cams.append(SimpleNamespace(image_name=n,pose_w2c=(r["pose_w2c"]+torch.diag(torch.tensor([0.,0.,0.,0.]))).numpy(),
            height=480,width=640,fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2])))
    # Intrinsics implementation is separately covered by the dataset/registry audit;
    # use the same exact known intrinsic for this CPU rebind unit test.
    from unittest.mock import patch
    with patch("experiments.cambridge_mainline_inputs.driver.camera_intrinsic",lambda c:p["queries"][c.image_name]["native_K"]):
        out=rebind_raw(p,{"mapping_names":p["mapping_names"],"test_names":p["test_names"]},cams,"new")
    for n in p["mapping_names"]:
        for k in ("native_keypoints","native_descriptors","native_scores","row_source","source_row"):
            assert torch.equal(out["queries"][n][k],p["queries"][n][k])

"""Stage-A CPU-only contracts and historical numerical parity."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.cambridge_mechanism_stage_a.driver import adapt_new_state, equal_geometry, BUDGET, DESIGN
from experiments.cambridge_mechanism_stage_a.parity import numerical_regression
from experiments.cambridge_input_matrix.driver import construct_any
# Reuse the existing synthetic observations without duplicating a numerical kernel.
spec=importlib.util.spec_from_file_location("matrix_test_fixture",Path(__file__).with_name("test_cambridge_input_matrix.py"))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
synthetic=fixture.synthetic

def test_load_uses_installed_torch_contract(tmp_path):
    from experiments.cambridge_mechanism_stage_a.driver import load
    p=tmp_path/"state.pt";torch.save({"value":torch.arange(4)},p)
    assert torch.equal(load(p)["value"],torch.arange(4))


@pytest.mark.parametrize("perturb",[False,True])
def test_historical_current_experimental_numeric_parity(perturb):
    p,_=synthetic()
    if perturb:
        for i,r in enumerate(p["queries"].values()):
            r["native_keypoints"] += torch.tensor([.03*(i-1),-.02*i])
            r["native_scores"] *= .8+.02*i
    assert numerical_regression(p)["exact_tensor_equality"]

def example():
    p,_=synthetic()
    # Intentionally create a hole so local row indices cannot be mistaken for original ones.
    for r in p["queries"].values():r["v2_valid"][1]=False
    xyz,features,e=construct_any(p,device="cpu",workers=1)
    return p,e,{"anchor_xyz":torch.from_numpy(xyz),"anchor_features":torch.from_numpy(features)}

def test_adapter_preserves_geometry_quality_covariance_and_true_source_rows():
    p,e,d=example();s=adapt_new_state(e,p,d)
    assert torch.equal(s["anchor_xyz"],d["anchor_xyz"])
    assert torch.equal(s["anchor_features"],d["anchor_features"])
    assert 1 not in s["projective_anchor_observations"]["keypoint_indices"]
    assert torch.equal(s["anchor_position_covariance"],torch.cat([x["anchor_position_covariance"] for x in e["parts"]]))
    assert torch.equal(s["anchor_matchability"],torch.cat([x["identity_reliability"]*x["geometry_reliability"] for x in e["parts"]]).clamp(0,1))
    assert not torch.cuda.is_initialized()

def test_filtered_source_corruption_rejected():
    p,e,d=example();p["queries"][p["mapping_names"][0]]["native_scores"][0]+=1
    with pytest.raises(ValueError,match="filtered/source"):adapt_new_state(e,p,d)

@pytest.mark.parametrize("field",["pose_w2c","native_K","native_input_hw"])
def test_camera_geometry_difference_rejected(field):
    p,_=synthetic();a=p["queries"];b=copy.deepcopy(a)
    b[p["mapping_names"][0]][field].reshape(-1)[0]+=1
    with pytest.raises(ValueError,match="geometry differs"):equal_geometry(a,b,p["mapping_names"])

def test_no_test_or_raw_provenance_is_accepted():
    p,e,d=example();p["uses_test_queries"]=True
    with pytest.raises(ValueError,match="source-free"):adapt_new_state(e,p,d)
    p["uses_test_queries"]=False;e["parts"][0]["uses_source_mapping_rgb"]=True
    with pytest.raises(ValueError,match="source-free"):adapt_new_state(e,p,d)

def test_frozen_budget_and_not_five_scene_benchmark():
    assert BUDGET=={"ShopFacade":12014,"OldHospital":9224,"KingsCollege":7735,"GreatCourt":44631}
    assert "StMarysChurch" not in BUDGET
    assert len(DESIGN["configs"])==5
    assert DESIGN["seed"]==2026

def test_failed_scene_stays_in_macro_denominator():
    from experiments.cambridge_mechanism_stage_a.driver import scene_mean
    assert np.isinf(scene_mean([1.,None,3.,4.]))
    assert scene_mean([1.,2.,3.,4.])==2.5

@pytest.mark.parametrize("change",["map_sha256","query_inputs_sha256","configuration","fresh_evaluation"])
def test_report_rejects_swapped_cell_or_snapshot(tmp_path,change):
    from experiments.cambridge_mechanism_stage_a.driver import verify_evaluation,write_new
    a={"protocol_sha256":"p","query_inputs_sha256":"q","map_sha256":"m","fresh_evaluation":True,
       "scene":"ShopFacade","configuration":"new_native","files":{}}
    a[change]="different"
    write_new(tmp_path/"attestation.json",a)
    with pytest.raises(ValueError,match="identity/configuration/snapshot"):
        verify_evaluation(tmp_path,"ShopFacade","new_native",{},"p","q",{"sha256":"m"})



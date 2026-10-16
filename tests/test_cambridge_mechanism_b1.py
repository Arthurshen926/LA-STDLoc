import copy
import pytest
from experiments.cambridge_mechanism_b1 import check_source,same_snapshot,DESIGN
from experiments.cambridge_mechanism_stage_a.driver import BUDGET

def good():
    return {"provenance":{"input_mode":"pure_gs","method":"classic","uses_source_mapping_rgb":False,"uses_test_queries":False,"cache_sha256":"frozen"}}

def test_existing_classic_mapping_only_source():
    check_source(good(),"frozen")

@pytest.mark.parametrize("key,value",[("input_mode","raw_only"),("method","anygsloc"),("uses_source_mapping_rgb",True),("uses_test_queries",True),("cache_sha256","different")])
def test_wrong_source_rejected(key,value):
    state=good();state["provenance"][key]=value
    with pytest.raises(ValueError):check_source(state,"frozen")

def test_shared_snapshot_all_fields():
    s={"mapping_registry":["m"],"calibration":{"a":1},"mask_sha256":"m","queries":[{"name":"q","image_sha256":"x","pose_w2c":[1]}],"protocol_sha256":"a"}
    b=copy.deepcopy(s);b["protocol_sha256"]="b";same_snapshot(s,b)
    for key in ("mapping_registry","calibration","mask_sha256","queries"):
        b=copy.deepcopy(s);b[key]=None
        with pytest.raises(ValueError):same_snapshot(s,b)

def test_frozen_four_scene_budget_no_new_search():
    assert DESIGN["B"]==BUDGET
    assert "StMarysChurch" not in DESIGN["scene_order"]
    assert DESIGN["construct_or_render"] is False

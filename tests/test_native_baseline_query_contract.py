"""CPU negative checks against a prepared, not-yet-evaluated real cell."""
import copy
import json
from pathlib import Path
import pytest
from experiments.native_baseline_repair.query_plan import validate_contract

ROOT=Path("/mnt/pool/sqy/anygsloc_native_baseline_stmary_repair_20261012_v1/query_semantic_on")

@pytest.fixture
def cell():
    path=ROOT/"cells/ULFLoc_StMarysChurch_mapping_semantics_off_eval/contract.json"
    if not path.exists():pytest.skip("real CPU cell not prepared")
    c=json.loads(path.read_text());p=json.loads(Path(c["protocol"]["path"]).read_text())
    return c,p,Path(c["protocol"]["path"])

def test_real_cell_is_native_recipe(cell):
    c,p,path=cell;validate_contract(c,p,path)

@pytest.mark.parametrize("field,value",[
    ("method","STDLoc"),("scene","ShopFacade"),("mapping_semantics","on"),("seed",7),
    ("query_semantics",False),("query_names",[]),("source","/tmp/other"),
    ("native_output","/tmp/output"),("model_view","/tmp/model"),("receipt","/tmp/result")])
def test_changed_contract_identity_is_rejected(cell,field,value):
    c,p,path=cell;c=copy.deepcopy(c);c[field]=value
    with pytest.raises(ValueError):validate_contract(c,p,path)

def test_native_recipe_and_asset_redirection_are_rejected(cell):
    original,p,path=cell
    for field in ["wrapper_argv","model_assets","env"]:
        c=copy.deepcopy(original)
        if field=="wrapper_argv":c[field]+=["--new-science-flag","true"]
        elif field=="model_assets":c[field]["cfg_args"]["path"]="/tmp/unaccepted-model"
        else:c[field]["ULFLOC_FUSION_BATCH_SIZE"]="100"
        with pytest.raises(ValueError):validate_contract(c,p,path)

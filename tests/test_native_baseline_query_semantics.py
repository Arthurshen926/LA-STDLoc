import ast
import pytest
import torch
from experiments.native_baseline_repair.launch_query import check_lease
from experiments.native_baseline_repair.query_plan import replace_value, asset_names
from experiments.native_baseline_repair.query_plan import verify_accepted_model, ref
from experiments.native_baseline_repair.query_semantic_adapter import mask_rgb, once, std_text

def test_mask_is_native_three_channel_and():
    image=torch.ones(3,2,3)
    a=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    b=torch.tensor([[1,0,1],[1,1,1]],dtype=torch.bool)
    c=torch.tensor([[1,1,1],[0,1,1]],dtype=torch.bool)
    result=mask_rgb("q",image,{"q":(a,b,c)},{"q"})
    assert torch.equal(result,image*(a&b&c)[None])

def test_mask_resize_and_mapping_exclusion():
    image=torch.ones(3,4,4)
    layers=(torch.tensor([[1,0],[1,1]]),torch.ones(2,2),torch.ones(2,2))
    result=mask_rgb("q",image,{"q":layers},{"q"})
    assert result[:,0:2,2:4].count_nonzero()==0
    with pytest.raises(ValueError):mask_rgb("mapping",image,{"mapping":layers},{"q"})

def test_mask_requires_three_channels():
    with pytest.raises(ValueError):mask_rgb("q",torch.ones(3,2,2),{"q":(torch.ones(2,2),)},{"q"})

def test_frozen_unique_source_anchor():
    assert once("abc","b","B")=="aBc"
    with pytest.raises(ValueError):once("abbc","b","B")
    with pytest.raises(ValueError):once("abc","d","D")

def test_std_only_query_and_output_insertions():
    source='def native_solver(x):\n    return x+1\n\nif True:\n    print("Output path:", output_path)\n    for camera_info in cameras:\n        query_image = camera_info.original_image.to("cuda")\n'
    result=std_text(source)
    before=ast.parse(source);after=ast.parse(result)
    assert ast.dump(before.body[0])==ast.dump(after.body[0])
    assert result.count("_QUERY_MASK_RGB(")==1
    assert result.count("_CONTROLLED_OUTPUT")==1


def test_masks_preserve_singleton_dimension_and_reject_rank():
    image=torch.ones(3,1,4)
    layers=(torch.tensor([[1,0,1,1]]),)*3
    assert mask_rgb("q",image,{"q":layers},{"q"}).shape==image.shape
    with pytest.raises(ValueError):
        mask_rgb("q",image,{"q":(torch.ones(1,1,4),)*3},{"q"})

def test_mask_matches_native_ulf_conversion():
    import importlib.util
    from pathlib import Path
    path=Path("/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915/baselines/ULFLoc/utils/mask_utils.py")
    if not path.exists():pytest.skip("native source not installed in this test environment")
    spec=importlib.util.spec_from_file_location("native_mask_utils",path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    generator=torch.Generator().manual_seed(2026)
    for shape in [(1,4),(7,9),(10,8)]:
        image=torch.rand(3,*shape,generator=generator)
        layers=tuple(torch.randn(3,5,generator=generator)*(torch.rand(3,5,generator=generator)>.4) for _ in range(3))
        masks={"q":layers}
        obj,sky,geom=native.masks_for_image(masks,"q",image.shape[-2:],device="cpu")
        expected=image*(obj&geom);expected*=sky
        assert torch.equal(mask_rgb("q",image,masks,{"q"}),expected)

def test_query_lease_bounded_and_contract_bound():
    lease=dict(approved=True,scope="native_baseline_query_semantic_on",contract_sha256="c",
               approval_reference="parent explicit task",owner_handover="completed",gpu_uuid="GPU-test",
               valid_from_unix=0,valid_until_unix=4000)
    assert check_lease(lease,"c",now=10)=="GPU-test"
    for changes in [dict(contract_sha256="other"),dict(valid_until_unix=100),dict(valid_until_unix=8000),dict(approved=False)]:
        with pytest.raises(ValueError):check_lease(dict(lease,**changes),"c",now=10)

def test_native_argument_and_storage_contracts():
    assert replace_value(["-m","old","--cfg","c"],"-m","new")==["-m","new","--cfg","c"]
    with pytest.raises(ValueError):replace_value(["-m","a","-m","b"],"-m","new")
    assert "detector/30000_detector.pth" in asset_names("STDLoc")
    assert "point_cloud/iteration_30000/point_cloud.ply" in asset_names("ULFLoc")
    assert "test/keypoints_features.pkl" in asset_names("ULFLoc")

def test_repaired_acceptance_binds_exact_product(tmp_path):
    import copy
    model=tmp_path/"model";model.mkdir()
    protocol=tmp_path/"protocol.json";protocol.write_text("{}")
    for rel in asset_names("ULFLoc"):
        path=model/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"accepted")
    assets={rel:ref(model/rel) for rel in asset_names("ULFLoc")}
    fields=dict(zip(asset_names("ULFLoc"),["cfg_args","ply","indices","features"]))
    accepted=dict(complete=True,case="ULFLoc_StMarysChurch_mapping_semantics_off_train",method="ULFLoc",
                  semantics="off",seed=2026,iterations=30000,training_protocol=ref(protocol),
                  artifacts={fields[k]:v for k,v in assets.items()})
    key="ULFLoc_StMarysChurch_mapping_semantics_off_eval"
    verify_accepted_model(accepted,key,"ULFLoc","off",model,assets,ref(protocol))
    for field,value in [("iterations",20000),("semantics","on"),("case","legacy_train"),("method","STDLoc")]:
        bad=copy.deepcopy(accepted);bad[field]=value
        with pytest.raises(ValueError):verify_accepted_model(bad,key,"ULFLoc","off",model,assets,ref(protocol))
    (model/"cfg_args").write_bytes(b"changed")
    refreshed={rel:ref(model/rel) for rel in asset_names("ULFLoc")}
    with pytest.raises(ValueError):verify_accepted_model(accepted,key,"ULFLoc","off",model,refreshed,ref(protocol))



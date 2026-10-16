"""CPU-only scientific invariants and pure-GS numerical equivalence."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.cambridge_input_matrix.contracts import MatrixObservationProvider, validate_payload
from experiments.cambridge_input_matrix.driver import (
    DESIGN, payload, merge_camera, safe_mapping_image, construct_any, construct_classic,
    common_rows, assert_unique_track_cameras, check_kernel_adaptations,
)
from anygsloc.mapping.observation_provider import GaussianRenderObservationProvider
from anygsloc.mapping.projective_association import build_projective_association_graph as original_association
from anygsloc.mapping.projective_reconstruction import reconstruct_projective_anchors as original_reconstruction
from anygsloc.mapping.projective_completion import build_projective_completion as original_completion
from experiments.cambridge_input_matrix.projective_association import build_projective_association_graph
from experiments.cambridge_input_matrix.projective_reconstruction import reconstruct_projective_anchors
from experiments.cambridge_input_matrix.projective_completion import build_projective_completion

torch.set_num_threads(1)

def synthetic(mode="pure_gs", count=8):
    K = torch.tensor([[700., 0, 320.], [0, 710., 240.], [0, 0, 1.]])
    xyz = torch.stack([torch.tensor([-.603 + .21 * (i % 4), -.173 + .25 * (i // 4), 5.123]) for i in range(count)])
    features = torch.eye(256)[:count]
    records = {}
    for i, center in enumerate(([-1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.])):
        pose = torch.eye(4); pose[:3, 3] = -torch.tensor(center)
        camera = xyz + pose[:3, 3]
        h = camera @ K.T
        r = {"native_keypoints": h[:,:2]/h[:,2:3]-.5,
             "native_descriptors": features.clone(), "native_scores": torch.linspace(.95,.8,count),
             "native_K": K.clone(), "pose_w2c": pose, "native_input_hw": torch.tensor([480,640]),
             "row_source": torch.full((count,), 0 if mode=="raw_only" else 1, dtype=torch.long),
             "source_row": torch.arange(count)}
        if mode != "raw_only":
            r.update(v2_valid=torch.ones(count,dtype=torch.bool),
                     native_valid_keypoint_mask=torch.ones(count,dtype=torch.bool),
                     native_depth_at_keypoints=camera[:,2].clone(),
                     native_alpha_at_keypoints=torch.ones(count))
        records["mapping_"+str(i)+".png"] = r
    scene = {"mapping_names": list(records), "test_names": ["test.png"]}
    return payload(mode,records,scene,"synthetic"), xyz

def same_tensor_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,dict):
        for k,v in a.items():
            if k in b:
                same_tensor_tree(v,b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_tensor_tree(x,y)
    elif isinstance(a,(int,float,bool)) and not isinstance(a,str):
        # Timing diagnostics are never part of numerical output comparisons.
        if isinstance(a,float) and np.isnan(a):assert np.isnan(b)
        else:assert a==b

def test_frozen_design_has_all_controls():
    assert DESIGN["seed"]==2026 and DESIGN["observation_budget"]==2048
    assert DESIGN["hybrid_quota"]=={"real":1024,"render":1024}
    assert DESIGN["query_semantics"] is True
    assert len(DESIGN["scene_order"])==5
    check_kernel_adaptations()

def test_original_source_free_contract_is_unchanged():
    raw,_=synthetic("raw_only")
    with pytest.raises(ValueError,match="uses_source_mapping_rgb=false"):
        GaussianRenderObservationProvider(raw)
    from anygsloc.mapping.contracts import require_mapping_only
    with pytest.raises(ValueError,match="source-mapping-RGB-free"):
        require_mapping_only(raw,label="raw")

def test_test_image_and_unknown_image_rejected_before_io():
    class Dataset:
        root=Path("/dataset")
        def load_image(self,camera):
            raise AssertionError("must not reach RGB loader")
    scene={"mapping_names":["map.png"],"test_names":["test.png"]}
    for name in ("test.png","unknown.png"):
        with pytest.raises(ValueError,match="refusing"):
            safe_mapping_image(Dataset(),SimpleNamespace(image_name=name,image_path=Path("/dataset/processed")/name),scene)
    bad,_=synthetic()
    bad["test_names"]=[bad["mapping_names"][0]]
    with pytest.raises(ValueError,match="test camera"):validate_payload(bad)

def test_duplicate_pose_modalities_are_not_independent_views():
    p,_=synthetic()
    p["queries"]["another_modality.png"]=copy.deepcopy(next(iter(p["queries"].values())))
    p["mapping_names"].append("another_modality.png")
    with pytest.raises(ValueError,match="duplicate physical"):validate_payload(p)
    with pytest.raises(ValueError,match="repeated physical"):
        assert_unique_track_cameras({"observation_offsets":[0,3],"query_indices":[0,0,1]})

def test_hybrid_fixed_quota_precedes_v2_and_keeps_full_available_count():
    p,_=synthetic()
    base=next(iter(p["queries"].values()))
    def many(source,n):
        r={k:v.clone() for k,v in base.items()}
        for k in ("native_keypoints","native_descriptors","native_scores","row_source","source_row",
                  "v2_valid","native_valid_keypoint_mask","native_depth_at_keypoints","native_alpha_at_keypoints"):
            r[k]=r[k].repeat((n//len(base["native_keypoints"])+1,)+(1,)*(r[k].ndim-1))[:n]
        r["row_source"].fill_(source)
        r["native_scores"]=torch.linspace(1.,0.,n)
        r["v2_valid"][500:]=False
        return r
    r=merge_camera(many(0,1800),many(1,1700))
    assert len(r["native_keypoints"])==2048
    assert (r["row_source"]==0).sum()==1024 and (r["row_source"]==1).sum()==1024
    assert r["available_real"]+r["available_render"]==3500
    assert int(r["v2_valid"].sum())==1000  # no post-V2 refill
    short=merge_camera(many(0,20),many(1,1700))
    assert len(short["native_keypoints"])==1044

def test_raw_any_does_not_access_completion_or_depth(monkeypatch):
    import experiments.cambridge_input_matrix.driver as d
    raw,xyz=synthetic("raw_only")
    def forbidden(*a,**k):raise AssertionError("Gaussian completion entered raw-only")
    monkeypatch.setattr(d,"build_projective_completion",forbidden)
    result,features,info=construct_any(raw,device="cpu",workers=1)
    assert len(result)==len(xyz) and np.allclose(result,xyz.numpy(),atol=1e-5)
    assert info["v2_enabled"] is False and info["completion_reason"]=="disabled_without_Gaussian"
    assert features.shape==(len(xyz),256)

def test_classic_is_invariant_to_gaussian_evidence():
    p,_=synthetic()
    a=construct_classic(p,device="cpu",workers=1)
    changed=copy.deepcopy(p)
    for r in changed["queries"].values():
        r["native_depth_at_keypoints"].fill_(float("nan"))
        r["native_alpha_at_keypoints"].zero_()
        r["v2_valid"].zero_()
        r["native_depth"]=torch.full((480,640),float("inf"))
    b=construct_classic(changed,device="cpu",workers=1)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert a[2]["uses_depth"] is False

def test_pure_gs_association_and_reconstruction_match_original_numerically():
    p,_=synthetic()
    old=GaussianRenderObservationProvider(p)
    new=MatrixObservationProvider(p)
    a=original_association(old,device="cpu")
    b=build_projective_association_graph(new,device="cpu")
    for k in ("tracks","component_statistics","query_bins","pair_sidecar"):
        same_tensor_tree(a[k],b[k])
    x=original_reconstruction(old,a,parallel_workers=1)
    y=reconstruct_projective_anchors(new,b,parallel_workers=1)
    for k in ("anchor_xyz","anchor_features","anchor_position_covariance",
              "identity_reliability","geometry_reliability","projective_anchor_observations"):
        same_tensor_tree(x[k],y[k])

def test_pure_gs_completion_matches_original_numerically():
    p,_=synthetic()
    old,new=GaussianRenderObservationProvider(p),MatrixObservationProvider(p)
    a=original_association(old,device="cpu")
    b=build_projective_association_graph(new,device="cpu")
    # Empty base support makes the same actual unused observations eligible.
    for graph in (a,b):
        graph["tracks"]["query_index"]=torch.empty(0,dtype=torch.long)
        graph["tracks"]["keypoint_index"]=torch.empty(0,dtype=torch.long)
    opts=dict(voxel_size_m=.05,alpha_minimum=.05,minimum_similarity=.7,
              minimum_camera_families=1,device="cpu")
    x=original_completion(old,a,**opts)
    y=build_projective_completion(new,b,**opts)
    for k in ("anchor_xyz","anchor_features","anchor_position_covariance",
              "identity_reliability","geometry_reliability","projective_anchor_observations"):
        same_tensor_tree(x[k],y[k])

def test_common_K_is_global_and_deterministic():
    assert common_rows([100,80,70,60,50,40],90)==40
    assert common_rows([100]*6,32)==32
    with pytest.raises(ValueError):common_rows([100]*5,32)
    with pytest.raises(ValueError):common_rows([100]*5+[0],32)
    assert not torch.cuda.is_initialized()

def test_nondefault_cuda_device_is_made_current_without_allocating(monkeypatch):
    import experiments.cambridge_input_matrix.driver as d
    from contextlib import contextmanager
    events=[]
    @contextmanager
    def context(device):
        events.append(("enter",str(device)));yield;events.append(("exit",str(device)))
    monkeypatch.setattr(torch.cuda,"device",context)
    monkeypatch.setattr(d,"_extract",lambda root,scene,device:events.append(("extract",str(device))))
    d.extract("unused","ShopFacade","cuda:1")
    assert events==[("enter","cuda:1"),("extract","cuda:1"),("exit","cuda:1")]
    with pytest.raises(ValueError,match="CUDA"):d.extract("unused","ShopFacade","cpu")
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("field",["rgb_sha","gt","K","mask","mapping"])
def test_query_snapshot_rejects_changed_inputs(tmp_path,field):
    from experiments.cambridge_input_matrix.driver import freeze_evaluation_snapshot
    original={"rgb_sha":"abc","gt":[0,1,2],"K":[700,700],"mask":"m","mapping":["camera"]}
    path=tmp_path/"query_inputs.json"
    first=freeze_evaluation_snapshot(path,original)
    assert first==freeze_evaluation_snapshot(path,original)
    changed=copy.deepcopy(original);changed[field]="changed"
    with pytest.raises(ValueError,match="snapshot changed"):
        freeze_evaluation_snapshot(path,changed)

def test_prior_full_original_track_leakage_and_xyz_drift_rejected():
    from experiments.cambridge_input_matrix.prior_provenance import audit_mapping_only_initialization
    xyz=np.array([1.,2.,3.])
    original={1:(xyz,np.array([1,2])),2:(xyz,np.array([1,99]))}
    prepared={1:(xyz,np.array([1,2]))}
    assert audit_mapping_only_initialization(original,prepared,{1,2})["original_track_contains_nonmapping"]==0
    with pytest.raises(ValueError,match="mapping-only"):
        audit_mapping_only_initialization(original,{2:(xyz,np.array([1]))},{1,2})
    with pytest.raises(ValueError,match="mapping-only"):
        audit_mapping_only_initialization(original,{1:(xyz+1,np.array([1,2]))},{1,2})

def test_all_six_materialized_maps_receive_global_K(tmp_path):
    import experiments.cambridge_input_matrix.driver as d
    from anygsloc.artifacts import sha256
    root=tmp_path/"matrix"
    p={"design":d.DESIGN,"code_sha256":d.fingerprints(),"scenes":{"ShopFacade":{"reference_K":7}}}
    d.write_new(root/"protocol.json",p)
    n=10
    for mode in d.MODES:
        for method in d.METHODS:
            folder=root/"maps/ShopFacade"/mode/method
            xyz=np.arange(n*3,dtype=np.float32).reshape(n,3)
            desc=np.zeros((n,256),dtype=np.float32);desc[:,0]=1
            provenance={"protocol_sha256":sha256(root/"protocol.json"),"uses_test_queries":False}
            full=d.save_map(folder/"full/deployment_map.pt",xyz,desc,np.arange(n),provenance)
            d.write_new(folder/"summary.json",{"complete":True,"full":full})
            n-=1
    d.select(root,"ShopFacade")
    selected=d.read(root/"selection/ShopFacade.json")
    assert selected["common_K"]==5
    assert {v["count"] for v in selected["maps"].values()}=={5}

def test_current_mask_priority_is_resolved_not_assumed(tmp_path):
    from experiments.cambridge_input_matrix.driver import effective_mask_path
    root=tmp_path/"dataset";root.mkdir()
    fallback=root/"masks.pkl";fallback.write_bytes(b"old")
    assert effective_mask_path(root)==fallback.resolve()
    (root/"processed").mkdir()
    preferred=root/"processed/masks.pkl";preferred.write_bytes(b"new")
    assert effective_mask_path(root)==preferred.resolve()
    assert effective_mask_path(root)!=fallback.resolve()

def test_unlisted_trainer_camera_and_duplicate_name_rejected():
    from experiments.cambridge_input_matrix.prior_provenance import audit_prepared_camera_registry
    entries={"map.png":("map__flat.png","sha")}
    extra={1:SimpleNamespace(name="map__flat.png"),2:SimpleNamespace(name="test__flat.png")}
    with pytest.raises(ValueError,match="trainer sparse image registry"):
        audit_prepared_camera_registry(extra,entries,{}, {})
    duplicate={1:SimpleNamespace(name="map__flat.png"),2:SimpleNamespace(name="map__flat.png")}
    with pytest.raises(ValueError,match="trainer sparse image registry"):
        audit_prepared_camera_registry(duplicate,entries,{}, {})

def test_point_text_reader_keeps_id_and_optional_track(tmp_path):
    from experiments.cambridge_input_matrix.prior_provenance import point_tracks
    p=tmp_path/"points3D.txt";p.write_text("# COLMAP\n12 1.1 2.2 3.3 10 20 30 1.0\n")
    rows=point_tracks(p)
    assert list(rows)==[12] and np.allclose(rows[12][0],[1.1,2.2,3.3])
    assert len(rows[12][1])==0


def test_query_pose_metadata_cannot_change_mapping_numerics(tmp_path):
    """The shared loader parses test metadata, but none reaches map numerics."""
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.mapping.virtual_camera_registry import build_virtual_camera_registry, camera_intrinsic
    from experiments.cambridge_input_matrix.driver import spatial_rows
    root=tmp_path/"scene";sparse=root/"sparse/0";sparse.mkdir(parents=True)
    (sparse/"cameras.txt").write_text("1 PINHOLE 640 480 700 710 320 240"+chr(10))
    base,_=synthetic("pure_gs")
    def prepared(mode, changed_test):
        lines=[]
        for i,(name,r) in enumerate(base["queries"].items(),1):
            t=r["pose_w2c"][:3,3].tolist()
            lines.extend([f"{i} 1 0 0 0 {t[0]} {t[1]} {t[2]} 1 {name}", ""])
        test_pose="0.7071067811865476 0 0 0.7071067811865476 999 -888 777" if changed_test else "1 0 0 0 0 0 0"
        lines.extend(["5 "+test_pose+" 1 test.png", ""])
        (sparse/"images.txt").write_text(chr(10).join(lines)+chr(10))
        (root/"dataset_test.txt").write_text("test.png "+test_pose+chr(10))
        dataset=ColmapDataset(root,load_masks=False)
        cameras,registry=build_virtual_camera_registry(dataset.split("mapping"))
        source,_=synthetic(mode)
        records={}
        for camera in cameras:
            record=copy.deepcopy(source["queries"][camera.image_name])
            record["pose_w2c"]=torch.from_numpy(camera.pose_w2c)
            record["native_K"]=camera_intrinsic(camera)
            if mode=="hybrid":record["row_source"][::2]=0
            records[camera.image_name]=record
        scene={"mapping_names":[c.image_name for c in cameras],"test_names":["test.png"]}
        return payload(mode,records,scene,"synthetic"),registry,dataset.camera("test.png").pose_w2c
    for mode in ("pure_gs","raw_only","hybrid"):
        before,registry_a,test_a=prepared(mode,False)
        after,registry_b,test_b=prepared(mode,True)
        assert not np.array_equal(test_a,test_b)  # Loader really parsed changed test geometry.
        assert registry_a==registry_b and before["mapping_names"]==after["mapping_names"]
        same_tensor_tree(before["queries"],after["queries"])
        assert "test.png" not in before["queries"] and not (root/"processed/test.png").exists()
        for constructor in (construct_any,construct_classic):
            a=constructor(before,device="cpu",workers=1)
            b=constructor(after,device="cpu",workers=1)
            assert len(a[0])>0 and np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
            assert np.array_equal(spatial_rows(a[0],4),spatial_rows(b[0],4))

def test_prior_audit_reuse_rechecks_evidence_and_mapping_rgb(tmp_path):
    import json
    from anygsloc.artifacts import sha256
    from experiments.cambridge_input_matrix.prior_provenance import revalidate_audit_record
    root=tmp_path/"data";(root/"processed").mkdir(parents=True)
    rgb=root/"processed/map.png";rgb.write_bytes(b"mapping")
    camera=SimpleNamespace(image_name="map.png",image_path=rgb,pose_w2c=np.eye(4),
        width=100,height=100,fov_x=np.pi/2,fov_y=np.pi/2)
    source=tmp_path/"source";sparse=source/"sparse/0";sparse.mkdir(parents=True)
    im=sparse/"images.txt";im.write_text("1 1 0 0 0 0 0 0 1 flat.png" + chr(10)*2)
    cam=sparse/"cameras.txt";cam.write_text("1 PINHOLE 100 100 50 50 50 50" + chr(10))
    data=SimpleNamespace(root=root,split=lambda key:[camera] if key=="mapping" else [SimpleNamespace(image_name="test.png")])
    native=tmp_path/"native";ply=native/"point_cloud/iteration_30000/point_cloud.ply"
    producer=tmp_path/"producer.json"
    producer.write_text(json.dumps({"prior":{"ply":str(ply),"ply_sha256":"bound"}}))
    inputs=tmp_path/"inputs.json"
    inputs.write_text(json.dumps({"test_used":False,"images":[{"source_name":"map.png","prepared_name":"flat.png","sha256":sha256(rgb)}]}))
    audit={"training_job_id":"ShopFacade_vanilla_3dgs_reconstruct","native_path":str(native),"source_path":str(source),
        "producer_manifest":str(producer),"input_manifest":str(inputs),
        "evidence_sha256":{str(p):sha256(p) for p in [producer,inputs,im,cam]},
        "complete_original_track_audit":{"prepared_points":1,"missing_original_id":0,"changed_xyz":0,
          "original_track_contains_nonmapping":0,"short_track":0,"initial_ply_maximum_coordinate_difference":0},
        "mapping_image_count":1,"prepared_camera_audit":{"trainer_camera_count":1,"exact_mapping_names":True}}
    record=tmp_path/"audit.json";record.write_text(json.dumps(audit))
    prior={"ply":str(ply),"ply_sha256":"bound"}
    reused,hashes=revalidate_audit_record("ShopFacade",data,prior,record)
    assert hashes=={"map.png":sha256(rgb)}
    assert reused["reused_audit_record"]["evidence_revalidated"] is True
    camera.fov_x=1.0
    with pytest.raises(ValueError,match="pose/intrinsics differ"):
        revalidate_audit_record("ShopFacade",data,prior,record)
    camera.fov_x=np.pi/2;camera.pose_w2c[0,3]=1
    with pytest.raises(ValueError,match="pose/intrinsics differ"):
        revalidate_audit_record("ShopFacade",data,prior,record)
    camera.pose_w2c[0,3]=0
    rgb.write_bytes(b"changed")
    with pytest.raises(ValueError,match="mapping RGB differs"):
        revalidate_audit_record("ShopFacade",data,prior,record)
    rgb.write_bytes(b"mapping");inputs.write_text("{}")
    with pytest.raises(ValueError,match="evidence changed"):
        revalidate_audit_record("ShopFacade",data,prior,record)




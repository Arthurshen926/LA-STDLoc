"""CPU-only release and process-limit contracts."""
import copy
import resource
import pytest
from experiments.native_baseline_repair.launch_ulf import nofile_pair, check_lease, check_review

def lease():
    return {"approved":True,"scope":"ulf_stmary_native_init_training",
            "protocol_sha256":"abc","mapping_semantics":["off","on"],
            "approval_reference":"parent","owner_handover":"owner","gpu_uuid":"GPU-test",
            "valid_from_unix":10,"valid_until_unix":100}

def test_nofile_preserves_hard():
    assert nofile_pair(1024,1048576)==(4096,1048576)
    assert nofile_pair(8192,1048576)==(8192,1048576)
    assert nofile_pair(resource.RLIM_INFINITY,resource.RLIM_INFINITY)==(resource.RLIM_INFINITY,resource.RLIM_INFINITY)
    assert nofile_pair(1024,resource.RLIM_INFINITY)==(4096,resource.RLIM_INFINITY)

def test_nofile_fails_closed():
    with pytest.raises(ValueError): nofile_pair(1024,2048)

def test_valid_lease():
    assert check_lease(lease(),"abc","off",50)=="GPU-test"

@pytest.mark.parametrize("key,value",[("approved",False),("scope","other"),("protocol_sha256","wrong"),("mapping_semantics",[]),("owner_handover",None),("approval_reference",None),("gpu_uuid","1"),("valid_until_unix",9)])
def test_invalid_lease(key,value):
    x=lease();x[key]=value
    with pytest.raises(ValueError):check_lease(x,"abc","off",50)

def test_expired_lease():
    with pytest.raises(ValueError):check_lease(lease(),"abc","off",101)

def test_long_lease():
    x=lease();x["valid_until_unix"]=10000
    with pytest.raises(ValueError):check_lease(x,"abc","off",50)

def test_lease_must_cover_full_runner_window():
    x=lease();x["valid_until_unix"]=4000
    assert check_lease(x,"abc","off",50,min_remaining_seconds=3460)=="GPU-test"
    with pytest.raises(ValueError):check_lease(x,"abc","off",1000,min_remaining_seconds=3460)

def test_review_binding():
    x={"complete":True,"verdict":"PASS","protocol_sha256":"abc","approved_methods":["ULFLoc"]}
    check_review(x,"abc")
    for key,value in [("complete",False),("verdict","PENDING"),("protocol_sha256","wrong"),("approved_methods",["STDLoc","ULFLoc"])]:
        y=copy.deepcopy(x);y[key]=value
        with pytest.raises(ValueError):check_review(y,"abc")

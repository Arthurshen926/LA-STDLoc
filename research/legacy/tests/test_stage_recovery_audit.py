import json
import pytest
from common.hashing import sha256_file
from scripts.audit_stage_recovery import audit


def fixture(tmp_path):
    snap=tmp_path/'code';(snap/'data').mkdir(parents=True)
    module=snap/'data/loader.py';module.write_text('original')
    input_file=tmp_path/'input';input_file.write_text('pixels')
    command=tmp_path/'command.json';command.write_text(json.dumps({'input_contract':{
        'code':{'data/loader.py':sha256_file(module),'scripts/execute_stdloc_cleanup_dry_run_v2.py':'old'},
        'inputs':{str(input_file):sha256_file(input_file)}}}))
    return snap,module,input_file,command


def test_unrelated_cleanup_does_not_invalidate_recovery(tmp_path):
    snap,_,_,command=fixture(tmp_path)
    assert audit(command,snap)['all_recorded_inputs_match']


@pytest.mark.parametrize('target',['code','input'])
def test_actual_dependency_change_still_rejected(tmp_path,target):
    snap,module,inp,command=fixture(tmp_path)
    (module if target=='code' else inp).write_text('changed')
    with pytest.raises(ValueError,match='dependencies changed'):audit(command,snap)

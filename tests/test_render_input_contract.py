import copy
import pytest
from evidence.render_input_contract import validate_render_input


def test_render_audit_cannot_change_prior_or_render_configuration():
    cache={'input':{'gaussian_ply_sha256':'a'*64},'configuration':{'gaussian_type':'2dgs','sh_degree':3,'background':'black'}}
    args=dict(prior_sha256='a'*64,gaussian_type='2dgs',sh_degree=3,background='black')
    assert validate_render_input(cache,**args)
    for key,value in [('prior_sha256','b'*64),('gaussian_type','3dgs'),('sh_degree',0),('background','white')]:
        with pytest.raises(ValueError):validate_render_input(cache,**{**args,key:value})
    legacy=copy.deepcopy(cache);legacy.pop('input')
    with pytest.raises(ValueError,match='lacks'):validate_render_input(legacy,**args)
    assert not validate_render_input(legacy,**args,allow_legacy=True)

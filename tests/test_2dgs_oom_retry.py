import pytest
import torch
import priors.rendering as rendering


def test_inference_oom_retries_only_tile_size(monkeypatch):
    calls=[]
    def raster(**kw):
        calls.append(kw)
        if len(calls)==1:raise torch.cuda.OutOfMemoryError('test')
        return 'ok'
    monkeypatch.setattr(rendering, 'rasterization_2dgs', raster)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda:None)
    with torch.inference_mode(), pytest.warns(RuntimeWarning):
        assert rendering._rasterization_2dgs_inference_retry(width=1920,height=1080)=='ok'
    assert calls==[{'width':1920,'height':1080},{'width':1920,'height':1080,'tile_size':32}]


@pytest.mark.parametrize('training,tile', [(True,16),(False,32)])
def test_no_unbounded_or_training_retry(monkeypatch,training,tile):
    def raster(**kw):raise torch.cuda.OutOfMemoryError('test')
    monkeypatch.setattr(rendering,'rasterization_2dgs',raster)
    with torch.set_grad_enabled(training), pytest.raises(torch.cuda.OutOfMemoryError):
        rendering._rasterization_2dgs_inference_retry(tile_size=tile)

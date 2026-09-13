from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from data.datasets import validate_centered_camera, _focal_to_fov, ColmapDataset


@pytest.mark.parametrize('model,params', [('PINHOLE',[500,501,320,240]),('SIMPLE_RADIAL',[500,320,240,.04]),('OPENCV',[500,501,320,240,.01,0,0,0])])
def test_centered_camera_accepted(model,params):
    validate_centered_camera(SimpleNamespace(model=model,params=params,width=640,height=480))


def test_principal_point_cannot_silently_disappear():
    with pytest.raises(ValueError,match='non-centered'):
        validate_centered_camera(SimpleNamespace(model='PINHOLE',params=[500,501,319.5,240],width=640,height=480))


@pytest.mark.parametrize('focal', [0,-1,float('nan'),float('inf')])
def test_invalid_focal_rejected(focal):
    with pytest.raises(ValueError): _focal_to_fov(focal,640)


def test_palette_pixels_are_decoded_as_rgb(tmp_path):
    image = Image.fromarray(np.ones((4,4),dtype=np.uint8),'P')
    image.putpalette([0,0,0, 255,20,30] + [0]*762)
    path = tmp_path/'palette.png'; image.save(path)
    camera = SimpleNamespace(image_path=path,height=4,width=4)
    actual = ColmapDataset.load_image(camera)
    assert actual.shape == (3,4,4)
    assert torch.allclose(actual[:,0,0],torch.tensor([255.,20.,30.])/255)


def test_rgb_numeric_path_is_bit_exact(tmp_path):
    pixels = np.random.default_rng(17).integers(0,256,(4,8,3),dtype=np.uint8)
    path = tmp_path/'rgb.png'; Image.fromarray(pixels).save(path)
    result = ColmapDataset.load_image(SimpleNamespace(image_path=path,height=4,width=8))
    expected = torch.from_numpy(pixels.astype(np.float32)/255).permute(2,0,1)
    assert torch.equal(result,expected)


@pytest.mark.parametrize('split,pattern',[('missing.png\n','unregistered'),('known.png\nknown.png\n','duplicate')])
def test_test_registry_errors_are_not_silently_ignored(tmp_path,split,pattern):
    sparse=tmp_path/'sparse/0';sparse.mkdir(parents=True)
    (sparse/'cameras.txt').write_text('1 PINHOLE 8 4 10 10 4 2\n')
    (sparse/'images.txt').write_text('1 1 0 0 0 0 0 0 1 known.png\n\n')
    (sparse/'list_test.txt').write_text(split)
    with pytest.raises(ValueError,match=pattern): ColmapDataset(tmp_path)


def test_official_cambridge_header_is_not_a_query(tmp_path):
    sparse=tmp_path/'sparse/0';sparse.mkdir(parents=True)
    (sparse/'cameras.txt').write_text('1 PINHOLE 8 4 10 10 4 2\n')
    (sparse/'images.txt').write_text('1 1 0 0 0 0 0 0 1 known.png\n\n')
    (tmp_path/'dataset_test.txt').write_text('Visual Landmark Dataset\nImageFile, Camera Position\nknown.png 0 0 0 1 0 0 0\n')
    assert [c.image_name for c in ColmapDataset(tmp_path).split('test')] == ['known.png']

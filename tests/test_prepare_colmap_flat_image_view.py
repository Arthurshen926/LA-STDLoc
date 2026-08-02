import os

from scripts.prepare_cambridge_mapping_only_colmap import (
    read_images_binary,
    write_images_binary,
)
from scripts.prepare_colmap_flat_image_view import stage_flat_image_view

from tests.test_prepare_cambridge_mapping_only_colmap import _image


def test_flat_image_view_rewrites_names_without_copying_pixels(tmp_path):
    source = tmp_path / "mapping"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = {
        1: _image(1, "seq1/frame.png", []),
        2: _image(2, "seq2/frame.png", []),
    }
    write_images_binary(images, sparse / "images.bin")
    for filename in ("cameras.bin", "points3D.bin", "points3D.ply"):
        (sparse / filename).write_bytes(filename.encode())
    for image in images.values():
        path = source / "images" / image.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image.name.encode())
    (source / "mapping_only_manifest.json").write_text(
        '{"semantic_mask_used": false, "undistortion_used": true}\n'
    )

    output = tmp_path / "flat"
    report = stage_flat_image_view(source=source, output=output)

    rewritten = read_images_binary(output / "sparse" / "0" / "images.bin")
    assert {image.name for image in rewritten.values()} == {
        "seq1__frame.png",
        "seq2__frame.png",
    }
    for image in rewritten.values():
        path = output / "images" / image.name
        assert path.is_symlink()
        assert os.path.samefile(path, source / "images" / image.name.replace("__", "/"))
    assert report["image_pixels_copied"] is False
    assert report["image_pixels_changed"] is False

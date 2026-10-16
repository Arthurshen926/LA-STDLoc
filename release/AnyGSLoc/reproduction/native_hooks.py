"""Native query-mask and timing adapters; localizer class bodies are unchanged."""

import ast

from anygsloc.pipeline.benchmark import Recorder, plain


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source anchor changed: " + old[:70])
    return text.replace(old, new)


def mask_rgb(name, image, masks, query_names):
    """Same object/sky/geometry AND as native ULF query preprocessing."""
    import torch
    import torch.nn.functional as F

    if name not in query_names:
        raise ValueError("non-query image passed to semantic adapter")
    layers = masks[name]
    if len(layers) != 3:
        raise ValueError("three mask channels required")
    converted = []
    for value in layers:
        value = torch.as_tensor(value)
        if value.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        value = value.bool()
        if tuple(value.shape) != tuple(image.shape[-2:]):
            value = F.interpolate(
                value.float()[None, None], size=image.shape[-2:], mode="nearest"
            )[0, 0]
        converted.append(value.bool().to(image.device))
    valid = converted[0] & converted[1] & converted[2]
    return image * valid[None]


def std_text(original):
    text = once(
        original,
        '        query_image = camera_info.original_image.to("cuda")\n',
        '        query_image = camera_info.original_image.to("cuda")\n'
        "        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n",
    )
    text = once(
        text,
        '    print("Output path:", output_path)\n',
        '    output_path = _CONTROLLED_OUTPUT\n    print("Output path:", output_path)\n',
    )
    ast.parse(text)
    return text


def transform_native(original, method):
    text = std_text(original) if method == "STDLoc" else original
    text = once(
        text,
        "    test_cameras = scene.getTestCameras()\n",
        "    test_cameras = _MATCHED.cameras(scene.getTestCameras(), %r)\n" % method,
    )
    if method == "STDLoc":
        text = once(
            text,
            "        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n",
            "        _MATCHED.start_mask()\n"
            "        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n"
            "        _MATCHED.end_mask()\n",
        )
        call = "        loc_res = stdloc.localize(query_image, fovx, fovy)\n"
    elif method == "ULFLoc":
        text = once(
            text,
            "        if masks is not None:\n",
            "        _MATCHED.start_mask()\n        if masks is not None:\n",
        )
        text = once(
            text,
            "        fovx = camera_info.FoVx\n",
            "        _MATCHED.end_mask()\n        fovx = camera_info.FoVx\n",
        )
        call = (
            "        loc_res = ulfloc.localize(camera_info, query_image, fovx, fovy)\n"
        )
    else:
        raise ValueError("unknown native method")
    text = once(
        text,
        call,
        call + "        if _MATCHED.finish(camera_info.image_name, loc_res):\n"
        "            continue\n",
    )
    ast.parse(text)
    before, after = ast.parse(original), ast.parse(text)
    bclasses = [
        ast.dump(n)
        for n in before.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    ]
    aclasses = [
        ast.dump(n)
        for n in after.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    ]
    if bclasses != aclasses:
        raise ValueError("native scientific definitions changed")
    return text


def fresh_camera(dataset, index, decode, materialize):
    """Fresh PIL decode, then the exact native dataset __getitem__ and original index."""
    from PIL import Image

    infos = dataset.scene.scene_info.test_cameras
    old = infos[index]

    def open_decode():
        image = Image.open(old.image_path)
        try:
            image.load()
        except BaseException:
            image.close()
            raise
        return image

    fresh = decode(open_decode)
    try:
        infos[index] = old._replace(image=fresh)
        try:
            return materialize(lambda: dataset[index])
        finally:
            infos[index] = old
    finally:
        fresh.close()


def native_test_dataset(loader, method):
    """Validate original loader configuration without starting its iterator/workers."""
    from torch.utils.data import DataLoader, Dataset, SequentialSampler
    from torch.utils.data._utils.collate import default_convert

    workers = {"STDLoc": 4, "ULFLoc": 0}
    if method not in workers or type(loader) is not DataLoader:
        raise ValueError("requires original native DataLoader and known method")
    dataset = loader.dataset
    if (
        not isinstance(dataset, Dataset)
        or type(dataset).__name__ != "SceneDataset"
        or type(dataset).__module__ != "scene"
        or getattr(dataset, "split", None) != "test"
    ):
        raise ValueError("requires original native SceneDataset test split")
    if (
        loader.num_workers != workers[method]
        or loader.batch_size is not None
        or loader.batch_sampler is not None
        or loader.collate_fn is not default_convert
        or type(loader.sampler) is not SequentialSampler
        or loader.pin_memory is not True
        or loader.drop_last is not False
        or loader.persistent_workers is not False
    ):
        raise ValueError("native test loader configuration differs from frozen method")
    # The matched synchronous frame intentionally materializes dataset[index].
    # Constructing a loader does not launch workers. Never iterate it or prefetch.
    return dataset


class NativeRecorder(Recorder):
    def finish(self, name, result=None):
        warmup = super().finish(name)
        if result is not None:
            (self.warmup_rows if warmup else self.rows)[-1]["localization"] = (
                native_evidence(result)
            )
        return warmup

    def archive_load(self, loader, *args, **kwargs):
        begin = self.clock()
        result = loader(*args, **kwargs)
        self.archive_loads_ms.append((self.clock() - begin) * 1000)
        return result

    def cameras(self, loader, method):
        dataset = native_test_dataset(loader, method)
        infos = dataset.scene.scene_info.test_cameras
        indexes = {c.image_name: i for i, c in enumerate(infos)}
        if len(indexes) != len(infos):
            raise ValueError("duplicate native query metadata")
        # No camera/feature cache; call native materialization again for every frame.
        for warmup, order, name in self.schedule:
            if name not in indexes:
                raise ValueError("frozen query absent from native metadata")
            self.begin(name)
            camera = fresh_camera(
                dataset,
                indexes[name],
                lambda op: self.timed("rgb_decode_ms", op),
                lambda op: self.timed("camera_materialize_ms", op),
            )
            if camera.image_name != name:
                raise ValueError("native camera name mismatch")
            yield camera
            if self.active is not None:
                raise ValueError("native frame was not completed")


def native_evidence(result):
    sparse, dense = result["sparse"], result["dense"][-1]
    return dict(
        success=result.get("success"),
        sparse_pose_w2c=plain(sparse["pose_w2c"]),
        dense_pose_w2c=plain(dense["pose_w2c"]),
        sparse_inliers=plain(sparse.get("inliers")),
        dense_inliers=plain(dense.get("inliers")),
        matches=plain(result.get("matches")),
        errors=dict(
            sparse_rotation_deg=result.get("sparse_AE"),
            sparse_translation_cm=result.get("sparse_TE"),
            dense_rotation_deg=result.get("dense_AE"),
            dense_translation_cm=result.get("dense_TE"),
        ),
    )

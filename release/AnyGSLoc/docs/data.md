# Data and assets

`configs/*.json` contain portable paths. Resolution order is `--asset-root`, `ANYGSLOC_ASSETS`, then this repository's `assets/`. Each run has a separate `--output`. No server-specific path is needed in the code or distributed configurations.

```text
assets/
  weights/superpoint_v1.pth
  datasets/Cambridge/{ShopFacade,OldHospital,KingsCollege,GreatCourt,StMarysChurch}/
  datasets/7Scenes/{chess,fire,heads,office,pumpkin,redkitchen,stairs}/
  datasets/12Scenes/<scene>/
  priors/<prior>/<scene>.ply
  priors/indoor/<dataset>/<scene>.ply
  maps/<experiment-group>/<scene>.pt
  observations/<prior>/<scene>.pt       # optional acceleration, not needed to rerender
  native/{STDLoc,ULFLoc}/{on,off}/<scene>/
  third_party/{STDLoc,ULFLoc}/
```

A scene includes COLMAP `sparse/0/{cameras,images}.{bin,txt}`, the correctly rectified `processed/` images, and its test split. Cambridge uses `dataset_test.txt`; indoor scenes use `sparse/0/list_test.txt`. Mapping cameras are the complement of the test list. Indoor reference poses are SfM pseudo-ground truth. Never replace them with the original RGB-D poses while comparing to these paper rows.

The renderer/PnP interface assumes a centered principal point. Pixels, camera intrinsics, Gaussian world coordinates, and poses must share the same rectified coordinate system. The Cambridge profile records `opencv_same_k` explicitly. Merely renaming a directory to `processed` does not perform rectification. Use the prepared dataset assets and verify their metadata hashes for the exact paper protocol.

Cambridge query masks are stored in `processed/masks.pkl`. Mapping in AnyGSLoc does not apply semantic masks. Query evaluation uses `explicit_valid_mask`; indoor evaluation uses `none`. Mask archives are loaded onto CPU regardless of the device on which their tensors were saved. Segmentation-network inference is not included in the timing experiment.

Check the asset inventory:

```bash
python -m reproduction.assets check --root assets --kind deployment_map --hash
python -m reproduction.assets check --root assets --kind prior --hash
python -m reproduction.assets check --root assets --kind dataset_metadata --hash
```

The manifest records sizes and available SHA-256 bindings, including the exact current priors and deployment maps. A rebuilt map can have different serialization metadata and therefore a different file hash; numerical parity means equal IDs/order, coordinates, and descriptors, not equal archive bytes. Map storage in the paper uses the accepted serialized deployment inventory.

On a machine already holding the experimental assets, use a local bindings file of the form `{"assets": {"priors/vanilla_3dgs/ShopFacade.ply": "/existing/path/point_cloud.ply", ...}}`:

```bash
python -m reproduction.assets link --bindings /path/to/asset_bindings.json --root assets
python -m reproduction.assets copy --bindings /path/to/asset_bindings.json \
  --root /destination/maps --kind deployment_map
# Add --execute after reviewing the displayed size to materialize a copy.
```

The server bindings are stored beside the source candidate, under `../local/asset_bindings.json`, and excluded from the distributable archive. Dataset directories can contain other historical files; copying a whole dataset directory is an explicit operation. For publication, package only its required images, calibration, splits, and masks, along with upstream dataset attribution.

The distributed source contains result records and manifests. Large asset download URLs are not yet assigned. The manifest and copy utility define the required bundle contents; they do not imply that a public download already exists.

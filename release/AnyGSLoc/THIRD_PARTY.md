# Third-party material

The repository retains the existing project MIT license and its original copyright notice. Numerical code extracted from the project retains its provenance in `reproduction/source_lineage.json`.

- SuperPoint architecture/weights originate from the Magic Leap SuperPoint pretrained network implementation. The exact weight hash and upstream URL are in `anygsloc/features/superpoint.py`. Pretrained weights are supplied separately under their original terms.
- PyTorch, torchvision, NumPy, SciPy, OpenCV, PoseLib, gsplat, Pillow, plyfile, and Matplotlib are external dependencies, not relicensed by this project.
- STDLoc and ULF-Loc remain separate native source/model assets. Their retained source files and native configuration hashes are recorded in `reproduction/native_versions.json`; their original notices and licenses must travel with any distributed source bundle.
- Cambridge Landmarks, 7Scenes, 12Scenes, and the SfM pseudo-ground-truth data retain their upstream dataset terms. Dataset images are not inside the source archive.
- MAtCha, vanilla 2DGS/3DGS, i3DGS, and Long-LRM priors and any pretrained producer checkpoints remain separate assets with their upstream attribution.
- The paper figure SVGs include selected dataset imagery and should be distributed with the same dataset attribution. Literature table rows retain their citations.

This file records provenance and packaging boundaries; it does not grant additional rights over external assets.

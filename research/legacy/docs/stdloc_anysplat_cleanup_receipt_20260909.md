# AnySplat data cleanup receipt

- Scope: `/mnt/pool/sqy` only.
- Explicit exclusion: `/mnt/pool/sqy/G4Splat_runs` was not scanned for deletion and will not be modified.
- Repository adapters/configuration under `/root/STDLoc` are not modified.
- This receipt records the intended data cleanup; the post-cleanup verification is added after the filesystem operation.

Targets include all identified AnySplat output/model paths outside `G4Splat_runs`:

- AnyGSLoc paper outputs for `KingsCollege_anysplat_ff` and `OldHospital_anysplat_ff`.
- AnyGSLoc view-audit directories and AnySplat visual comparison images.
- Off-the-shelf-prior AnySplat trainer/evaluation trees, final PLY exports, failed/invalid-tail trees, setup files, logs, figures and AnySplat-only reports.
- Empty/symlink AnySplat entries under `Cambridge_stdloc/gaussian`.
- `third_party_lafgs_priors/AnySplat` under `/mnt/pool/sqy`.
- The standalone AnySplat render used by `lafgs_paper_figures_20260806`.

The goal is to remove reusable AnySplat data, not to remove incidental `anysplat` strings from source code or unrelated aggregate manifests.

## Post-cleanup verification

- Completed: 2026-09-09.
- 39 paths were preflighted; 38 were removed by the cleanup command and one empty probe directory had already been removed during the earlier safety check.
- Preflight size of the listed targets was approximately 282.628GB.
- A post-cleanup scan of the AnyGSLoc, off-the-shelf-prior, Cambridge, paper-figure and third-party roots found no remaining path whose name contains `anysplat` (case-insensitive).
- `/mnt/pool/sqy/G4Splat_runs` was not a target and remains untouched.

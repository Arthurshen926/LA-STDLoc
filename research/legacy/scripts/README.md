# Source-checkout mapping and historical research tools

Release online entrypoint: `python -m anygsloc.cli localize`.
Release compact selection: `python -m anygsloc.cli compact-map` (no evaluator).
Frozen Full mapping: `python -m scripts.run_anygsloc_scene --mapping-only ...`.

Other scripts are research/compatibility utilities, not alternative release
defaults. In particular queue preparation, feedback/metric training and sparse
refinement experiments are not part of Compact + Fast. Historical paths remain
stable for reproducibility. See `research/README.md` and `docs/release/`.

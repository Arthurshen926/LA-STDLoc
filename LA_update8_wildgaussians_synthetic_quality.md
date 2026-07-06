# LA_update8: WildGaussians Synthetic Appearance Quality

Date: 2026-06-27

## Scope

This update improves the synthetic RGB rendering path used by pseudo-query generation. The target was to avoid zero/no-target WildGaussians appearance for synthetic views when a checkpoint supports train appearance embeddings, while keeping no-appearance checkpoints safe.

## Code Changes

- Added `read_wildgaussians_config()` and `resolve_wildgaussians_appearance_mode()` in `la_artifacts/rgb_teacher.py`.
  - `auto` resolves to `record` only when `appearance_enabled: true` and all synthetic records carry appearance metadata.
  - `auto` resolves to `none` for `appearance_enabled: false` checkpoints, preventing accidental embedding lookup.
- Added `apply_wildgaussians_appearance_strategy()` in `la_artifacts/pseudo_query.py`.
  - Supports `blend`, `nearest`, `none`, `endpoint_a`, and `endpoint_b`.
  - `nearest` chooses the higher-weight interpolation endpoint and uses its train embedding with weight 1.0.
- Updated `scripts/build_pseudo_query_manifest.py`.
  - Default `--wildgaussians_appearance_mode` is now `auto`.
  - Added `--synthetic_appearance_strategy`.
  - Manifest records store both requested and resolved appearance mode.
- Updated `scripts/run_la_pseudo_query_pipeline.sh`.
  - Exposes `WILDGAUSSIANS_APPEARANCE_MODE=auto`.
- Exposes `SYNTHETIC_APPEARANCE_STRATEGY=nearest`.

## ShopFacade Smoke

Checkpoint:

`/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/ShopFacade_wg_app_nounc_sky_stopdens7k_15k_960/checkpoint-15000`

Settings:

- `synthetic_count=8`
- `seed=2026`
- render scale `0.5`
- same sampled poses for all rows

| Variant | Resolved appearance mode | Strategy | Accepted | Mean artifact score | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | `none` | blend metadata ignored | 8/8 | 0.591384 | 0.534088 | 0.651116 |
| `blend` | `record` | weighted adjacent train embeddings | 8/8 | 0.579097 | 0.495502 | 0.674077 |
| `nearest` | `record` | nearest endpoint train embedding | 8/8 | 0.577337 | 0.494267 | 0.656376 |

Visual check:

`/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/ShopFacade_none_blend_nearest_grid.png`

Interpretation:

- Appearance-conditioned render improves the synthetic-only artifact score relative to `none`.
- `nearest` is slightly better than `blend` on this smoke, but the margin is small.
- The dominant remaining defects are blur, road/sidewalk smearing, transient/occluder artifacts, and imperfect thin structures. Appearance conditioning helps but does not solve synthetic fidelity by itself.

## OldHospital Smoke

Checkpoint:

`/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/OldHospital_wg_noapp_nounc_nosky_stopdens7k_30k_960/checkpoint-30000`

Settings:

- `synthetic_count=4`
- `seed=2026`
- render scale `0.5`
- requested appearance mode `auto`

Result:

- `auto` resolved to `none`, as intended for `appearance_enabled: false`.
- Accepted synthetic: 4/4
- Mean artifact score: 0.428862
- Min/max: 0.260467 / 0.678653

Visual check:

`/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/OldHospital_auto_grid.png`

Interpretation:

- The OldHospital auto path no longer risks a record-embedding call against a no-appearance checkpoint.
- The renders are no longer black or globally broken, but still show blur, edge/branch artifacts, and large-view local distortion.

## Recommendation

Use `WILDGAUSSIANS_APPEARANCE_MODE=auto` and `SYNTHETIC_APPEARANCE_STRATEGY=nearest` as the defaults. For ShopFacade-style appearance-enabled checkpoints, nearest was marginally better and avoids interpolating potentially nonlinear appearance embeddings. For no-appearance checkpoints such as the current OldHospital fallback, `auto` correctly degrades to `none`.

The next quality improvement should not be more appearance wiring. It should be stricter synthetic QA and teacher-cache gating: reject synthetic views with high artifact score, strong low-detail regions, or teacher sparse/dense instability before student training.

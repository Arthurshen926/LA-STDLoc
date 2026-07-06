# LA-STDLoc Mainline Boundaries

## Status

The previous broad LA-STDLoc goal is paused. The validated control path is the
clean all-train RGB mainline from `LA_update29_clean_mainline_closure.md`.

Do not treat the paused broad goal as complete. The defensible current claim is
limited to sparse-only median pose and 5cm-level improvement from the clean
real-train control.

## Default Control

Use `scripts/run_la_clean_real_train_mainline.sh` for one scene and
`scripts/run_la_clean_control_matrix.sh` for the three validated controls.

The control path uses only real Cambridge train RGB pseudo queries and disables:

- synthetic RGB;
- teacher gates;
- pseudo-query selectors;
- no-reference valid/support masks;
- artifact detector or repair weighting;
- reliability weighting;
- direct depth checks.

This is the baseline for future ablations. Any new result should report against
this control before being compared with older mixed runs.

## Experimental Backend

Use `scripts/run_la_pseudo_query_pipeline.sh` only for ablations. It contains:

- MAtCha and WildGaussians synthetic rendering;
- spatial synthetic pose sampling;
- teacher cache valid masks;
- teacher gates;
- pseudo-query selection;
- artifact weights and region weights;
- reliability weighting.

None of these branches should be treated as default until they produce a
positive result against the clean control.

## Validated Controls

| Scene | Capacity | Steps | Seed | Output root |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 8192 | 2000 | 301 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` |
| OldHospital | 8192 | 2000 | 302 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` |
| OldHospital | 16384 | 2000 | 303 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_16384_2000_20260630` |

The corresponding official sparse-only summaries are recorded in
`LA_update29_clean_mainline_closure.md`.

## Next Smaller Objective

The next goal should be an OldHospital student-objective ablation against this
control, focused on high-precision recall and stability.

Synthetic RGB and artifact modules should stay disabled until the
student-objective ablation has a clear positive or negative result. The next
objective should answer one question at a time:

> Can a modified student objective improve OldHospital high-precision recall or
> reduce sparse-only instability without relying on synthetic RGB, teacher gates,
> or artifact masks?

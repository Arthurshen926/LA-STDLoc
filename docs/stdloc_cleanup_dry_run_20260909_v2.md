# STDLoc / LaFGS cleanup dry-run v2 (read-only)

Generated: `2026-09-09T17:57:31+08:00`

本报告只读取文件并生成清单；没有删除、移动或覆盖 `/mnt/pool/sqy` 下任何文件。`G4Splat_runs` 不在扫描范围内。

当前运行保护：`/mnt/pool/sqy/anygsloc_execution_queue_20260909` 仍有 AnyGSLoc scene/map/evaluation 任务，因此本报告只把它作为外部保护项，不对其做逐文件清理。

动作含义：`KEEP` 保留原位；`ARCHIVE` 只保留小型 provenance/结果或待集中归档；`DELETE_CANDIDATE` 表示可重建、但仍需用户最终批准后才删除。

## Root summary

| root | allocated (`du`) | files | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|---:|
| `/mnt/pool/sqy/anygsloc_paper_experiments_20260902` | 330.67 GiB | 835 | 459 files / 0.18 GiB | 216 files / 0.59 GiB | 160 files / 329.91 GiB |
| `/mnt/pool/sqy/stdloc_lafgs_legacy_archive_20260801` | 124.92 GiB | 9285 | 0 files / 0.00 GiB | 8966 files / 5.24 GiB | 319 files / 119.66 GiB |
| `/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725` | 93.33 GiB | 1411 | 28 files / 0.80 GiB | 912 files / 0.72 GiB | 471 files / 91.90 GiB |
| `/mnt/pool/sqy/lafgs_v6_full_mapping_accuracy_20260823` | 82.72 GiB | 44 | 18 files / 0.01 GiB | 2 files / 0.00 GiB | 24 files / 82.70 GiB |
| `/mnt/pool/sqy/stdloc_lafgs_prior_robust_a1_20260801` | 80.13 GiB | 231 | 86 files / 0.06 GiB | 120 files / 0.39 GiB | 25 files / 79.68 GiB |
| `/mnt/pool/sqy/stdloc_lafgs_clean_rebuild_20260728_v1` | 78.53 GiB | 2300 | 3 files / 0.00 GiB | 1635 files / 1.18 GiB | 662 files / 77.53 GiB |
| `/mnt/pool/sqy/stdloc_lafgs_phase2_archive_20260801` | 62.68 GiB | 799 | 0 files / 0.00 GiB | 645 files / 0.10 GiB | 154 files / 62.58 GiB |
| `/mnt/pool/sqy/lafgs_rgb_prior_cache` | 57.44 GiB | 2 | 0 files / 0.00 GiB | 0 files / 0.00 GiB | 2 files / 57.44 GiB |
| `/mnt/pool/sqy/lafgs_query_cache` | 15.58 GiB | 1 | 0 files / 0.00 GiB | 0 files / 0.00 GiB | 1 files / 15.58 GiB |
| `/mnt/pool/sqy/lafgs_v6_feedback_core_20260824` | 23.26 GiB | 73 | 0 files / 0.00 GiB | 30 files / 0.00 GiB | 43 files / 23.25 GiB |
| `/mnt/pool/sqy/lafgs_v6_feedback_control_20260824` | 24.83 GiB | 44 | 0 files / 0.00 GiB | 20 files / 0.00 GiB | 24 files / 24.82 GiB |
| `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825` | 25.35 GiB | 52 | 0 files / 0.00 GiB | 28 files / 0.23 GiB | 24 files / 25.11 GiB |

## Per-root interpretation

### `/mnt/pool/sqy/anygsloc_paper_experiments_20260902`

- Scope: 当前 AnyGSLoc 论文主线/消融输出；不整目录删除
- Allocated size: **330.67 GiB**; apparent file size: **313.21 GiB**; files: **835**.
- `KEEP`: 459 files, 0.18 GiB allocated.
- `ARCHIVE`: 216 files, 0.59 GiB allocated.
- `DELETE_CANDIDATE`: 160 files, 329.91 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=anygsloc_paper`.

### `/mnt/pool/sqy/stdloc_lafgs_legacy_archive_20260801`

- Scope: 旧 LaFGS final-state 模型库
- Allocated size: **124.92 GiB**; apparent file size: **117.09 GiB**; files: **9285**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 8966 files, 5.24 GiB allocated.
- `DELETE_CANDIDATE`: 319 files, 119.66 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=legacy_archive`.

### `/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725`

- Scope: OldHospital RGB prior 清洗/sky-no-sky 历史实验
- Allocated size: **93.33 GiB**; apparent file size: **87.78 GiB**; files: **1411**.
- `KEEP`: 28 files, 0.80 GiB allocated.
- `ARCHIVE`: 912 files, 0.72 GiB allocated.
- `DELETE_CANDIDATE`: 471 files, 91.90 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=rgb_sanitization`.

### `/mnt/pool/sqy/lafgs_v6_full_mapping_accuracy_20260823`

- Scope: v6 mapping accuracy 与 invalid closed-loop 诊断
- Allocated size: **82.72 GiB**; apparent file size: **77.35 GiB**; files: **44**.
- `KEEP`: 18 files, 0.01 GiB allocated.
- `ARCHIVE`: 2 files, 0.00 GiB allocated.
- `DELETE_CANDIDATE`: 24 files, 82.70 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=v6_mapping_accuracy`.

### `/mnt/pool/sqy/stdloc_lafgs_prior_robust_a1_20260801`

- Scope: OldHospital sky/no-sky prior robustness 历史实验
- Allocated size: **80.13 GiB**; apparent file size: **75.28 GiB**; files: **231**.
- `KEEP`: 86 files, 0.06 GiB allocated.
- `ARCHIVE`: 120 files, 0.39 GiB allocated.
- `DELETE_CANDIDATE`: 25 files, 79.68 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=prior_robust_a1`.

### `/mnt/pool/sqy/stdloc_lafgs_clean_rebuild_20260728_v1`

- Scope: OldHospital clean rebuild 与 selector/replay 历史实验
- Allocated size: **78.53 GiB**; apparent file size: **74.29 GiB**; files: **2300**.
- `KEEP`: 3 files, 0.00 GiB allocated.
- `ARCHIVE`: 1635 files, 1.18 GiB allocated.
- `DELETE_CANDIDATE`: 662 files, 77.53 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=clean_rebuild`.

### `/mnt/pool/sqy/stdloc_lafgs_phase2_archive_20260801`

- Scope: 早期 LaFGS phase-2 模型变体归档
- Allocated size: **62.68 GiB**; apparent file size: **62.62 GiB**; files: **799**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 645 files, 0.10 GiB allocated.
- `DELETE_CANDIDATE`: 154 files, 62.58 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=phase2_archive`.

### `/mnt/pool/sqy/lafgs_rgb_prior_cache`

- Scope: 独立 RGB prior query cache
- Allocated size: **57.44 GiB**; apparent file size: **54.12 GiB**; files: **2**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 0 files, 0.00 GiB allocated.
- `DELETE_CANDIDATE`: 2 files, 57.44 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=rgb_prior_cache`.

### `/mnt/pool/sqy/lafgs_query_cache`

- Scope: 独立 12Scenes query cache
- Allocated size: **15.58 GiB**; apparent file size: **14.62 GiB**; files: **1**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 0 files, 0.00 GiB allocated.
- `DELETE_CANDIDATE`: 1 files, 15.58 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=query_cache`.

### `/mnt/pool/sqy/lafgs_v6_feedback_core_20260824`

- Scope: v6 feedback core 历史实验
- Allocated size: **23.26 GiB**; apparent file size: **22.49 GiB**; files: **73**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 30 files, 0.00 GiB allocated.
- `DELETE_CANDIDATE`: 43 files, 23.25 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=v6_feedback_core`.

### `/mnt/pool/sqy/lafgs_v6_feedback_control_20260824`

- Scope: v6 feedback control 历史实验（20260824）
- Allocated size: **24.83 GiB**; apparent file size: **24.54 GiB**; files: **44**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 20 files, 0.00 GiB allocated.
- `DELETE_CANDIDATE`: 24 files, 24.82 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=v6_feedback_control_20260824`.

### `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825`

- Scope: v6 feedback control 历史实验（20260825）
- Allocated size: **25.35 GiB**; apparent file size: **25.33 GiB**; files: **52**.
- `KEEP`: 0 files, 0.00 GiB allocated.
- `ARCHIVE`: 28 files, 0.23 GiB allocated.
- `DELETE_CANDIDATE`: 24 files, 25.11 GiB allocated.
- Exact file-level manifest: [`stdloc_cleanup_dry_run_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv), filtered by `root=v6_feedback_control_20260825`.

## First-level directory rollups

下表按每个 root 的一级目录汇总；更深层的逐文件动作见 TSV。

### `/mnt/pool/sqy/anygsloc_paper_experiments_20260902`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `12Scenes` | 175.21 GiB | 180 / 0.05 GiB | 110 / 0.00 GiB | 73 / 175.16 GiB |
| `CambridgePrior` | 97.79 GiB | 191 / 0.09 GiB | 61 / 0.59 GiB | 63 / 97.11 GiB |
| `7Scenes` | 57.68 GiB | 63 / 0.04 GiB | 44 / 0.00 GiB | 24 / 57.64 GiB |
| `audits` | 0.00 GiB | 8 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_2745419afe43_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_all_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_291512a7a2c3_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_all_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_6d2267b247a8_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_d9340888b4ae_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_d9340888b4ae_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_b8b174fb99d5_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_a1257ce0cedf_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_all_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_291512a7a2c3_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_b8b174fb99d5_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_24_scene_a1257ce0cedf_state.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `prior_robustness_6d2267b247a8_input_audit.json` | 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB |
| `primary_remaining_gpu2.log` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |

### `/mnt/pool/sqy/stdloc_lafgs_legacy_archive_20260801`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `final_state` | 119.66 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | 319 / 119.66 GiB |
| `metadata` | 4.98 GiB | 0 / 0.00 GiB | 8774 / 4.98 GiB | 0 / 0.00 GiB |
| `viewpoint` | 0.26 GiB | 0 / 0.00 GiB | 187 / 0.26 GiB | 0 / 0.00 GiB |
| `metadata_files.txt` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `final_state_manifest.tsv` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `viewpoint_files.txt` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `final_state_files.txt` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `target_manifest.tsv` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |

### `/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `OldHospital` | 93.42 GiB | 28 / 0.80 GiB | 912 / 0.72 GiB | 471 / 91.90 GiB |

### `/mnt/pool/sqy/lafgs_v6_full_mapping_accuracy_20260823`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `GreatCourt` | 26.04 GiB | 4 / 0.00 GiB | 0 / 0.00 GiB | 5 / 26.04 GiB |
| `StMarysChurch` | 25.31 GiB | 4 / 0.00 GiB | 0 / 0.00 GiB | 5 / 25.30 GiB |
| `office` | 20.23 GiB | 5 / 0.01 GiB | 0 / 0.00 GiB | 5 / 20.22 GiB |
| `closed_loop_d4ae8c9` | 11.14 GiB | 5 / 0.00 GiB | 2 / 0.00 GiB | 9 / 11.14 GiB |

### `/mnt/pool/sqy/stdloc_lafgs_prior_robust_a1_20260801`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `rgb_nosky` | 40.17 GiB | 19 / 0.00 GiB | 49 / 0.20 GiB | 13 / 39.97 GiB |
| `rgb_sky_dirty` | 39.90 GiB | 17 / 0.00 GiB | 48 / 0.19 GiB | 12 / 39.71 GiB |
| `evaluation` | 0.06 GiB | 50 / 0.05 GiB | 16 / 0.00 GiB | 0 / 0.00 GiB |
| `logs` | 0.00 GiB | 0 / 0.00 GiB | 7 / 0.00 GiB | 0 / 0.00 GiB |

### `/mnt/pool/sqy/stdloc_lafgs_clean_rebuild_20260728_v1`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `OldHospital` | 78.71 GiB | 3 / 0.00 GiB | 1635 / 1.18 GiB | 662 / 77.53 GiB |

### `/mnt/pool/sqy/stdloc_lafgs_phase2_archive_20260801`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `final_state` | 62.58 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | 154 / 62.58 GiB |
| `metadata` | 0.10 GiB | 0 / 0.00 GiB | 641 / 0.10 GiB | 0 / 0.00 GiB |
| `metadata_files.txt` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `final_state_manifest.tsv` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `final_state_files.txt` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |
| `target_manifest.tsv` | 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB |

### `/mnt/pool/sqy/lafgs_rgb_prior_cache`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `rgb_2dgs_query_cache_native_fullres_k2048.pt` | 28.95 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 28.95 GiB |
| `rgb_sky_dirty_query_cache_native_fullres_k2048.pt` | 28.48 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 28.48 GiB |

### `/mnt/pool/sqy/lafgs_query_cache`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `12Scenes_apt2_kitchen_query_cache.pt` | 15.58 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 15.58 GiB |

### `/mnt/pool/sqy/lafgs_v6_feedback_core_20260824`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `StMarysChurch` | 23.26 GiB | 0 / 0.00 GiB | 30 / 0.00 GiB | 43 / 23.25 GiB |

### `/mnt/pool/sqy/lafgs_v6_feedback_control_20260824`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `StMarysChurch` | 24.83 GiB | 0 / 0.00 GiB | 20 / 0.00 GiB | 24 / 24.82 GiB |

### `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825`

| first-level directory | allocated | KEEP | ARCHIVE | DELETE_CANDIDATE |
|---|---:|---:|---:|---:|
| `StMarysChurch` | 25.35 GiB | 0 / 0.00 GiB | 28 / 0.23 GiB | 24 / 25.11 GiB |

## Additional large roots not included in the exact manifest

这些目录在下一轮 dry-run 中优先复核；本报告不对它们做任何操作：

- `/mnt/pool/sqy/lafgs_projective_anchor_v4_cambridge_full_20260817` — allocated **63.61 GiB**
- `/mnt/pool/sqy/lafgs_7scenes_undistorted_factor_20260809` — allocated **67.34 GiB**
- `/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901` — allocated **44.08 GiB**
- `/mnt/pool/sqy/lafgs_v5_phase1_resource_oracle_20260821` — allocated **44.04 GiB**
- `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827` — allocated **16.50 GiB**
- `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802` — allocated **9.81 GiB**

## Protection notes

- 保留当前 indoor 先验、5 个 Cambridge MAtCha 先验、Cambridge/12Scenes 原始或处理数据。
- `anygsloc_paper_experiments_20260902` 的 PLY 先验不在该目录内；其中 feature cache/map 的候选删除不会直接删除先验 PLY。
- 在用户批准前，本清单中的 `DELETE_CANDIDATE` 只表示候选，不执行删除。

## Size accounting

文件级 `apparent_bytes` 与 allocated bytes 都写入 TSV；总目录占用使用 `du -sB1`，因此可能因稀疏文件/目录块与文件 apparent size 略有差异。


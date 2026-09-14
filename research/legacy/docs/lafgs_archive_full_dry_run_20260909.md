# LaFGS / STDLoc full archive dry-run

Generated: `2026-09-09T20:52:05+08:00`

本报告只读扫描并生成逐文件清单；没有创建、移动、删除或修改 `/mnt/pool/sqy` 下的任何实验文件。

统一归档目标：`/mnt/pool/sqy/LaFGS_archive/<功能>/<日期>/<原目录>/<相对路径>`。实际执行前仍需再次确认；`G4Splat_runs` 与 `/mnt/pool/sqy/datasets` 原始数据树不在移动/删除范围内。

动作含义：`KEEP` 原位保留；`ARCHIVE` 移入统一 LaFGS_archive；`DELETE_CANDIDATE` 可重建的删除候选，当前不执行。

## Overall summary

- 扫描 root：**185**；逐条文件/符号链接记录：**89564**。
- KEEP：27451 条，559.37 GiB allocated。
- ARCHIVE：40514 条，46.58 GiB allocated。
- DELETE_CANDIDATE：21599 条，692.27 GiB allocated。

## Root summary

| root | allocated | records | KEEP | ARCHIVE | DELETE_CANDIDATE | refs |
|---|---:|---:|---:|---:|---:|---|
| `anygsloc_execution_queue_20260909` | 66.61 GiB | 77 | 77 / 66.61 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | doc:3 |
| `anygsloc_paper_experiments_20260902` | 330.67 GiB | 835 | 835 / 330.67 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | config:1, script:2, doc:5 |
| `Cambridge_stdloc` | 115.03 GiB | 21964 | 21964 / 115.03 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | config:2, script:5, doc:11 |
| `huggingface_cache/hub/models--lhjiang--anysplat` | 2.93 GiB | 10 | 0 / 0.00 GiB | 0 / 0.00 GiB | 10 / 2.93 GiB | none |
| `lafgs_7scenes_track_repair_20260812` | 14.51 GiB | 74 | 0 / 0.00 GiB | 42 / 0.03 GiB | 32 / 14.48 GiB | doc:2 |
| `lafgs_7scenes_undistorted_factor_20260809` | 67.34 GiB | 493 | 0 / 0.00 GiB | 304 / 0.59 GiB | 189 / 66.75 GiB | doc:1 |
| `lafgs_adaptive_v3_followup_20260807` | 0.09 GiB | 60 | 0 / 0.00 GiB | 53 / 0.05 GiB | 7 / 0.04 GiB | none |
| `lafgs_adaptive_v3_full_benchmark_20260807` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | doc:20 |
| `lafgs_adaptive_v3_validation_20260806` | 9.51 GiB | 162 | 0 / 0.00 GiB | 109 / 0.12 GiB | 53 / 9.39 GiB | doc:1 |
| `lafgs_alias_density_ablation_20260810` | 0.16 GiB | 80 | 0 / 0.00 GiB | 71 / 0.16 GiB | 9 / 0.00 GiB | none |
| `lafgs_anchor_identity_p51_validation_20260812` | 4.12 GiB | 89 | 0 / 0.00 GiB | 41 / 0.03 GiB | 48 / 4.09 GiB | doc:4 |
| `lafgs_assets` | 0.00 GiB | 1 | 1 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_closed_loop_ablation_20260810` | 0.85 GiB | 97 | 0 / 0.00 GiB | 72 / 0.29 GiB | 25 / 0.57 GiB | none |
| `lafgs_deployment_revision_20260806` | 0.04 GiB | 32 | 0 / 0.00 GiB | 23 / 0.03 GiB | 9 / 0.02 GiB | none |
| `lafgs_descriptor_gap_attribution_20260821` | 4.81 GiB | 16 | 0 / 0.00 GiB | 7 / 0.00 GiB | 9 / 4.81 GiB | doc:1 |
| `lafgs_detachpnp_ng_l1_oh` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_experiment_tools` | 0.00 GiB | 6 | 6 / 0.00 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_fbguard30keepgt_reproj12_oh` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_frontend_ceiling_probe_20260813` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | doc:6 |
| `lafgs_identity_mode_revision_20260810` | 0.18 GiB | 11 | 0 / 0.00 GiB | 7 / 0.05 GiB | 4 / 0.13 GiB | doc:1 |
| `lafgs_indoor_pgt_runs_20260804` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | doc:2 |
| `lafgs_indoor_runs_20260803` | 20.92 GiB | 75 | 0 / 0.00 GiB | 32 / 0.08 GiB | 43 / 20.84 GiB | none |
| `lafgs_indoor_sfm_env` | 0.02 GiB | 882 | 0 / 0.00 GiB | 882 / 0.02 GiB | 0 / 0.00 GiB | none |
| `lafgs_indoor_sfm_env_install.log` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_indoor_sfm_py39` | 0.15 GiB | 2793 | 0 / 0.00 GiB | 2793 / 0.15 GiB | 0 / 0.00 GiB | none |
| `lafgs_indoor_sfm_py39_install.log` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_mapping_eval_shard_probe_20260820_3a033fe` | 0.00 GiB | 18 | 0 / 0.00 GiB | 9 / 0.00 GiB | 9 / 0.00 GiB | doc:2 |
| `lafgs_model_cache` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_offline_accel_probe_20260820` | 0.65 GiB | 19 | 0 / 0.00 GiB | 0 / 0.00 GiB | 19 / 1.63 GiB | doc:1 |
| `lafgs_oldhospital_topology_ablation_20260806` | 0.10 GiB | 30 | 0 / 0.00 GiB | 23 / 0.08 GiB | 7 / 0.01 GiB | none |
| `lafgs_p7_density_factor_20260812` | 10.63 GiB | 25 | 0 / 0.00 GiB | 13 / 0.00 GiB | 12 / 10.63 GiB | doc:14 |
| `lafgs_p7_pair_policy_factor_20260812` | 50.10 GiB | 32 | 0 / 0.00 GiB | 23 / 0.00 GiB | 9 / 50.10 GiB | doc:10 |
| `lafgs_p7_pair_policy_fullchain_20260812` | 5.24 GiB | 53 | 0 / 0.00 GiB | 40 / 0.02 GiB | 13 / 5.22 GiB | doc:3 |
| `lafgs_p8_cycle_verified_fisher_20260813` | 1.09 GiB | 94 | 0 / 0.00 GiB | 77 / 0.00 GiB | 17 / 1.09 GiB | doc:7 |
| `lafgs_paper_figures_20260806` | 0.06 GiB | 23 | 0 / 0.00 GiB | 23 / 0.06 GiB | 0 / 0.00 GiB | script:1, doc:3 |
| `lafgs_paper_release_gate_20260803` | 9.50 GiB | 80 | 0 / 0.00 GiB | 44 / 0.83 GiB | 36 / 8.68 GiB | none |
| `lafgs_pg002_l1_oh` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_pg0_l1_oh` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_pg0_oh` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_pose_set_oracle_20260810` | 0.00 GiB | 10 | 0 / 0.00 GiB | 4 / 0.00 GiB | 6 / 0.00 GiB | none |
| `lafgs_pose_set_refinement_20260810` | 0.09 GiB | 33 | 0 / 0.00 GiB | 29 / 0.00 GiB | 4 / 0.09 GiB | none |
| `lafgs_private_archive_20260803` | 4.90 GiB | 8774 | 0 / 0.00 GiB | 5317 / 2.75 GiB | 3457 / 2.13 GiB | none |
| `lafgs_projective_anchor_architecture_v1_20260816` | 0.43 GiB | 93 | 0 / 0.00 GiB | 61 / 0.01 GiB | 32 / 0.42 GiB | doc:23 |
| `lafgs_projective_anchor_v4_assignment_followup.log` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_projective_anchor_v4_assignment_mapping_loo_20260819` | 0.02 GiB | 64 | 0 / 0.00 GiB | 44 / 0.00 GiB | 20 / 0.02 GiB | doc:1 |
| `lafgs_projective_anchor_v4_assignment_mapping_loo_20260819_coordinator.log` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_projective_anchor_v4_assignment_top1_resume.log` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_projective_anchor_v4_cambridge_full_20260817` | 63.61 GiB | 355 | 0 / 0.00 GiB | 265 / 0.15 GiB | 90 / 63.46 GiB | config:1, doc:1 |
| `lafgs_projective_anchor_v4_cycle_core_20260816` | 0.31 GiB | 44 | 0 / 0.00 GiB | 26 / 0.00 GiB | 18 / 0.31 GiB | doc:7 |
| `lafgs_projective_anchor_v4_point_refine_20260817` | 0.47 GiB | 50 | 0 / 0.00 GiB | 28 / 0.00 GiB | 22 / 0.47 GiB | doc:11 |
| `lafgs_projective_anchor_v4_single_seed_20260818` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | doc:1 |
| `lafgs_render_perturbation_stability_20260821` | 0.02 GiB | 6 | 0 / 0.00 GiB | 4 / 0.00 GiB | 2 / 0.02 GiB | doc:1 |
| `lafgs_render_track_full_reference_history_replay_20260816` | 14.08 GiB | 182 | 0 / 0.00 GiB | 146 / 0.07 GiB | 36 / 14.01 GiB | script:1, doc:38 |
| `lafgs_render_track_full_reference_v5_20260815` | 7.41 GiB | 122 | 0 / 0.00 GiB | 93 / 0.05 GiB | 29 / 7.36 GiB | script:1, doc:67 |
| `lafgs_render_track_only_appearance_v11_20260814` | 7.27 GiB | 115 | 0 / 0.00 GiB | 100 / 0.04 GiB | 15 / 7.23 GiB | script:1, doc:136 |
| `lafgs_render_track_only_artifact_r1_20260815` | 10.12 GiB | 132 | 0 / 0.00 GiB | 92 / 0.03 GiB | 40 / 10.09 GiB | script:2, doc:31 |
| `lafgs_render_track_only_conditional_gwff_20260815` | 14.88 GiB | 152 | 0 / 0.00 GiB | 113 / 0.08 GiB | 39 / 14.81 GiB | script:1, doc:31 |
| `lafgs_render_track_only_fullchain_20260814` | 1.00 GiB | 156 | 0 / 0.00 GiB | 116 / 0.10 GiB | 40 / 0.89 GiB | config:1, script:1, doc:42 |
| `lafgs_render_track_only_fullmap_v14_20260815` | 0.35 GiB | 92 | 0 / 0.00 GiB | 77 / 0.02 GiB | 15 / 0.33 GiB | script:2, doc:29 |
| `lafgs_render_track_only_support_v12_20260814` | 2.87 GiB | 416 | 0 / 0.00 GiB | 276 / 0.38 GiB | 140 / 2.50 GiB | script:1, doc:49 |
| `lafgs_render_track_only_support_v13_20260814` | 2.05 GiB | 223 | 0 / 0.00 GiB | 122 / 0.08 GiB | 101 / 1.97 GiB | script:1, doc:108 |
| `lafgs_rendered_rgb_only_probe_20260813` | 2.36 GiB | 17 | 0 / 0.00 GiB | 6 / 0.00 GiB | 11 / 2.35 GiB | script:1, doc:18 |
| `lafgs_rendered_rgb_only_trainonly_20260813` | 3.69 GiB | 126 | 0 / 0.00 GiB | 41 / 0.50 GiB | 85 / 3.19 GiB | script:1, doc:20 |
| `lafgs_resource_reverse_ablation_20260821` | 0.05 GiB | 14 | 0 / 0.00 GiB | 10 / 0.00 GiB | 4 / 0.05 GiB | doc:2 |
| `lafgs_shopfacade_topology_ablation_20260806` | 0.08 GiB | 34 | 0 / 0.00 GiB | 27 / 0.07 GiB | 7 / 0.01 GiB | none |
| `lafgs_surface_supported_tracks_g1_20260810` | 3.19 GiB | 96 | 0 / 0.00 GiB | 58 / 0.17 GiB | 38 / 3.01 GiB | none |
| `lafgs_symmetric_photometric_clahe_v1_20260821` | 0.05 GiB | 14 | 0 / 0.00 GiB | 12 / 0.05 GiB | 2 / 0.00 GiB | doc:1 |
| `lafgs_symmetric_photometric_v1_20260821` | 0.05 GiB | 14 | 0 / 0.00 GiB | 12 / 0.05 GiB | 2 / 0.00 GiB | doc:1 |
| `lafgs_v10_actionable_20260828` | 3.79 GiB | 354 | 0 / 0.00 GiB | 13 / 0.00 GiB | 341 / 3.79 GiB | doc:1 |
| `lafgs_v11_pose_detector_20260828` | 2.04 GiB | 319 | 0 / 0.00 GiB | 16 / 0.00 GiB | 303 / 2.04 GiB | doc:1 |
| `lafgs_v13_risk_closed_loop_20260828` | 4.97 GiB | 179 | 0 / 0.00 GiB | 8 / 0.00 GiB | 171 / 4.97 GiB | none |
| `lafgs_v14_valid_row_closed_loop_20260828` | 9.44 GiB | 743 | 0 / 0.00 GiB | 30 / 3.14 GiB | 713 / 6.30 GiB | doc:1 |
| `lafgs_v15_global_sufficiency_20260828` | 19.62 GiB | 540 | 0 / 0.00 GiB | 112 / 8.04 GiB | 428 / 11.57 GiB | doc:1 |
| `lafgs_v16_competitive_closed_loop_20260829` | 8.29 GiB | 276 | 0 / 0.00 GiB | 14 / 0.52 GiB | 262 / 7.77 GiB | doc:1 |
| `lafgs_v17_competitive_metric_20260829` | 11.95 GiB | 454 | 0 / 0.00 GiB | 30 / 0.00 GiB | 424 / 11.95 GiB | doc:1 |
| `lafgs_v18_provenance_truth_20260829` | 4.78 GiB | 158 | 0 / 0.00 GiB | 38 / 0.09 GiB | 120 / 4.69 GiB | doc:1 |
| `lafgs_v19_track_extension_20260830` | 0.06 GiB | 31 | 0 / 0.00 GiB | 12 / 0.00 GiB | 19 / 0.06 GiB | doc:1 |
| `lafgs_v20_sparse_anchor_20260831` | 3.73 GiB | 293 | 0 / 0.00 GiB | 28 / 3.36 GiB | 265 / 0.37 GiB | doc:2 |
| `lafgs_v21_test_targeted_20260831` | 2.39 GiB | 48 | 0 / 0.00 GiB | 14 / 0.01 GiB | 34 / 2.38 GiB | none |
| `lafgs_v23_online_cambridge_20260831` | 0.02 GiB | 64 | 0 / 0.00 GiB | 64 / 0.02 GiB | 0 / 0.00 GiB | doc:1 |
| `lafgs_v24_cambridge_high_capacity_online_20260901` | 44.08 GiB | 358 | 358 / 44.08 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | config:4, doc:4 |
| `lafgs_v4_assignment_accel_smoke_20260820` | 0.50 GiB | 10 | 0 / 0.00 GiB | 7 / 0.00 GiB | 3 / 0.50 GiB | doc:1 |
| `lafgs_v4_assignment_margin_smoke_20260820` | 0.00 GiB | 9 | 0 / 0.00 GiB | 5 / 0.00 GiB | 4 / 0.00 GiB | doc:1 |
| `lafgs_v4_covisibility_oracle_38d30a6_20260820` | 0.00 GiB | 4 | 0 / 0.00 GiB | 2 / 0.00 GiB | 2 / 0.00 GiB | none |
| `lafgs_v4_failure_chain_e29536b_20260821` | 0.01 GiB | 6 | 0 / 0.00 GiB | 3 / 0.00 GiB | 3 / 0.01 GiB | doc:1 |
| `lafgs_v4_failure_chain_e36215c_20260821` | 0.00 GiB | 4 | 0 / 0.00 GiB | 2 / 0.00 GiB | 2 / 0.00 GiB | none |
| `lafgs_v4_failure_chain_panel_e29536b_20260821` | 0.02 GiB | 14 | 0 / 0.00 GiB | 7 / 0.00 GiB | 7 / 0.02 GiB | doc:1 |
| `lafgs_v4_fullmap_shard_probe_20260820_48f35e4` | 0.00 GiB | 18 | 0 / 0.00 GiB | 9 / 0.00 GiB | 9 / 0.00 GiB | doc:2 |
| `lafgs_v4_shopfacade_sentinel_20260812` | 0.64 GiB | 30 | 0 / 0.00 GiB | 19 / 0.01 GiB | 11 / 0.62 GiB | doc:1 |
| `lafgs_v4_sufficiency_virtual_render_shop_20260821` | 0.88 GiB | 35 | 0 / 0.00 GiB | 29 / 0.05 GiB | 6 / 0.83 GiB | doc:1 |
| `lafgs_v4_view_mixture_audit_20260820` | 0.04 GiB | 21 | 0 / 0.00 GiB | 17 / 0.00 GiB | 4 / 0.04 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_20260820` | 0.00 GiB | 8 | 0 / 0.00 GiB | 4 / 0.00 GiB | 4 / 0.00 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_final_20260820_invalid_dirty_producer` | 0.00 GiB | 5 | 0 / 0.00 GiB | 4 / 0.00 GiB | 1 / 0.00 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_final_sharded_20260820` | 0.01 GiB | 72 | 0 / 0.00 GiB | 52 / 0.00 GiB | 20 / 0.01 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820` | 0.00 GiB | 2 | 0 / 0.00 GiB | 0 / 0.00 GiB | 2 / 0.00 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820_run1` | 0.00 GiB | 2 | 0 / 0.00 GiB | 0 / 0.00 GiB | 2 / 0.00 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820_run2` | 0.00 GiB | 2 | 0 / 0.00 GiB | 0 / 0.00 GiB | 2 / 0.00 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820_run3` | 0.01 GiB | 28 | 0 / 0.00 GiB | 10 / 0.00 GiB | 18 / 0.01 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820_run4` | 0.02 GiB | 48 | 0 / 0.00 GiB | 18 / 0.00 GiB | 30 / 0.02 GiB | none |
| `lafgs_v4_view_mixture_pose_gate_level1_8073_20260820_run5` | 0.01 GiB | 24 | 0 / 0.00 GiB | 6 / 0.00 GiB | 18 / 0.01 GiB | none |
| `lafgs_v5_adapter_aux_phase1_prereg_20260821` | 0.00 GiB | 2 | 0 / 0.00 GiB | 2 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_v5_adapter_aux_phase1_prereg_v2_20260821` | 0.00 GiB | 2 | 0 / 0.00 GiB | 2 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_v5_adapter_aux_phase1_prereg_v3_20260821` | 0.00 GiB | 2 | 0 / 0.00 GiB | 2 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_v5_adapter_aux_phase1_training_20260821` | 6.72 GiB | 8 | 0 / 0.00 GiB | 2 / 0.00 GiB | 6 / 6.72 GiB | none |
| `lafgs_v5_adapter_aux_phase1_training_v2_20260821` | 2.21 GiB | 5 | 0 / 0.00 GiB | 2 / 0.00 GiB | 3 / 2.21 GiB | none |
| `lafgs_v5_adapter_aux_phase1_training_v3_20260821` | 6.70 GiB | 10 | 0 / 0.00 GiB | 4 / 0.00 GiB | 6 / 6.70 GiB | none |
| `lafgs_v5_adapter_aux_smoke_20260821` | 3.72 GiB | 13 | 0 / 0.00 GiB | 5 / 0.00 GiB | 8 / 3.72 GiB | none |
| `lafgs_v5_adapter_aux_sparse_pair_prereg_fix_20260822` | 0.00 GiB | 2 | 0 / 0.00 GiB | 2 / 0.00 GiB | 0 / 0.00 GiB | none |
| `lafgs_v5_adapter_aux_sparse_pair_probe_20260821` | 0.15 GiB | 8 | 0 / 0.00 GiB | 4 / 0.00 GiB | 4 / 0.15 GiB | none |
| `lafgs_v5_certified_anchor_smoke_20260822` | 0.21 GiB | 69 | 0 / 0.00 GiB | 33 / 0.05 GiB | 36 / 0.16 GiB | none |
| `lafgs_v5_phase1_resource_oracle_20260821` | 44.04 GiB | 117 | 0 / 0.00 GiB | 73 / 0.03 GiB | 44 / 44.01 GiB | doc:1 |
| `lafgs_v5_projective_support_cert_20260821` | 1.50 GiB | 64 | 0 / 0.00 GiB | 28 / 0.34 GiB | 36 / 1.16 GiB | none |
| `lafgs_v6_closed_loop_projective_20260822` | 1.30 GiB | 65 | 0 / 0.00 GiB | 40 / 0.05 GiB | 25 / 1.25 GiB | none |
| `lafgs_v6_full_mapping_accuracy_20260823` | 0.01 GiB | 18 | 0 / 0.00 GiB | 18 / 0.01 GiB | 0 / 0.00 GiB | doc:1 |
| `lafgs_v6_mainline_convergence_20260823` | 11.34 GiB | 74 | 0 / 0.00 GiB | 33 / 0.00 GiB | 41 / 11.34 GiB | none |
| `lafgs_v7_anchor_contamination_audit_20260827` | 1.58 GiB | 36 | 0 / 0.00 GiB | 16 / 0.00 GiB | 20 / 1.58 GiB | doc:1 |
| `lafgs_v7_full_continuous_mainline_20260827` | 16.50 GiB | 1117 | 0 / 0.00 GiB | 7 / 0.00 GiB | 1110 / 16.50 GiB | doc:2 |
| `lafgs_v7_render_real_gap_p05_20260827` | 0.01 GiB | 10 | 0 / 0.00 GiB | 10 / 0.01 GiB | 0 / 0.00 GiB | doc:1 |
| `lafgs_v7_safe_closed_loop_20260826` | 20.15 GiB | 1466 | 0 / 0.00 GiB | 122 / 0.52 GiB | 1344 / 19.63 GiB | none |
| `lafgs_v8_clean_anchor_scene_detector_20260828` | 2.12 GiB | 638 | 0 / 0.00 GiB | 28 / 0.05 GiB | 610 / 2.07 GiB | doc:1 |
| `lafgs_v8_v2_feedback_20260828` | 1.23 GiB | 83 | 0 / 0.00 GiB | 11 / 0.52 GiB | 72 / 0.71 GiB | doc:1 |
| `lafgs_v8_v2_full_rebuild_20260828` | 1.14 GiB | 7 | 0 / 0.00 GiB | 7 / 1.14 GiB | 0 / 0.00 GiB | doc:2 |
| `lafgs_v9_no_loo_20260828` | 13.90 GiB | 669 | 0 / 0.00 GiB | 20 / 1.57 GiB | 649 / 12.33 GiB | doc:1 |
| `lafgs_xfeat_arm_a_20260813` | 0.06 GiB | 5 | 0 / 0.00 GiB | 3 / 0.01 GiB | 2 / 0.05 GiB | doc:3 |
| `lafgs_xfeat_arm_b_20260813` | 0.54 GiB | 5 | 0 / 0.00 GiB | 2 / 0.00 GiB | 3 / 0.54 GiB | doc:6 |
| `lafgs_xfeat_equal_energy_descriptor_fullchain_20260813` | 11.78 GiB | 17 | 0 / 0.00 GiB | 12 / 0.02 GiB | 5 / 11.77 GiB | doc:4 |
| `stdloc_checkpoint_cleanup_20260709_chkpnt_locaware_deleted.tsv` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `stdloc_closed_loop_archive_20260909` | 10.09 GiB | 15411 | 0 / 0.00 GiB | 15411 / 10.06 GiB | 0 / 0.00 GiB | doc:1 |
| `stdloc_la_update4_prune_mb500_combined_summary_final.json` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `stdloc_lafgs_2dgs_anchor_ablation_20260707` | 5.67 GiB | 124 | 0 / 0.00 GiB | 67 / 0.00 GiB | 57 / 5.67 GiB | none |
| `stdloc_lafgs_2dgs_longloc_20260707` | 6.63 GiB | 71 | 0 / 0.00 GiB | 30 / 0.00 GiB | 41 / 6.63 GiB | none |
| `stdloc_lafgs_2dgs_surface_external_20260706` | 0.00 GiB | 3 | 0 / 0.00 GiB | 0 / 0.00 GiB | 3 / 0.00 GiB | none |
| `stdloc_lafgs_2dgs_surface_full_20260706` | 3.03 GiB | 80 | 0 / 0.00 GiB | 16 / 0.57 GiB | 64 / 2.46 GiB | none |
| `stdloc_lafgs_2dgs_surface_preflight_diffpnp_20260706` | 0.14 GiB | 12 | 0 / 0.00 GiB | 9 / 0.00 GiB | 3 / 0.14 GiB | none |
| `stdloc_lafgs_2dgs_surface_smoke_20260706` | 0.30 GiB | 26 | 0 / 0.00 GiB | 12 / 0.00 GiB | 14 / 0.30 GiB | none |
| `stdloc_lafgs_active_map_v2_20260727` | 0.17 GiB | 31 | 0 / 0.00 GiB | 28 / 0.02 GiB | 3 / 0.15 GiB | none |
| `stdloc_lafgs_active_map_v3_20260727` | 0.79 GiB | 139 | 0 / 0.00 GiB | 110 / 0.28 GiB | 29 / 0.51 GiB | none |
| `stdloc_lafgs_active_map_v4_20260727` | 2.88 GiB | 216 | 0 / 0.00 GiB | 168 / 0.36 GiB | 48 / 2.52 GiB | none |
| `stdloc_lafgs_active_map_v5_20260727` | 4.34 GiB | 99 | 0 / 0.00 GiB | 52 / 0.06 GiB | 47 / 4.28 GiB | none |
| `stdloc_lafgs_active_map_v6_20260728` | 0.94 GiB | 985 | 0 / 0.00 GiB | 967 / 0.54 GiB | 18 / 0.40 GiB | none |
| `stdloc_lafgs_adaptive_v2_20260806` | 0.15 GiB | 21 | 0 / 0.00 GiB | 13 / 0.02 GiB | 8 / 0.13 GiB | none |
| `stdloc_lafgs_alternating_20260727` | 0.74 GiB | 48 | 0 / 0.00 GiB | 41 / 0.37 GiB | 7 / 0.37 GiB | none |
| `stdloc_lafgs_cambridge_best_crossscene_20260711` | 0.01 GiB | 56 | 0 / 0.00 GiB | 56 / 0.01 GiB | 0 / 0.00 GiB | none |
| `stdloc_lafgs_cambridge_matcha2dgs_strict_20260711` | 17.75 GiB | 763 | 0 / 0.00 GiB | 213 / 1.43 GiB | 550 / 16.32 GiB | none |
| `stdloc_lafgs_candidate_aligned_20260710` | 0.06 GiB | 332 | 0 / 0.00 GiB | 326 / 0.01 GiB | 6 / 0.05 GiB | none |
| `stdloc_lafgs_clean_field_childguard_smoke_20260710` | 0.53 GiB | 16 | 0 / 0.00 GiB | 7 / 0.00 GiB | 9 / 0.53 GiB | none |
| `stdloc_lafgs_clean_field_full30k_20260710` | 0.03 GiB | 7 | 0 / 0.00 GiB | 5 / 0.00 GiB | 2 / 0.03 GiB | none |
| `stdloc_lafgs_clean_field_independent_smoke_20260710` | 0.54 GiB | 16 | 0 / 0.00 GiB | 7 / 0.00 GiB | 9 / 0.54 GiB | none |
| `stdloc_lafgs_clean_field_late_smoke_20260709` | 0.54 GiB | 16 | 0 / 0.00 GiB | 7 / 0.00 GiB | 9 / 0.54 GiB | none |
| `stdloc_lafgs_clean_field_objective_smoke_20260709` | 0.57 GiB | 33 | 0 / 0.00 GiB | 16 / 0.00 GiB | 17 / 0.57 GiB | none |
| `stdloc_lafgs_clean_field_resdiag_smoke_20260710` | 0.55 GiB | 29 | 0 / 0.00 GiB | 12 / 0.00 GiB | 17 / 0.55 GiB | none |
| `stdloc_lafgs_clean_rebuild_20260728_v1` | 0.00 GiB | 3 | 0 / 0.00 GiB | 3 / 0.00 GiB | 0 / 0.00 GiB | doc:1 |
| `stdloc_lafgs_feedbackfix_full_20260708` | 2.28 GiB | 25 | 0 / 0.00 GiB | 12 / 0.00 GiB | 13 / 2.28 GiB | none |
| `stdloc_lafgs_feedbackfix_probe_20260708` | 1.07 GiB | 9 | 0 / 0.00 GiB | 6 / 0.00 GiB | 3 / 1.07 GiB | none |
| `stdloc_lafgs_joint_assignment_20260731` | 1.55 GiB | 3658 | 0 / 0.00 GiB | 66 / 0.21 GiB | 3592 / 1.34 GiB | none |
| `stdloc_lafgs_joint_assignment_p1_20260731` | 4.83 GiB | 248 | 0 / 0.00 GiB | 207 / 0.24 GiB | 41 / 4.59 GiB | none |
| `stdloc_lafgs_legacy_cleanup_logs` | 0.00 GiB | 1 | 0 / 0.00 GiB | 1 / 0.00 GiB | 0 / 0.00 GiB | none |
| `stdloc_lafgs_matrix_20260709` | 0.78 GiB | 35 | 0 / 0.00 GiB | 25 / 0.00 GiB | 10 / 0.78 GiB | none |
| `stdloc_lafgs_matrix_fixlineage_20260709` | 11.80 GiB | 875 | 0 / 0.00 GiB | 136 / 0.62 GiB | 739 / 11.18 GiB | none |
| `stdloc_lafgs_micro_anchors_20260726` | 4.57 GiB | 571 | 0 / 0.00 GiB | 469 / 1.58 GiB | 102 / 2.99 GiB | none |
| `stdloc_lafgs_offtheshelf_prior_20260802` | 9.81 GiB | 9143 | 20 / 1.57 GiB | 6414 / 0.01 GiB | 2709 / 8.23 GiB | config:1, doc:5 |
| `stdloc_lafgs_preemptive_20260731` | 0.03 GiB | 41 | 0 / 0.00 GiB | 41 / 0.03 GiB | 0 / 0.00 GiB | none |
| `stdloc_lafgs_prior_robust_a1_20260801` | 0.06 GiB | 86 | 0 / 0.00 GiB | 86 / 0.06 GiB | 0 / 0.00 GiB | doc:1 |
| `stdloc_lafgs_resume_fix_smoke_20260708` | 0.39 GiB | 8 | 0 / 0.00 GiB | 4 / 0.00 GiB | 4 / 0.39 GiB | none |
| `stdloc_lafgs_rgb_prior_sanitization_20260725` | 0.80 GiB | 31 | 0 / 0.00 GiB | 27 / 0.80 GiB | 4 / 0.00 GiB | doc:1 |
| `stdloc_lafgs_scss_20260710` | 0.00 GiB | 41 | 0 / 0.00 GiB | 41 / 0.00 GiB | 0 / 0.00 GiB | none |
| `stdloc_lafgs_sfm_zero_20260708` | 14.21 GiB | 43 | 0 / 0.00 GiB | 12 / 0.47 GiB | 31 / 13.74 GiB | none |
| `stdloc_lafgs_shop_strict_fresh_20260712` | 8.93 GiB | 994 | 0 / 0.00 GiB | 348 / 0.17 GiB | 646 / 8.76 GiB | none |
| `stdloc_lafgs_slps_20260731` | 14.29 GiB | 593 | 0 / 0.00 GiB | 490 / 0.14 GiB | 103 / 14.15 GiB | none |
| `stdloc_lafgs_track_teacher_20260726` | 2.29 GiB | 177 | 0 / 0.00 GiB | 68 / 0.04 GiB | 109 / 2.25 GiB | none |
| `stdloc_lafgs_v1_frozen_multiscene_20260731` | 0.95 GiB | 17 | 17 / 0.95 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | config:1, script:1, doc:8 |
| `stdloc_lafgs_v1_group_consensus_oracle_20260731` | 0.05 GiB | 197 | 0 / 0.00 GiB | 193 / 0.05 GiB | 4 / 0.00 GiB | none |
| `stdloc_lafgs_v1_group_saturated_20260731` | 0.04 GiB | 58 | 0 / 0.00 GiB | 54 / 0.04 GiB | 4 / 0.00 GiB | none |
| `stdloc_lafgs_v1_group_saturated_v2_20260731` | 0.00 GiB | 1 | 0 / 0.00 GiB | 0 / 0.00 GiB | 1 / 0.00 GiB | none |
| `stdloc_lafgs_v7_20260728` | 0.66 GiB | 314 | 0 / 0.00 GiB | 291 / 0.60 GiB | 23 / 0.07 GiB | none |
| `stdloc_lafgs_v8_20260728` | 1.62 GiB | 224 | 0 / 0.00 GiB | 174 / 0.33 GiB | 50 / 1.30 GiB | none |
| `stdloc_lafgs_v9_20260728` | 0.66 GiB | 107 | 0 / 0.00 GiB | 80 / 0.28 GiB | 27 / 0.38 GiB | none |
| `stdloc_matcha_bugfix_validation_20260715` | 0.00 GiB | 19 | 0 / 0.00 GiB | 18 / 0.00 GiB | 1 / 0.00 GiB | none |
| `stdloc_matcha_bugfix_validation_v2_20260715` | 0.00 GiB | 19 | 0 / 0.00 GiB | 18 / 0.00 GiB | 1 / 0.00 GiB | none |
| `stdloc_matcha_pretrained_exact_taskscaled_v2_20260715` | 5.81 GiB | 236 | 0 / 0.00 GiB | 202 / 0.13 GiB | 34 / 5.68 GiB | none |
| `stdloc_matcha_pretrained_mainline_v1_20260714` | 33.44 GiB | 822 | 0 / 0.00 GiB | 563 / 0.27 GiB | 259 / 33.16 GiB | none |
| `stdloc_matcha_taskscaled_smoke_20260715` | 0.00 GiB | 19 | 0 / 0.00 GiB | 18 / 0.00 GiB | 1 / 0.00 GiB | none |
| `stdloc_matcha_taskscaled_smoke_20260715b` | 0.00 GiB | 19 | 0 / 0.00 GiB | 18 / 0.00 GiB | 1 / 0.00 GiB | none |
| `stdloc_rgb2dgs_fullsplit_strict_20260712` | 0.53 GiB | 70 | 0 / 0.00 GiB | 18 / 0.00 GiB | 52 / 0.53 GiB | none |
| `third_party_lafgs_priors` | 0.44 GiB | 4173 | 4173 / 0.44 GiB | 0 / 0.00 GiB | 0 / 0.00 GiB | script:2 |

## Protection and migration notes

- `anygsloc_paper_experiments_20260902`、`anygsloc_execution_queue_20260909`、`Cambridge_stdloc`、五个 MAtCha 先验、`lafgs_v24_cambridge_high_capacity_online_20260901` 均按 KEEP 处理。
- `stdloc_lafgs_offtheshelf_prior_20260802` 只保留四个 Cambridge vanilla 2DGS/3DGS 最终先验 PLY、manifest、quality/protocol；训练日志、事件文件、重复数据集副本会分别归档或列为删除候选。
- v23 闭环 map/metric 所在目录按 ARCHIVE 迁移；迁移后若要重新运行 v23，需要同步更新配置中的绝对路径。
- 旧 `stdloc_closed_loop_archive_20260909` 计划迁入 `LaFGS_archive/legacy_archive_migration/2026-09-09/`，避免继续保留多个归档入口。
- 5 个顶层 `lafgs_*_oh` 失效符号链接会列为 DELETE_CANDIDATE，不会被复制成新的归档垃圾。
- 旧 AnySplat 内容（包括 `lafgs_paper_release_gate_20260803` 下的残留 prior、嵌套 HuggingFace cache）列为 DELETE_CANDIDATE；只保留上一轮清理 receipt 作为归档记录。
- `docs/` 中对旧实验路径的引用只作为历史证据；不等于这些二进制仍是运行时依赖。

## File-level manifest

完整 TSV：[`lafgs_archive_full_dry_run_20260909.tsv`](/root/STDLoc/docs/lafgs_archive_full_dry_run_20260909.tsv)。字段包含源路径、动作、原因和统一归档目标路径。

## Not scanned

- `/mnt/pool/sqy/G4Splat_runs`：按用户要求保持不动。
- `/mnt/pool/sqy/datasets`：作为原始/处理后数据树保持不动；仅在清单中通过当前 AnyGSLoc 路径说明其受保护状态。

# STDLoc / LaFGS cleanup execution receipt v2

Started: `2026-09-09T18:11:51+08:00`
Finished: `2026-09-09T18:12:23+08:00`

本次只处理 v2 dry-run manifest 中的旧 STDLoc/LaFGS 根目录。`anygsloc_paper_experiments_20260902`、当前 AnyGSLoc 队列和 `G4Splat_runs` 均跳过。

| action | files | apparent | allocated before |
|---|---:|---:|---:|
| `KEEP` | 135 | 0.82 GiB | 0.87 GiB |
| `ARCHIVE` | 12358 | 7.25 GiB | 7.87 GiB |
| `DELETE_CANDIDATE` | 1749 | 627.43 GiB | 660.25 GiB |
| `PROTECTED_SKIP` | 835 | — | — |

Empty directories pruned under selected old roots: `449`.

Follow-up empty-root cleanup: the following seven roots were verified to contain no regular files or symlinks and then removed as empty directory trees: `stdloc_lafgs_legacy_archive_20260801`, `stdloc_lafgs_phase2_archive_20260801`, `lafgs_rgb_prior_cache`, `lafgs_query_cache`, `lafgs_v6_feedback_core_20260824`, `lafgs_v6_feedback_control_20260824`, `lafgs_v6_feedback_control_20260825`.

Machine receipt: [`stdloc_cleanup_execution_receipt_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_execution_receipt_20260909_v2.tsv)

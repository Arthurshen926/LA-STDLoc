# STDLoc 存储清理 dry-run 清单

- 生成时间：2026-09-09 14:36（Asia/Shanghai）
- 范围：`/mnt/pool/sqy` 下 STDLoc、AnyGSLoc、LAFGS、G4Splat、Cambridge 和室内数据相关内容。
- 模式：只读盘点；本报告没有执行删除、移动、重命名或覆盖。
- 当前文件系统快照：`/mnt/pool` 总 44T，已用约 42T，可用约 1.7T（97%）。

状态含义：

- `KEEP`：长期保留。
- `HOLD`：当前有运行、可能有引用，暂不处理。
- `ARCHIVE`：保留小型归档后移到冷存储。
- `DELETE-CANDIDATE`：满足前置核验后可删除；本报告不执行。
- `VERIFY`：需要先检查 manifest、hash 或代码引用。

## 1. 当前必须保护的路径

本次 `ps` 快照发现 G4Splat 仍有活动任务，以下目录及其输入不能处理：

- `HOLD` `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/formal_v178_native_optical_resume1k_to3k`
- `HOLD` `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/formal_v163_atomic_measured_split_balanced_3k`
- `HOLD` `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/diagnostic_v186_joint_static_center800_supervision`
- `HOLD` `/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/diagnostic_v187_joint_temporal_center800_supervision`
- `HOLD` 上述任务引用的 warm-start checkpoint、`formal_v113_rigid32k` PLY、evidence store 和 `Cambridge_stdloc/StMarysChurch/processed/masks_with_tree.pkl`。

本次快照没有看到正在运行的 AnyGSLoc 进程，但在实际清理前仍应重新执行一次 `ps` 和引用扫描。

另外两个 AnyGSLoc 当前工作目录先保留：

- `HOLD/VERIFY` `/mnt/pool/sqy/anygsloc_primary_resume_20260909`（约 0.504GB）：包含 office/office2_5a 的恢复结果和部分 projective map。
- `HOLD` `/mnt/pool/sqy/anygsloc_corrected_cambridge_matrix_20260909`（约 16.698GB）：仍是 corrected Cambridge matrix 的输入准备目录，不能当作历史垃圾处理。

## 2. AnyGSLoc 主线：应长期保留

### 2.1 室内 19 个主线先验

- `KEEP` `/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5`
- `KEEP` `/mnt/pool/sqy/datasets/12Scenes_pgt_full_reference_v5`
- 每个场景保留 `lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply` 和 `rgb_prior_manifest.json`。
- 19 个 PLY 合计约 3.661GB；它们是 full-reference SfM 初始化的 vanilla 2DGS，不是 MAtCha。

`/mnt/pool/sqy/indoor_priors_pgt_full_reference_v5` 是导出副本。已对 19 个场景逐一 `cmp`，与上述 datasets 中的 PLY 全部相同：

- `VERIFY` 先把 datasets 路径作为 canonical，并检查 manifest 是否仍指向导出副本。
- `DELETE-CANDIDATE` `/mnt/pool/sqy/indoor_priors_pgt_full_reference_v5` 中的重复 PLY；建议先保留 JSON/日志归档。

另有较旧的 `/mnt/pool/sqy/indoor_priors_pgt_v1`：38 个 PLY 是 19 个场景各自的 `vanilla_2dgs_v1`/`lafgs_prior_v1` 双份，合计约 8.349GB。它不是 full-reference v5 的替代品：

- `VERIFY` 如果仍需复现旧 v1 结果，保留一份树和对应 manifest。
- `ARCHIVE/DELETE-CANDIDATE` 如果旧 v1 不再使用，先归档小型 manifest/日志，再清理重复 PLY 或整个旧 v1 trainer tree。

### 2.2 Cambridge 五场景主线先验

- `KEEP` `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/*/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply`
- 五个最终 PLY 合计约 0.961GB。
- 同时保留对应 `rgb_prior_manifest.json`、hash 和 source-method 说明。

### 2.3 Cambridge 消融先验

保留“最终导出的 PLY + manifest”，不需要保留完整 trainer tree：

- `KEEP` `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/priors/{KingsCollege,OldHospital}/vanilla_2dgs`
- `KEEP` `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/priors/{KingsCollege,OldHospital}/vanilla_3dgs`
- `KEEP/VERIFY` 同目录下 KingsCollege、OldHospital 的 `anysplat_ff`（只在仍需要历史 AnySplat 消融时保留）。
- `KEEP` ShopFacade 的 Long-LRM raw/+1k、i3DGS raw/+1k、MAtCha 7k 最终 PLY：
  `/mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/long_lrm_world_pose32/point_cloud.ply`
  `/mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/long_lrm_globaljoint_d1_ft1000_cpuimages/point_cloud/iteration_1000/point_cloud.ply`
  以及 `ShopFacade/prior/` 下带 `rgb_prior_manifest.json` 的选定 i3DGS/MAtCha 变体。
- `KEEP` `/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908/{KingsCollege_allocator_retry,OldHospital}` 中 corrected vanilla 2DGS 的最终 PLY 与 manifest；其中 `native` 与 `stdloc_model` 是重复导出，见下方 dedup 条目。

当前并不是 Cambridge 五场景的全部变体都已完成。AnySplat 属于 K/Old 历史消融，不应因为目录名存在就把所有 AnySplat 中间结果都当作主线资产。

## 3. AnyGSLoc 可重建缓存和 checkpoint 候选

### 3.1 `rendered_rgb_feature_cache.pt`

这些是由先验高斯、图像/相机和特征模型重新生成的缓存，不是最终先验：

| 路径 | 数量 | 当前大小 | dry-run 状态 |
|---|---:|---:|---|
| `/mnt/pool/sqy/anygsloc_paper_experiments_20260902/**/observations/rendered_rgb_feature_cache.pt` | 24 | 351.945GB | `DELETE-CANDIDATE`，先确认没有任务引用 |
| `/mnt/pool/sqy/anygsloc_fast_prior_20260904/**/observations/rendered_rgb_feature_cache.pt` | 7 | 26.907GB | `ARCHIVE/DELETE-CANDIDATE` |
| `/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908/**/observations/rendered_rgb_feature_cache.pt` | 3 | 35.394GB | `HOLD/VERIFY`，保留 corrected 结果后再处理 |
| `/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/**/observations/rendered_rgb_feature_cache.pt` | 2 | 29.786GB | `ARCHIVE/DELETE-CANDIDATE` |

上述四组 feature cache 合计约 444.032GB，但不代表当前可以立即回收；删除前必须保留输入先验、图像、相机、特征模型版本和 manifest。

最大的单文件包括：

- `12Scenes/office1_gates362/.../rendered_rgb_feature_cache.pt`：38.611GB
- `12Scenes/office1_gates381/.../rendered_rgb_feature_cache.pt`：31.347GB
- `7Scenes/office/.../rendered_rgb_feature_cache.pt`：25.478GB
- KingsCollege 的 vanilla/AnySplat/3DGS cache：约 20GB/个

### 3.2 AnyGSLoc 训练 checkpoint

- `DELETE-CANDIDATE` `/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908/KingsCollege_allocator_retry/native/chkpnt*.pth`：5 个，合计约 2.493GB；最终 PLY 已验证后无需保留全部。
- `VERIFY` `anygsloc_paper_experiments_20260902` 中的 `.pt/.pth` map、association、metric 文件约 21GB；它们多为评估状态而非训练 checkpoint，若要保留精确复现实验，先保留 manifest 和最终结果。
- `ARCHIVE/DELETE-CANDIDATE` fast prior 中的中间 PLY、trainer input 和 events；只保留选定最终 PLY、manifest、配置和 hash。

### 3.3 可去重的 AnyGSLoc PLY

- `VERIFY` `/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908/KingsCollege_allocator_retry/{native,stdloc_model}`
- `VERIFY` `/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908/OldHospital/{native,stdloc_model}`
- 每个场景两份 PLY 内容相同，保留一份并更新 manifest 后，另一份才是 `DELETE-CANDIDATE`。
- Long-LRM raw 与 +1k 目录中存在约 638.6MB 的重复 PLY/input，需先做 hash 比较，不能仅凭文件名删除。

## 4. 闭环反馈历史

如果近期不继续研究，建议只保留 README、结论、最终指标、manifest、resolved config、git commit 和关键 hash：

| 根目录 | 当前大小 | 建议 |
|---|---:|---|
| `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731` | 266.462GB | 保留五个 MAtCha PLY/manifest；`runs/**/query_cache*.pt` 等约 236.388GB 为 `ARCHIVE/DELETE-CANDIDATE` |
| `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802` | 452.337GB | 保留 K/Old 最终消融 PLY；trainer/cache 删除前先拆分保存 priors |
| `/mnt/pool/sqy/lafgs_indoor_pgt_runs_20260804` | 392.043GB | `ARCHIVE`；只保留最终 JSON/metrics/manifest |
| `/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807` | 346.127GB | `ARCHIVE`；大型 bootstrap/query cache 可重建但精确复现成本高 |
| `/mnt/pool/sqy/lafgs_projective_anchor_v4_single_seed_20260818` | 335.793GB | `ARCHIVE`；保留结论性结果，删除旧 feature/appearance cache |
| `/mnt/pool/sqy/G4Splat_runs` | 约 1.42TB | 当前仍有活动任务，先 `HOLD`；活动结束后按 run 选择最终 PLY/metrics，其余 checkpoint/cache 才处理 |

G4Splat 当前粗扫到 1,691 个 `.pt/.pth/checkpoint` 类文件，约 1.027TiB；这个数字包含缓存和状态，不能直接作为“全部 checkpoint 可删除量”。

显式失败的 AnySplat 目录：

- `DELETE-CANDIDATE` `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/priors/OldHospital/anysplat_ff_allviews_flat_failed`：约 6.253GB
- `DELETE-CANDIDATE` `/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/lafgs_strict_v2/anysplat_ff_allviews_flat_failed`：约 31.298GB
- 合计约 37.551GB；建议先保存失败 manifest 和一份简短原因说明。

## 5. 数据源和处理数据

- `KEEP` `/mnt/pool/sqy/Cambridge_stdloc/processed`：当前脚本使用的 canonical Cambridge 图像/相机数据。
- `VERIFY/ARCHIVE` `/mnt/pool/sqy/Cambridge_stdloc/processed_ulfrepro1920`：另一套较大处理数据，先用 `rg` 检查是否仍被配置或 manifest 引用。
- `KEEP` `/mnt/pool/sqy/datasets/7Scenes_raw`、`12Scenes_raw`、full-reference model 和相机/SfM 数据；这是动态重建的基础。
- `ARCHIVE-FIRST` 官方 zip 或重复导出可以在抽取结果和 hash 验证后移到冷存储，但不建议删除唯一源数据。

## 6. 第一批候选回收量（未批准，不执行）

只按“缓存/失败目录/重复 PLY”粗略计算，理论候选约 724GB：

- AnyGSLoc feature cache：约 444.032GB
- frozen Cambridge query/function cache：约 236.388GB
- 显式失败 AnySplat：约 37.551GB
- corrected prior 重复 checkpoint：约 2.493GB
- 室内重复 PLY：约 3.661GB

这些数字包含需要人工确认的项目，不能直接相加后执行；活动任务、精确复现实验和论文证据优先级更高。

## 7. 下一步执行前置条件

1. 重新检查活动进程，确认 `HOLD` 目录已结束。
2. 对每个候选目录保留 `manifest/config/hash/结论摘要`。
3. 对重复 PLY 做全量 `cmp` 或 SHA256，并修正 manifest 引用。
4. 先移动到带日期的归档目录，观察一轮实验和部署是否正常，再考虑永久删除。

## 8. 本次已执行结果（2026-09-09）

- AnySplat：已删除校验到的 39 个目标中的 38 个，另 1 个空目录在前置安全探测中已不存在；预估释放约 282.628GB。`G4Splat_runs` 未触碰。
- 闭环历史第一批：已删除 `stdloc_lafgs_offtheshelf_prior_20260802/lafgs_strict_v2`、其旧 `lafgs` trainer tree、`lafgs_indoor_pgt_runs_20260804` 的 7/12Scenes、`lafgs_adaptive_v3_full_benchmark_20260807` 的 Cambridge/7Scenes、`lafgs_projective_anchor_v4_single_seed_20260818` 的场景树，预估释放约 1.273TB。
- frozen multiscene：保留五个 Cambridge `prior/rgb_matcha_2dgs`，删除其余 runs/function graph/query cache/map state，预估释放约 265.439GB。
- 归档目录：`/mnt/pool/sqy/stdloc_closed_loop_archive_20260909`，已保存本报告、AnySplat receipt，以及五个历史根目录中筛选出的 3,000 余个小型 manifest/JSON/日志文件；复制筛选上限为单文件 50MB。
- 当前 `/mnt/pool` 可用空间约 3.3TB（93% 使用率）。

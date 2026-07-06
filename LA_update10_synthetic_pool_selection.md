# LA_update10: Synthetic Pool Selection and Teacher-Cache Gating

## 背景

WildGaussians synthetic RGB 经过上一轮 QA 后仍不能直接进入默认 student 训练池。主要原因不是单纯肉眼图像可接受, 而是完整 STDLoc teacher 流程中大量 synthetic query 会落到 `mixed_or_uncertain`, `dense_rescues_sparse`, `sparse_failure` 等阶段。为了避免把不稳定 teacher 信号传给 student, 本轮把 synthetic 入口改成候选池流程:

1. 渲染更多 near-train synthetic 候选。
2. 先做 RGB artifact QA。
3. 再跑完整 STDLoc teacher cache。
4. 只允许 `teacher_ok` synthetic 进入训练 manifest。
5. 对进入训练池的 synthetic 按 teacher dense error 排序并 cap 数量。

## 代码改动

- 新增 `scripts/select_pseudo_query_pool.py`
  - 输入 gated manifest 和 teacher cache。
  - 保留全部 accepted `train_rgb`。
  - 对 accepted `synthetic_rgb` 按 `dense_te`, `sparse_te`, `artifact_score` 排序。
  - `--max_synthetic` 控制最多进入训练池的 synthetic 数量。
  - 未入选 synthetic 明确标记为 `synthetic_pool_not_selected`。
  - 输出 `pseudo_queries_selected.jsonl` 和 `pseudo_query_selection_summary.json`。

- 更新 `scripts/run_la_pseudo_query_pipeline.sh`
  - 新增 `SYNTHETIC_CANDIDATE_MULTIPLIER`, 默认 1。
  - `SYNTHETIC_COUNT` 现在表示目标 synthetic 数量, 实际渲染数为:
    `SYNTHETIC_RENDER_COUNT=$((SYNTHETIC_COUNT * SYNTHETIC_CANDIDATE_MULTIPLIER))`
  - 新增 `RUN_PSEUDO_QUERY_SELECT`, 默认 1。
  - 新增 `PSEUDO_QUERY_SELECT_MAX_SYNTHETIC`, 默认等于 `PSEUDO_QUERY_MAX_SYNTHETIC`。
  - 训练默认读取 `pseudo_queries_selected.jsonl`。
  - 新增 `PSEUDO_QUERY_SAMPLING_MODE`, 默认 `source_balanced`, 可显式切换到 `record_proportional`。

- 更新 `la_artifacts/pseudo_query.py` / `train_locaware.py`
  - `PseudoQuerySampler` 保留原 `source_balanced` 行为: 先按 source 权重选 `train_rgb` 或 `synthetic_rgb`, 再在该 source 内均匀采样。
  - 新增 `record_proportional` 行为: 按 record 级别采样, 每条样本权重由 source 权重决定。这样 5 条 synthetic 不会在 500-step 长训中被当成 1/3 query 分布反复过采样。
  - `train_locaware.py` 新增 `--pseudo_query_sampling_mode {source_balanced,record_proportional}`。

- 测试
  - `tests/test_la_artifacts.py` 增加 pool selector 行为测试。
  - `tests/test_la_artifacts.py` 增加 pseudo-query sampler 采样比例测试。
  - `tests/test_full_script_args.py` 增加 pipeline 接线测试。

## ShopFacade 控制实验

共同设置:

- RGB teacher checkpoint:
  `/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/ShopFacade_wg_app_nounc_sky_stopdens7k_15k_960/checkpoint-15000`
- render scale: `0.5`
- appearance strategy: `nearest`
- strict synthetic QA:
  mean <= 0.60, mild frac <= 0.85, severe frac <= 0.58, low-detail mean <= 0.60
- teacher gate:
  `allowed_stages=teacher_ok`, sparse/dense TE <= 100cm

### 1. Mid interpolation + full scale 对照

路径:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha035_065_scale10`

结果:

- 8 synthetic rendered。
- 0/8 通过 QA。
- 平均 artifact mean: 0.7592。
- 结论: 提高 render scale 到 1.0 没有解决 synthetic 质量问题, 反而暴露/放大了低细节伪影。问题不是单纯输出分辨率。

### 2. Near-train 单次 8 张

路径:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha010_025_scale05`

结果:

- 8 synthetic rendered。
- 7/8 通过 QA。
- teacher cache: 1/7 `teacher_ok`。
- strict gated manifest: 1 synthetic + 231 train RGB。
- accepted synthetic:
  - `synthetic/000002.png`: sparse TE 4.220cm, dense TE 4.530cm。

结论:
靠近 train camera 的 synthetic 明显优于 mid interpolation, 但单次随机采样不稳定, 不足以默认训练。

### 3. Candidate pool: alpha=0.10-0.25, 32 候选

路径:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha010_025_scale05_pool32`

结果:

- 32 synthetic rendered。
- 21/32 通过 QA。
- teacher cache stage:
  - `teacher_ok`: 2
  - `mixed_or_uncertain`: 10
  - `dense_rescues_sparse`: 3
  - `dense_improves_sparse`: 2
  - `sparse_failure`: 4
- strict gated/selected: 2 synthetic + 231 train RGB。

accepted synthetic:

- `synthetic/000006.png`: alpha 0.183, artifact 0.511, sparse TE 4.485cm, dense TE 2.673cm。
- `synthetic/000029.png`: alpha 0.144, artifact 0.499, sparse TE 1.363cm, dense TE 1.521cm。

### 4. Candidate pool: alpha=0.02-0.12, 32 候选

路径:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha002_012_scale05_pool32`

结果:

- 32 synthetic rendered。
- 26/32 通过 QA。
- teacher cache stage:
  - `teacher_ok`: 5
  - `mixed_or_uncertain`: 15
  - `dense_improves_sparse`: 3
  - `dense_rescues_sparse`: 1
  - `sparse_failure`: 2
- strict gated/selected: 5 synthetic + 231 train RGB。

accepted synthetic:

- `synthetic/000000.png`: alpha 0.061, artifact 0.574, sparse TE 2.206cm, dense TE 2.485cm。
- `synthetic/000007.png`: alpha 0.086, artifact 0.487, sparse TE 3.005cm, dense TE 3.268cm。
- `synthetic/000012.png`: alpha 0.087, artifact 0.438, sparse TE 3.265cm, dense TE 3.681cm。
- `synthetic/000023.png`: alpha 0.093, artifact 0.516, sparse TE 4.802cm, dense TE 2.513cm。
- `synthetic/000024.png`: alpha 0.039, artifact 0.576, sparse TE 2.628cm, dense TE 2.283cm。

结论:
`alpha=0.02-0.12` 是当前最稳的 ShopFacade synthetic pose preset。它不能证明 WG synthetic 已足够做大范围新视角增强, 但证明了保守 near-train synthetic + QA + teacher-cache gate 可以筛出 teacher 稳定的训练样本。

## 可视化

- 旧 near-train accepted/rejected:
  `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha010_025_scale05/teacher_gate_visuals/contact_sheet_synthetic_rgb_all_dense_te_desc.jpg`

- candidate pool rejected/worst view:
  `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha002_012_scale05_pool32/teacher_gate_visuals/contact_sheet_synthetic_rgb_all_dense_te_desc.jpg`

- candidate pool accepted-only:
  `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_alpha002_012_scale05_pool32/teacher_gate_visuals_accepted/contact_sheet_synthetic_rgb_accepted_dense_te_desc.jpg`

定性观察:

- rejected 样本常见明显 blur, 局部白雾, 路面和高反差边缘的低细节伪影。
- accepted 样本仍有 WG blur, 但 pose 非常接近 train camera, sparse/dense teacher 均稳定。
- 因此当前 synthetic 使用策略应是保守增广, 不是自由新视角生成。

## 100-step Student Smoke

基于 `alpha=0.02-0.12`, pool32 的 selected manifest 做了 ShopFacade 100-step 对照。两组都从同一个
`ShopFacade_baseline` checkpoint 继续训练到 iteration 30100, 并使用 official test sparse-only 评估。

### all train RGB only

模型:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student100_trainrgb_seed0`

评估:
`results/trainrgb-30100-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student100_trainrgb_seed0-20260628_011139/summary.json`

结果:

- sparse median TE: 3.3668 cm
- sparse median AE: 0.1625 deg
- recall_5cm_5d: 72.82%
- recall_2cm_2d: 23.30%
- avg inliers: 428.86

### all train RGB + selected synthetic RGB

模型:
`/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student100_trainrgb_synth_a002_012_pool32_seed0`

评估:
`results/trainrgb-synth-a002-012-30100-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student100_trainrgb_synth_a002_012_pool32_seed0-20260628_011139/summary.json`

结果:

- sparse median TE: 3.3334 cm
- sparse median AE: 0.1580 deg
- recall_5cm_5d: 73.79%
- recall_2cm_2d: 28.16%
- avg inliers: 425.83

对比结论:

- median TE: 3.3668 -> 3.3334 cm, 小幅改善。
- median AE: 0.1625 -> 0.1580 deg, 小幅改善。
- recall_5cm_5d: 72.82% -> 73.79%, +0.97 pp。
- recall_2cm_2d: 23.30% -> 28.16%, +4.85 pp。
- avg inliers 略降, 说明收益不是来自更多 2D-3D inlier, 更可能来自 selected synthetic 对局部定位表征的补充。

这个结果只支持 "strict selected synthetic 可以进入下一轮 500-step/多 seed 验证", 还不能作为最终精度主张。

## 500-step Student 对照与采样混杂

100-step 正向后继续跑 500-step, 发现新的高影响混杂: selected synthetic 只有 5 张, 但旧默认
`source_balanced` + `real:synthetic=2:1` 会让 synthetic 约 33% 的 localization episode 被反复采样。
这对 100-step smoke 尚可, 但 500-step 明显有过采样风险。

采样模拟:

- `source_balanced`, real=2, synthetic=1: synthetic fraction ~= 33.0%。
- `record_proportional`, real=1, synthetic=5: synthetic fraction ~= 9.3%。
- `record_proportional`, real=1, synthetic=2: synthetic fraction ~= 3.8%。

### 500-step 结果

| 配置 | 模型 | median TE (cm) | median AE (deg) | recall_5cm_5d | recall_2cm_2d | avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train RGB only | `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student500_trainrgb_seed0` | 2.9528 | 0.1652 | 75.73% | 27.18% | 491.70 |
| selected synthetic, source-balanced 33% | `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student500_trainrgb_synth_a002_012_pool32_seed0` | 3.2058 | 0.1614 | 72.82% | 22.33% | 455.98 |
| selected synthetic, record-proportional 9% | `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student500_trainrgb_synth_a002_012_pool32_recordprop_seed0` | 3.2105 | 0.1654 | 73.79% | 29.13% | 488.73 |
| selected synthetic, record-proportional 4% | `/mnt/pool/sqy/stdloc_la_wg_stage_opt_v1/ShopFacade_student500_trainrgb_synth_a002_012_pool32_recordprop_s2_seed0` | 3.0752 | 0.1574 | 75.73% | 31.07% | 489.91 |

评估输出:

- train RGB only:
  `results/trainrgb-30500-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student500_trainrgb_seed0-20260628_013335/summary.json`
- source-balanced synthetic:
  `results/trainrgb-synth-a002-012-30500-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student500_trainrgb_synth_a002_012_pool32_seed0-20260628_013422/summary.json`
- record-proportional 9%:
  `results/trainrgb-synth-a002-012-recordprop-30500-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student500_trainrgb_synth_a002_012_pool32_recordprop_seed0-20260628_014656/summary.json`
- record-proportional 4%:
  `results/trainrgb-synth-a002-012-recordprop-s2-30500-_mnt_pool_sqy_stdloc_la_wg_stage_opt_v1_ShopFacade_student500_trainrgb_synth_a002_012_pool32_recordprop_s2_seed0-20260628_015615/summary.json`

结论:

- 33% synthetic source-balanced 长训是负向的, 基本确认 synthetic 过采样是一个真实混杂。
- 9% record-proportional 修复了部分问题, 但 median TE 和 5cm recall 仍不够好。
- 4% record-proportional 是当前最稳设置: 5cm recall 与 train-only 持平, 2cm recall 从 27.18% 提到 31.07%, median AE 从 0.1652 降到 0.1574。median TE 仍从 2.9528 退到 3.0752cm, 所以还不能称为全面胜出。
- 当前更合理的 synthetic 角色不是常规 query 分布, 而是低频、teacher-cache-gated 的局部校正样本。

## 当前结论

1. 用户怀疑的 synthetic render 伪影混杂点是真实存在的。
2. 只靠 RGB artifact QA 不够, teacher-cache gate 是必要的, 因为很多 QA 通过样本仍会导致 teacher stage 不稳定。
3. 直接把所有 synthetic 放进 student 默认训练池不合理。
4. 更保守的 pose interpolation 能显著提高 teacher-ok 产出:
   - `alpha=0.10-0.25`: 2/21 teacher-ok。
   - `alpha=0.02-0.12`: 5/26 teacher-ok。
5. 当前推荐 ShopFacade synthetic v1 preset:
   - `SYNTHETIC_COUNT=8`
   - `SYNTHETIC_CANDIDATE_MULTIPLIER=4`
   - `synthetic_alpha_min=0.02`
   - `synthetic_alpha_max=0.12`
   - `PSEUDO_QUERY_SELECT_MAX_SYNTHETIC=8`
   - strict teacher gate: `teacher_ok` only。
6. ShopFacade 100-step smoke 已经出现小幅正向 sparse-only 信号。
7. ShopFacade 500-step 证明旧 `source_balanced` synthetic 采样会过采样, 负向影响明显。
8. `record_proportional` 低频采样能恢复并增强部分精度指标, 但目前仍是 mixed positive, 需要多 seed/OldHospital 继续验证。

## 未完成项

- 100-step selected synthetic 对照已完成。
- ShopFacade 500-step single-seed 对照已完成, 但还没有多 seed。
- OldHospital 还没有完成同样的 pool preset 搜索。
- 当前 WG synthetic 仍偏糊, 因此不应扩大到更远 pose 区间。

下一步建议:

1. 把默认推荐从 `source_balanced` 切到显式 ablation: `record_proportional`, real=1, synthetic=2。
2. 对 OldHospital 先跑 `alpha=0.02-0.12`, `scale=0.5`, pool32, 验证是否也能产出 teacher-ok。
3. ShopFacade 用 `record_proportional` 低频 synthetic 跑 seed 1/2, 确认 2cm recall 增益是否稳定。

# LA_update9: WildGaussians Synthetic QA + Teacher-Cache Gating

## 目标

本轮不继续只调 WildGaussians appearance，而是把 synthetic query 进入 student 前的质量闭环拆成两道门：

1. **Synthetic QA gate**：基于渲染图的 artifact/low-detail 连续指标筛掉明显不可信 synthetic RGB。
2. **Teacher-cache gate**：只有完整 STDLoc teacher cache 的 sparse/dense stage 可信时，query 才允许进入 student 训练采样池。

## 代码改动

- 新增 `la_artifacts/quality_gate.py`
  - `SyntheticQualityGate`：按 `artifact_score_mean / artifact_mild_frac / artifact_severe_frac / low_detail_mean / p95` 做多指标 QA。
  - `TeacherCacheGate`：按 cache missing、failed flag、sparse/dense TE、stage 白名单做 post-cache gate。
- 新增 `scripts/gate_pseudo_query_manifest.py`
  - 输入 manifest + teacher cache，输出 `pseudo_queries_gated.jsonl` 和 gate summary。
  - 支持 `--teacher_gate_sources`，方便 smoke 只 gate synthetic，全量默认可 gate train+synthetic。
- 更新 `scripts/build_pseudo_query_manifest.py`
  - fresh synthetic manifest 会写入完整 `artifact_summary` 和 `synthetic_quality_gate`，不再只保留单个 `artifact_score`。
- 更新 `scripts/run_la_pseudo_query_pipeline.sh`
  - 默认启用 strict QA + gate manifest。
  - 默认 `PSEUDO_QUERY_TEACHER_ALLOWED_STAGES=teacher_ok`。
  - 训练使用 `pseudo_queries_gated.jsonl`。
- 更新 `train_locaware.py`
  - 直接调用训练脚本时，pseudo-query teacher stage 默认也要求 `teacher_ok`。

## Smoke 结果

### Synthetic QA-only gate

阈值：

- `max_artifact_mean=0.60`
- `max_artifact_mild_frac=0.85`
- `max_artifact_severe_frac=0.58`
- `max_low_detail_mean=0.60`
- `max_artifact_p95` 暂不做硬阈值，因为现有 WildGaussians synthetic 的 p95 普遍接近 0.97，容易清空样本池。

结果：

- ShopFacade nearest WG synthetic：8 -> 4 accepted，4 rejected。
- OldHospital auto WG synthetic：4 -> 3 accepted，1 rejected。

输出：

- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/ShopFacade_nearest/pseudo_query_qa_gate_summary.json`
- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/OldHospital_auto/pseudo_query_qa_gate_summary.json`
- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/ShopFacade_nearest/qa_gate_visuals/contact_sheet_synthetic_rgb_all_artifact_desc.jpg`
- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/OldHospital_auto/qa_gate_visuals/contact_sheet_synthetic_rgb_all_artifact_desc.jpg`

定性观察：

- ShopFacade 被拒样本明显更糊，窗面/街景区域低细节占比更高。
- OldHospital 被拒样本有明显建筑边缘拖影和天空/立面低细节区域。

### ShopFacade teacher-cache smoke

只对 QA 后 accepted 的 4 个 ShopFacade synthetic 跑完整 STDLoc teacher cache。

第一次运行失败原因：

- gsplat JIT 使用了错误的 nvcc 路径 `/root/miniconda3/envs/iclpose/bin/nvcc`，报 `cuda_runtime.h` 缺失。

修复方式：

- 按 pipeline 显式设置 `CUDA_HOME=/usr/local/cuda-11.8`、`PATH`、`LD_LIBRARY_PATH` 后重跑成功。

teacher cache stage：

- `mixed_or_uncertain`: 3
- `dense_improves_sparse`: 1
- `teacher_ok`: 0

逐样本：

- `synthetic/000000.png`: sparse TE 5.816 cm, dense TE 2.747 cm, stage `mixed_or_uncertain`
- `synthetic/000003.png`: sparse TE 5.263 cm, dense TE 2.143 cm, stage `mixed_or_uncertain`
- `synthetic/000005.png`: sparse TE 16.180 cm, dense TE 12.079 cm, stage `mixed_or_uncertain`
- `synthetic/000007.png`: sparse TE 9.343 cm, dense TE 2.504 cm, stage `dense_improves_sparse`

strict `teacher_ok` gate 结果：

- ShopFacade QA accepted synthetic：4 -> 0 accepted。
- train RGB 在 synthetic-only smoke 中不参与 teacher gate；全量 pipeline 默认会缓存并 gate train+synthetic。

输出：

- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/ShopFacade_nearest/pseudo_teacher_cache_smoke_summary.json`
- `/mnt/pool/sqy/stdloc_la_wg_synth_quality_v1/ShopFacade_nearest/pseudo_query_teacher_gate_smoke_summary.json`

## 当前结论

1. Synthetic QA gate 是必要的，能自动筛掉一部分明显糊/低细节/伪影占比高的 WildGaussians synthetic。
2. 仅靠渲染 QA 不够。ShopFacade smoke 里，画面 QA 通过的 synthetic 在完整 STDLoc teacher 后仍没有 `teacher_ok`，严格 gate 会全部拒绝。
3. 这支持下一步策略：默认 student 训练不应直接使用 WG synthetic，除非通过 teacher-cache gate；先用 all-train RGB 作为稳态训练池，synthetic 作为单独 ablation。
4. 若要继续使用 synthetic，需要单独验证更宽松 stage 白名单，例如 `teacher_ok,dense_improves_sparse`，但不能作为默认主线，因为当前 student 的 sparse-only 训练仍会受 sparse init 质量影响。

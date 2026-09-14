# Cambridge 扩展实验队列（2026-09-14）

执行根目录：`/mnt/pool/sqy/anygsloc_cambridge_suite_v2_20260914`。
实际命令与依赖：`gpu0/jobs.json`；分组清单：`registry.json`；进程状态：`gpu0/state.json`。
331 个执行步骤，不等于 331 次定位评估。以新协议单 seed 2026 执行，查询端无语义先验。

## 已接入的范围

| 组别 | 条件数 | 执行内容 |
|---|---:|---|
| 多先验 | 30 | 五场景×六替代先验；已有紧凑地图复用，否则从已完成 Full 地图压缩，再重评 |
| 核心及附加消融 | 73 | 原三份消融清单的 seed2026；关联/几何、V2、选择器、同 ID 深度对照、描述子/completion |
| STDLoc / ULF-Loc | 20 | 两方法×五场景×建图语义 on/off；每次均重新建图，随后输出 coarse 与 fine |
| AnyGSLoc 转换成本 | 5 | 给定现成 MAtCHA 先验，完整转换链路重新测量分阶段时间、进程树 RSS |
| AnyGSLoc 部署基准 | 5 | 每场景 50 次预热、3 轮完整 test，记录核心时间与图像/几何处理在内的时间 |
| 外部部署基准 | 20 | 各建图条件 50 次预热、3 轮 native 两阶段评估，另存结果不覆盖精度评估 |
| 存储 | 25 | 本方法5份部署地图；两基线20个模型目录与共享特征权重的资产清单 |

MAtCHA 的五个主结果由24场景队列提供，不再重复。所有后续组仅 Cambridge。
当前使用 GPU0，依赖24场景队列结束后接续；不抢占 GPU1/2 上其它任务。
耗时基准前设 GPU 空闲等待，非计时精度结果不充当正式性能基准。

## 基线实现边界

STDLoc 固定官方 `upstream/main` 的本地已获取 commit；ULF-Loc 固定已完成过历史复现的
`/tmp/ULF-Loc-repro-b28d532` commit。具体 SHA 在 `baseline_commits.json`。
ULF-Loc 不是毫无改动的原始发布文件，包含已记录的兼容性修复与计时；不冒称 bitwise 原版。
原生代码不改写定位方法，新 wrapper 仅处理 seed、语义/几何掩码和计时。
两套原生 train/evaluate 入口及几何适配器已在 `ulfloc_repro` 环境通过导入预检；
完整 GPU 建图及两阶段准确性需等待实际任务完成，导入成功不算复现成功。

## 成本比较不可混淆的边界

- 存储清单可直接统计，不需要重新定位。外部模型目录含日志等，先标记为目录总量，
  不把它冒充最小部署包；须分离 checkpoint/地标/检测器/共享权重后再作最终部署对比。
- 基线 fresh training 自动记录总时间；ULF-Loc 另测 sampling/fusion，STDLoc 另测 detector。
  STDLoc 的 Gaussian/feature 联合训练无法直接解释成“给定现成 RGB Gaussian 后的纯转换成本”。
- AnyGSLoc 转换成本不含上游 Gaussian 训练；必须与基线分阶段成本并列说明，
  不拿增量成本对比对方全链路训练后声称绝对加速倍数。
- 本轮基线建图读取既有语义结果，不重新执行分割网络；生成成本未知时记 NR，不能填0。
- coarse 计时包含原生两阶段流水线到 coarse 结束的前缀；ULF-Loc 原生记录还提供
  sparse-only 时间。二者不要混淆。fine 同时记录增量与 coarse+fine 累计时间。
- query 图像加载/有效区域处理与模型初始化必须独立说明。GPU 空闲等待不保证 CPU
  在后续整段运行中无外部竞争；最终计时仍须审计，不能仅因有 warmup 就宣称严格隔离。

## V2 不变

此次不再用 RGB 非黑判断的是 query 的去畸变有效区域；V2 的渲染质量检测未删除。
V2 中现有的 alpha、深度、结构与黑洞等诊断保持原样；不能声称整套项目完全不再检查暗像素。

## 报表

`tools/report_cambridge_suite_v2.py --root <上述执行根目录> --watch` 可生成
`accuracy.csv`、`costs.csv`、`report_status.json`。外部 coarse/fine 分行；
无新内点但保留有效旧位姿的 refinement fallback 不错误计成失败。
缺文件、失败或未执行保持独立状态，不进入完整结果计数。

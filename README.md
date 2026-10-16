# AnyGSLoc

> **最新论文与开源复现入口（2026-10-15）**：请先阅读 [release/AnyGSLoc/README.zh-CN.md](release/AnyGSLoc/README.zh-CN.md)。其中已集中整理当前 TRO 论文的代码、数据配置、22 张表的复现步骤和九张图的素材。下文保留原工作区的冻结版本说明；当前论文复现命令应在 `release/AnyGSLoc/` 内执行。

> **当前编辑稿（2026-10-16）**：[editorial r3](paper/AnyGSLoc_TRO_20261016_editorial_r3/README.md)，正文源为 [main.tex](paper/AnyGSLoc_TRO_20261016_editorial_r3/main.tex)。这是活跃编辑稿，不代表其新增表述已完成全部科学一致性验收。复现候选仍以 `release/AnyGSLoc/` 的记录和验证边界为准。历史稿、本机资产与编译输出保留本地，不纳入本次 Git 整理；旧补充稿未作为 r3 的可构建附件发布。

本仓库 `/root/STDLoc` 是唯一项目。`anygsloc/` 是整仓主线代码的 Python 命名空间，**不是另一个项目或仅用于部署的旁支**。
当前分支 `codex/anygsloc-repository-refactor` 统一离线构图、在线定位、评估；历史开发代码实际迁入 `research/legacy/`。

## 冻结主线

```text
Gaussian + mapping cameras
  → rendered RGB/alpha/depth → 观测有效性 → track关联
  → 射线三角化 + 单描述子 → 完整紧凑选择 → 部署地图

query RGB + calibration
  → SuperPoint → exact cosine Top-1 → 65%对应 → 一次PoseLib
```

没有离线自定位反馈、描述子训练、在线渲染、密集匹配、二阶段refinement或地图写回。
Fast采用confidence=.999、min250、max30000，至少保留256对应（不超过原有数量）。
source-image-free指下游转换/部署，不表示Gaussian重建不需要RGB或mapping相机。

## 仓库结构

```text
anygsloc/
  cli.py                  统一命令入口
  config.py               在线冻结设置
  artifacts.py            部署地图合同与导出
  mapping/                渲染观测、关联、几何、描述子、选择器、构图调度
  online/                 特征到匹配到一次PnP的运行时
  data/                   相机/图像域、数据集、mapping-only标定
  features/               冻结SuperPoint与采样
  priors/                 Gaussian读取和渲染（仅离线）
  evaluation/             冻结配置的独立评估与指标
  utils/                  哈希等基础工具
tests/                    主线测试与显式历史等价对照
docs/                     当前架构、迁移说明和投稿计划
research/legacy/          旧脚本、反馈/训练代码、配置、报告和研究测试
tools/                    迁移工具，不参与算法运行
```

正式代码不导入research/legacy，不依赖已迁走的根目录scripts、map_learning等。
旧lafgs产物schema保留兼容，不为改名破坏已有地图/缓存。

## 安装

```bash
pip install .
pip install '.[mapping,test]' # 构图/测试时追加依赖
python -m anygsloc --help
```

wheel只包含anygsloc及子包；离线代码也随包提供，但在线不会导入renderer等重依赖。
预置SuperPoint权重，兼容变量 `LAFGS_SUPERPOINT_WEIGHTS`；首次下载/初始化不计入稳态计时。

## 统一命令

先用dry-run查看合同和计划（不创建结果目录，不开始构图）：

```bash
python -m anygsloc build-map --dataset /absolute/scene \
  --prior-manifest /absolute/rgb_prior_manifest.json \
  --image-domain rectified --output /absolute/new_build --gpu 0 --dry-run
```

确认输入域正确后去掉dry-run才执行；构图不会顺便跑test评估。
`--resume`只复用合同和输出哈希匹配的已完成阶段；部分/篡改产物不会被当成完成或自动覆盖。
已有Full地图可跳过重建直接选择；已有Compact可直接导出：

```bash
python -m anygsloc compact-map --map-dir /absolute/full/projective_map \
  --cache /absolute/observations/rendered_rgb_feature_cache.pt --output /absolute/new_compact
python -m anygsloc export-map --source /absolute/compact/projective_anchor_map.pt \
  --metric /absolute/compact/identity_metric.pt --output /absolute/deployment_map.pt
```

单图定位：

```bash
python -m anygsloc localize --map /absolute/deployment_map.pt \
  --query /absolute/query.pt --keypoints 2048 --reprojection-px 12 \
  --mask-policy none --device cuda:0 --output /absolute/pose.json
```

2048/12只是示例，须换成冻结的场景解析值。query.pt为受信任字典：RGB float32 `[3,H,W]`、`fov_x/fov_y`（弧度），可带二维valid_mask。
接口要求已正确矫正、中心主点的pinhole图像，不会自动再次去畸变。
使用mask时指定explicit_valid_mask；不能声称历史实验没用mask。
批量调用复用 `anygsloc.online.runtime.Localizer`，一个实例串行使用。
CLI固定PyTorch CPU线程4，宿主调用需自行设置。

评估单独执行，使用mapping相机解析query密度/重投影阈值，不从test调参：

```bash
python -m anygsloc evaluate --dataset /absolute/scene --map /absolute/deployment_map.pt \
  --image-domain rectified --mask-policy explicit_valid_mask \
  --output /absolute/new_evaluation --dry-run
```

普通评估耗时是telemetry，不是预热、独占硬件下的正式实时性报告。
输出默认拒绝覆盖；失败进入召回分母，不能把占位位姿当成功定位。

## 验证与文档

```bash
python -m pytest tests -q
```

- [整仓架构与模块边界](docs/architecture.md)
- [迁移清单、回归和剩余门槛](docs/repository_migration.md)
- [冻结方法合同](docs/release/method_and_contract.md)
- [投稿实验计划：尚未启动](docs/release/journal_experiment_plan.md)
- [已有证据与结论边界](docs/release/evidence_status.md)

此次不新增方法、不重新跑正式实验。目录和依赖迁移不等于全场景数值等价或速度提升已证明。
旧LaFGS的CITATION元数据已归档；正式论文作者/题名确认后再生成新引用信息，不把旧论文元数据作为AnyGSLoc引用。

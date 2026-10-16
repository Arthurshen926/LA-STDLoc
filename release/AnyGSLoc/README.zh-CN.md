# AnyGSLoc 最新论文主线：代码与复现实验入口

本目录是面向开源整理的候选版本，对应 **2026-10-13 的实验结果、2026-10-14 的图件**。原实验目录及其哈希绑定保留在原位置；这里集中维护可迁移的代码、配置、实验说明和结果核对入口。运行前请先进入本目录，避免调用外层工作区的旧入口。

## 从哪里开始

- [完整复现实验指南](docs/reproduction.md)：逐项对应主文 9 张表和补充材料 13 张表。
- [环境安装](docs/installation.md)、[数据与模型清单](docs/data.md)。
- [多 Conda 环境的检查、启动与保存](docs/environments.md)：主线和原生定位对比使用两个环境；上游先验生成另有三个环境，可归档并在需要时恢复。
- [算法与代码对应关系](docs/method.md)。
- [STDLoc / ULF-Loc 原生复现](docs/baselines.md)、[高斯先验与 PSNR](docs/priors.md)。
- [九张图的原稿及重绘方法](docs/figures.md)。
- [本次实际验证范围](docs/validation.md)。

先做最轻量的核验，无需 GPU、数据集或模型：

```bash
python -m pip install numpy==1.26.4
python -m reproduction.aggregate --check --output runs/reference_report
python -m reproduction.verify_tables --metrics runs/reference_report/metrics.json
```

这会从随代码提供的逐查询、逐帧记录重新计算指标，并与表稿逐项核对。已覆盖 244 个场景/配置/输出单元、35 个场景/先验 PSNR 单元和 90 次正式计时进程。它与重新运行定位是两种不同的复现层次。

配置好数据和模型后，可以重新定位，或从先验重新构图：

```bash
export ANYGSLOC_ASSETS="$PWD/assets"
export ANYGSLOC_SUPERPOINT_WEIGHTS="$ANYGSLOC_ASSETS/weights/superpoint_v1.pth"
python -m anygsloc reproduce evaluate --config configs/cambridge.json \
  --scene ShopFacade --map compact --output runs/shop_eval
python -m reproduction.run main --config configs/cambridge.json \
  --scene ShopFacade --output runs/shop_rebuild --execute
```

不加 `--execute` 时只输出任务计划。后续扩展到所有场景时去掉 `--scene`。正式计时需每次只运行一个场景、一轮，保证使用独立进程。

需要特别区分的口径：Cambridge 主结果是 vanilla 3DGS 的 Full；T2/T3 室内主结果是 2DGS 的 Compact。主文过滤/融合表是 Compact，Full 对应补充材料。PSNR 使用完整 mapping/reference 集，不是 test 集。基线初始/精化时间是同一次完整调用中的累计时间。其他方法的文献横评数字保留引用，不冒充本地重新训练的结果。

大型资产的公共下载尚未发布。当前服务器上的对应关系在同级 `../local/asset_bindings.json`，可用 `reproduction.assets link` 建立本地布局；源码包不携带私人绝对路径、数据集、原生基线仓库或大模型。正式开源前仍需把清单对应的资产上传到公开存储，并填写下载地址。

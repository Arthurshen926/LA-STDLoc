# AnyGSLoc 最新论文主线交接

当前可继续维护、测试和打包的源码在 [AnyGSLoc/](AnyGSLoc/README.zh-CN.md)。版本为 `0.4.0`，对应 2026-10-13 实验与 2026-10-14 图件。当前活跃论文编辑稿为 [editorial r3](../paper/AnyGSLoc_TRO_20261016_editorial_r3/README.md)，正文入口为 [main.tex](../paper/AnyGSLoc_TRO_20261016_editorial_r3/main.tex)。其编辑状态不替代本目录实验验证范围；2026-10-13 的旧正文与补充稿保留在本机历史目录，未作为 r3 的可构建补充附件发布。

- `AnyGSLoc/`：开源候选代码、逐实验文档、配置、参考记录、表稿与九张 SVG。
- `local/asset_bindings.json`：本服务器资产的路径映射；不进入源码包。
- `local/environment_snapshots_20261015/`：五个论文 Conda 环境清理前的完整 Conda/pip 清单与本地恢复记录；多环境启动和检查方式见 `AnyGSLoc/docs/environments.md`。
- `local/conda_cleanup_20261015/README.md`：环境清理结果、MoGeAlign 依赖核查与完整二进制备份的恢复入口。主线 `g4splat`、原生对比 `ulfloc_repro` 仍在；三个上游环境已归档，需要生成新先验时先恢复。
- `local/figure_inputs/`：三张定性实验图的显示输入，可单独打包。
- `maintainer/`：本次从实验目录收集资料、清理记录和打包的服务器工具。收集脚本是本次迁移记录，后续修改已在候选源码中继续完成；不要直接重跑收集脚本覆盖现有源码。这七个历史脚本依赖服务器绝对路径或固定的本地归档/安装目录，已逐文件忽略并保留本地；它们不是候选包的运行、表格重生成或测试依赖。后续若需要可移植的发布维护工具，应先参数化输入和输出并单独验收。
- `dist/`：源码 ZIP、可选图件输入 ZIP、Python wheel 与归档哈希。
- `AnyGSLoc/checks/`：实际重建、定位、PSNR、基线、计时与安装检查输出，不进入源码包。小型核验结论已收录在 `AnyGSLoc/reproduction/validation/`。

本机开始使用：

```bash
cd /root/STDLoc/release/AnyGSLoc
python -m reproduction.assets link --bindings ../local/asset_bindings.json --root assets
python -m reproduction.assets check --root assets
```

运行环境见 `AnyGSLoc/docs/installation.md`；本机对应环境为 `g4splat`，原生基线另用 `ulfloc_repro`。每个完整实验的命令、配置及表格对应关系见 `AnyGSLoc/docs/reproduction.md`。

正式公开前需要上传清单中允许分发的数据/模型资产，并在 `reproduction/assets.json` 中补齐公共获取地址。当前仅准备本地候选包，尚未推送或发布。全场景重新运行的结果应放在新的 `runs/` 目录，并通过新结果汇总入口检查；不要修改随包保留的论文参考记录。

# 发布候选重构与归档回执

此页记录上一轮部署包整理，不代表最新整仓状态。后续整体迁移见 `../repository_migration.md`。

日期：2026-09-13；基线git：`bd144a4`；此次修改未自动commit或发布到外部。
范围：冻结 Compact + Fast 的发布入口、部署资产格式、代码/文档隔离、实验计划；**没有启动新性能实验**。

## 已实现

- 新 `anygsloc/` 包：固定设置、精简matcher/PoseLib/runtime、最小地图校验/导出、独立CLI。
- `localize` 没有训练、renderer、query refinement、GT选择或地图更新依赖。
- `compact-map` 从明确指定的 Full地图/缓存运行既有完整选择器，不再读取历史队列生成正式入口。
- 离线构图内核保持研究树兼容，不宣称已经独立发布或重新验证所有场景；原始输入仍需正确的数据域。
- wheel仅包含 `anygsloc`、`features` 和包元数据；mapping额外依赖分离；保留原 `requirements.txt` 为研究环境。
- README指向新的唯一发布协议；旧Full/Standard YAML保留用于历史重现，原始数值不改。
- 新 `docs/release/` 梳理方法、已知证据/限制、完整投稿实验计划；旧消融进度标记纠正为219/219。

## 回归结果

命令（源码根目录、g4splat环境）：

```bash
python -m ruff check anygsloc tests/test_release_runtime.py
python -m pytest tests/test_release_runtime.py tests/test_metric_and_matcher.py \
  tests/test_sufficiency_selector.py tests/test_group_pose_solver.py \
  tests/test_render_valid_superpoint.py -q
```

结果：lint通过；69 passed，其中新release测试16项。仅TypedStorage弃用警告。
覆盖：空/小/大匹配集、跨chunk同分、65%保留顺序、PoseLib合成数据等价、mask/归一化、
地图tensor精确导出、拒绝错误hash/ID/identity/测试数据来源、运行时导入隔离。

真实前端冒烟使用现有 ShopFacade Compact地图（12,520点），仅一张mapping图像
`seq2/frame00001.png`、实际mask、CUDA0、同一1024关键点/12px设置作新旧代码等价检查。
这些值是回归参数，不是该场景正式标定，也没有计算/汇报测试定位精度。
关键点、描述子、匹配query/anchor索引、匹配分数、PnP inliers、最终pose七项均逐值相等。
此测试读取真实mapping图像仅作软件回归；没有用它重新建图、训练或改变地图，不计入source-image-free方法评估。

冒烟实际导出的最小地图：13,072,582 bytes；tensor payload 13,070,880 bytes。
SHA256：`4e678c81d60d4a640d3630e8f5e95cad720bf27c1b111affcd4d88c9a80dd63a`。
临时位置：`/tmp/anygsloc-release-regression-9l4n68of/deployment_map.pt`，不是正式存储实验或长期发布资产。
权重、相机、运行缓存不包含在这个map byte数字内；不由一个例子推出全部场景的压缩比例。

隔离部署冒烟：Python `-I`、工作目录 `/tmp`，从wheel导入，现有预置SuperPoint权重 + 导出地图 + 合成RGB。
成功返回pose结构；进程未导入 `map_learning/priors/topology/gsplat/evaluation/localization/data/evidence`。
这验证离线Python模块不依赖，不代表操作系统文件访问已经在容器级白名单沙箱下审计。
空白合成图像返回求解器结果不等于定位正确，本测试只检查运行链路。

## 回归发现并修复的本轮问题

1. 旧Fast在地图加载时做两次归一化。最初新runtime少一次，索引/pose相等但cosine分数不逐值相等。
   已恢复第二次归一化并加入回归；不能把这件事解释成旧实验已有精度bug。
2. 老 `build/lib` 有历史研究模块，第一次wheel误打包进去。已用独立build目录隔离，并检查最终wheel根模块白名单。
   初次wheel仅归档为诊断，不可用作部署包。

最终候选wheel目录：`/mnt/pool/sqy/anygsloc_release_archive_20260913/release_candidate/`。
文件：`anygsloc-0.2.0-py3-none-any.whl`，34,971 bytes（不含模型权重）。
SHA256：`5b66f3844b3ae24ed8ed41b23d88104636db067700239f6e8abe6bc770308fa1`。
最终wheel内所有 `anygsloc/*.py` 与当前源码逐byte一致；旧YAML解析后的配置与重构前相等。
`wheels/` 是首次污染打包，`validated_wheels/` 是中间隔离验证版本；两者不是最终发布目录。
本轮没有上传PyPI或修改系统已安装包。

## 归档与可恢复性

完整重构前快照与SHA见 `research/README.md`。移出了旧README和两份无入站引用的历史报告。
历史代码从部署包排除、保留源码路径和完整快照；有传递依赖的离线代码不批量移动，避免破坏旧产物复现。
未删除任何Gaussian、cache、dataset或实验结果；没有对 `/mnt/pool/sqy` 进行大文件清理。

## 尚未通过的发布门槛

- 新 `compact-map` 的全规模输出与历史selector逐行等价；本轮保留算法调用顺序，不冒充已经全量验证。
- 大小场景完整query集的新旧runtime等价；当前仅单元/单图冒烟。
- 同一发布配置24场景正式结果、外部同机比较、独占计时与完整部署内存统计。
- 原始相机到准备好的query tensor的独立发布适配/文件白名单测试。
- 依赖精确lock与跨机器复现；pyproject版本范围不是正式实验环境锁文件。

这些已纳入实验计划，不通过时不把新release直接替换论文最终结果来源。本轮不继续扩展模块或启动实验。

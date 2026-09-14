# 发布实验执行记录（2026-09-13）

用户已授权执行。未新增传统基线实现，未重新训练 MAtCHA 或 AnySplat。

## 已启动

- E1：24 场景 × 3 seed = 72 次评估，GPU 0/1/2 三条持久队列。
  `/mnt/pool/sqy/anygsloc_release_validation_20260913`
- E3：30 个替代先验地图 × 3 seed = 90 次评估，所有地图和观测缓存路径存在；
  `/mnt/pool/sqy/anygsloc_release_priors_20260913`。
  协调进程等待 E1 三条队列结束后启动，不与同卡定位进程争用。
  E3 中 MAtCHA 的 15 次评估直接复用 E1，不重复跑。
- 室内主线使用 Vanilla2DGS，而非 MAtCHA；室内无语义 mask，Cambridge 显式使用 mask。
- Cambridge 复用冻结紧凑地图；室内和替代先验从已完成 Full 地图做冻结 selector，再评估。
- 每张地图仅选择/导出一次。代码快照在实验根目录 `code/`，依赖/GPU 环境在 `environment.json`。

## 已通过的检查

- 原有 77 项软件测试通过；新增 5 项报告完整性测试通过。
- ShopFacade 103 张、StMarysChurch 530 张：新旧抽取计算模块的特征、匹配、内点、位姿完全一致。
  该测试共用发布调度组合，不冒称独立历史 driver 复现。
- 首个可用历史 driver 全集对照：GreatCourt seed2026，760/760 张图顺序与位姿逐元素完全一致。
- 首批已产生完整评估结果，完整图像清单/分母审计通过；运行数量不断变化，查询队列状态而非本文数字。

## 还没有完成的部分

162 次评估并非全部完成。E2 官方基线同机复现、E4 历史消融逐项合同核验、
E5 隔离预热延迟/RAM/VRAM、E6 完整构建成本仍待执行/补证。
目前不能把并行评估的耗时作为正式实时性结论，也不能声称全部输入/历史 driver 审计完成。
官方基线阶段及成本口径见 `baseline_stage_and_cost_audit.md`，包括语义先验。
本轮测试调参历史仍需披露；新增 seed 不构成独立验证集。

## 状态与恢复

各根目录 `gpu*/state.json` 有作业状态、PID、进程身份和 heartbeat；`*.log` 有子进程日志。
遇到失败/孤儿进程保持产物并标记，不把 partial 记成 complete。
已启动后不重新运行 prepare 覆盖队列；修复后需显式审计并恢复对应 lane。

更新结果清单：

```bash
/root/miniconda3/envs/g4splat/bin/python tools/report_release_validation.py --root /mnt/pool/sqy/anygsloc_release_validation_20260913
```

输出 `results.csv` 和 `report_status.json`；报告明确标注当前计时非正式 benchmark。

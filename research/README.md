# 历史开发归档

当前代码只认根目录README、anygsloc及docs。legacy是实际迁移的历史项目树，不进入发布包，也不供正式主线导入。

- legacy/scripts：历史脚本，不是当前入口。
- legacy/map_learning、localization、topology等：旧反馈、训练、refinement及对应实现。
- legacy/configs：历史Full/Standard配置与队列合同，数值不改。
- legacy/docs、paper等：开发报告/论文材料，含已失效或被推翻的记录。
- legacy/tests：完整历史测试；相关主线测试经命名空间迁移另置根目录tests。
- migration_symbols.json：38个原模块→新模块及保留符号、迁移前SHA。
- repository_moves.json：根目录迁移清单。
- development：上一轮已移出的旧README和报告。

必要时独立启动历史环境，不向主线全局注入旧模块：

```bash
cd /root/STDLoc/research/legacy
PYTHONPATH=/root/STDLoc/research/legacy python -m scripts.evaluate --help
```

历史脚本含机器绝对路径，迁移不保证每个旧队列可原样重跑。优先使用各实验自己的代码快照和合同。
旧文档中的绝对路径仍是历史记录，不自动改写成新实验来源。

## 可恢复快照

最早干净基线bd144a4：
`/mnt/pool/sqy/anygsloc_release_archive_20260913/pre_refactor_bd144a4.tar.gz`
SHA256 `a7dd2206226ea2e20623246006c9682b4cd73d1cd114c35468be21b501d04544`。

本轮迁移前工作树检查点（含上一轮未提交内容及已抽取、尚未移动的内核）：
`/mnt/pool/sqy/anygsloc_release_archive_20260913/repository_migration_checkpoint.tar.gz`
SHA256 `8037db2dffb3ceef86b91c35ce8b2eae997ee57c9639c49dcbbe7f418933d336`。
不包含git数据库、build和Python缓存；先验和实验产物仍在原数据盘。

第一次repository_refactor_before_worktree.tar.gz并发归档时tar报告目录变化，不作为完整恢复依据。
使用上述校验通过的快照解压到新目录，不覆盖当前树。
没有删除任何Gaussian、数据集、观测缓存或实验结果。

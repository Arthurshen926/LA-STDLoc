## 核心判断

**方法已经出现真实的正向证据，但目前只证明了其中一个子命题，还没有完成最初设想的完整 LA-STDLoc。**

根据你给出的同阈值结果，提升并不只是来自 RANSAC 调参：

| 公平比较                       |    AE 变化 |   TE 变化 | 5cm/5deg | 2cm/2deg |
| -------------------------- | -------: | ------: | -------: | -------: |
| v0.2 12px vs baseline 12px | 降低 17.6% | 降低 9.3% |   -1.0pp |   +4.9pp |
| v0.2 6px vs baseline 6px   |  降低 5.5% | 降低 6.2% |   +3.0pp |   +5.9pp |

因此更准确的结论是：

> **RANSAC 6px 放大了 LA descriptor 的收益，但不是收益的唯一来源。**

12px 下 LA 已经明显改善中位误差和 2cm recall；6px 下，在同一求解参数下四项指标全部优于 baseline。这已经足以支持“多视图定位监督可以改善 sparse Gaussian descriptor”。

但当前验证的是：

[
\boxed{\text{GT-projected multi-view landmark descriptor distillation}}
]

而不是完整的：

[
\boxed{\text{dense localization teacher}
\rightarrow
\text{per-Gaussian credit assignment}
\rightarrow
\text{topology-aware sparse map}}
]

所以仍有相当大的方法提升空间。

---

# 一、当前方法究竟完善到了哪里

## 已经比较可靠的部分

当前 feature phase 确实只开放了 `loc_feature`，`use_loc_opacity` 默认关闭，因此 v0.2 的主结果基本实现了 geometry、landmark、detector 不变，只更新 Gaussian descriptor 的受控实验。([GitHub][1])

Direct teacher 也已经完成了几个关键闭环：

* 将固定 landmark center 投影到 query；
* 通过 target depth 和 alpha 做可见性过滤；
* 在 GT 投影位置采样 query descriptor；
* 用 query observation 更新 prototype；
* 将统计直接写回明确的 Gaussian index。([GitHub][2])

因此当前的正向结果不是随机 pipeline 偶然跑通，而是有明确机制的：

> 多视图 query observation 使 landmark descriptor 更接近跨视角稳定表征。

---

# 二、当前最主要的方法缺口

## 1. Direct teacher 还不是 dense localization teacher

训练代码虽然生成了 `pose_init`，但 direct 分支实际上只把 `pose_gt` 传给 `direct_landmark_teacher()`；`pose_init` 并未参与 direct matching 或监督生成。([GitHub][1])

所以当前成功部分没有验证：

* sparse 初始位姿是否被 dense stage 修正；
* dense stage 找到的对应能否蒸馏给 sparse landmarks；
* 定位失败或 pose improvement 能否反馈到 Gaussian map。

这不是缺陷，而是一个合理的阶段性简化。但论文层面的核心贡献还需要把 dense teacher 接回来。

---

## 2. 训练目标与 sparse inference 仍不完全一致

当前 direct loss 本质上是：

[
\mathcal L_{\text{direct}}
==========================

1-\cos(f_i,q_i).
]

multiview contrastive 的负样本主要来自当前 query 中共同可见的 landmark，并通过 2D 距离排除近邻。([GitHub][2])

而 sparse inference 面对的是：

* 全部 16384 个 landmark；
* detector 产生的非 GT keypoints；
* MNN 或 top-k 离散匹配；
* 外观重复但空间相距很远的 hard negatives；
* 最终 PnP 对应集合。

因此目前优化得更多的是 **invariance**，但对以下两个量约束不足：

1. 全地图范围的 **distinctiveness**；
2. 对 PnP 有效的 **geometric conditioning**。

这能解释你观察到的现象：

> 匹配和 inlier 可能变多，但新增对应集中在同一平面、同一图像区域或具有系统性偏差，最终把 pose 拉偏。

---

## 3. 当前 utility 的统计语义仍然较弱

Direct teacher 目前定义：

[
p_i=\frac{\cos(f_i,q_i)+1}{2},
]

然后使用：

* `positive_prob = p_i`；
* `entropy = -p_i\log p_i`；
* `repeatability = 1[p_i>0.25]`；
* `reproj_error = 0`；
* `information = 0`。([GitHub][2])

这里有几个问题：

* (p_i) 只是 cosine 的线性变换，不是真正匹配概率；
* entropy 不是完整候选分布的熵；
* (p_i>0.25) 等价于 cosine (>-0.5)，通常过于宽松，repeatability 容易饱和；
* 没有实际预测匹配的重投影误差；
* 没有 pose information；
* 没有真实 PnP inlier/outlier 反馈。

因此当前 utility Spearman 只有约 0.106 是符合预期的。现阶段它不应该控制 topology，也不适合大权重参与 landmark sampling。

---

## 4. 现有 descriptor diagnostics 仍偏乐观

目前 `descriptor_alignment_metrics()` 计算的是同一个 query 中成对可见 descriptor 的 (N\times N) 相似度矩阵，并将对角线作为正确匹配。它没有把每个 query descriptor 与完整 16384-landmark bank 比较。([GitHub][3])

这意味着当前 Level-1 指标回答的是：

> 在当前选出来的共同可见 landmark 子集中，正确 pair 能否赢过其他 pair？

而实际 sparse localization 的问题是：

> 在完整地图、重复结构和 detector 噪声存在时，正确 Gaussian 能否成为全局 top-1/MNN？

下一版 diagnostics 必须改成 full-bank。

---

## 5. topology 的生产路径实际上仍未完全落地

`LocalizationTopologyController` 已经改成显式调用：

```python
densify_and_split_selected(selected_mask=...)
```

并对点数变化做检查，这是正确方向。([GitHub][4])

但在当前公开分支的实际 3DGS `GaussianModel` 中，我没有查到 `densify_and_split_selected()` 的具体实现；controller 在 split 非空时会主动抛出异常。当前能找到的 densification/prune 实现在 `class GaussianModel` 声明之前，属于前面的模型类。([GitHub][5])

此外还有一个小问题：

```python
"candidate_count": int(split.numel())
```

记录的是全部 Gaussian 数量，而不是候选数，应改成 eligible mask 或 split candidate mask 的 `sum()`。([GitHub][4])

所以目前 topology 的状态更准确地说是：

> controller 接口和测试框架已准备好，但真实 3DGS optimizer mutation 尚未完成生产级验证。

---

## 6. 后续阶段还有一个学习率恢复问题

当前 `_set_phase_lrs()` 在 feature phase 将非 `loc_feature` 参数组设为零；进入 geometry/topology phase 后，只显式恢复或设置 xyz、scale、rotation，`loc_opacity` 等此前被设为零的组不会自动恢复。([GitHub][1])

建议每次切 phase 时先执行：

```python
for group in optimizer.param_groups:
    group["lr"] = group["la_base_lr"]
```

再根据 phase 将冻结组设零、对 geometry 组乘 multiplier。否则后续实验可能名义上解锁了参数，实际上学习率仍为零。

---

# 三、下一版方法应从“descriptor alignment”升级为“pose-effective correspondence learning”

建议把下一版本定义为：

# LA-STDLoc v0.3：Inference-Aligned and Geometry-Aware Descriptor Distillation

核心不再只是让 Gaussian feature 接近 query feature，而是直接训练：

1. 正确 Gaussian 在完整地图中被检索出来；
2. 错误但外观相似的 Gaussian 被压下去；
3. 保留的对应在 2D/3D 上具有良好空间分布；
4. utility 能预测真实正确匹配和 PnP inlier。

---

## 3.1 使用 full-bank 双向匹配损失

对 query observation (q_i) 和完整 landmark bank ({f_j})：

[
P_G(j\mid q_i)
==============

\operatorname{softmax}_j
\frac{q_i^\top f_j}{\tau}.
]

同时对 Gaussian (f_i) 和 query descriptor pool：

[
P_Q(k\mid f_i)
==============

\operatorname{softmax}_k
\frac{f_i^\top q_k}{\tau}.
]

使用：

[
\mathcal L_{\text{bi-MNN}}
==========================

-\log P_G(i\mid q_i)
-\log P_Q(i\mid f_i).
]

它比单纯 cosine loss更接近实际 MNN inference。

完整 16384 bank 不一定造成不可接受显存：

* 每次使用 256–512 个 query observations；
* 全 bank 分块计算；
* FP16 logits；
* 只保留 top-32 或 top-64 hard negatives。

推荐损失：

[
\mathcal L_{\text{v0.3}}
========================

\lambda_d\mathcal L_{\text{direct}}
+
\lambda_m\mathcal L_{\text{bi-MNN}}
+
\lambda_h\mathcal L_{\text{hard-neg}}
+
\lambda_a\mathcal L_{\text{anchor}}
+
\lambda_q\mathcal L_{\text{quality}}.
]

---

## 3.2 Hard negatives 要来自完整地图

重点选择：

* 外观极相似但 3D 距离远的 landmark；
* ShopFacade 中重复窗户、边缘和立面纹理；
* 当前 sparse matcher 实际产生过的错误匹配；
* cosine 排名前列但 GT reprojection 不一致的 Gaussian。

同时设置 ignore set：

* 3D 距离极近的 Gaussian；
* 投影距离小于 1–2 feature pixels 的 landmark；
* 同一局部表面、无法可靠区分的 Gaussian；
* 深度不确定或遮挡边界。

当前只使用当前 query 内的 negatives，不足以覆盖真正困难的地图级混淆。

---

## 3.3 把 FIFO memory 改成 view-diverse prototype bank

当前 memory 每个 landmark 保存固定数量 descriptor，但没有记录：

* 相机方向；
* 相机距离；
* observation confidence；
* visibility quality；
* 与已有 observation 的视角差异。([GitHub][2])

建议每个 landmark 保存：

```python
descriptor
view_direction
camera_distance
confidence
camera_or_sequence_id
```

插入新 observation 时，不简单覆盖下一个 slot，而是：

1. 如果与已有 observation 视角接近，只保留质量更高者；
2. 如果视角差异足够大，写入新 slot；
3. memory 满时删除最冗余的 observation。

这样 multiview loss学习到的是真正的跨视角稳定性，而不是邻近视频帧的一致性。

---

## 3.4 用 trust region 控制 descriptor drift

drift 约 0.232 说明描述子已经发生较大变化。它不一定错误，但目前缺少约束来防止对 ShopFacade query split 过适配。

建议保存 baseline descriptor (f_i^0)，使用残差参数化：

[
f_i
===

\operatorname{normalize}
\left(
f_i^0+g_i\Delta f_i
\right),
\qquad 0\le g_i\le1.
]

其中 (g_i) 由以下因素控制：

* 足够的独立视角数；
* observation quality；
* full-bank correct-match rate；
* feature consistency。

并加入：

[
\mathcal L_{\text{anchor}}
==========================

\sum_i
w_i
\left(
1-\cos(f_i,f_i^0)
\right).
]

权重应是自适应的：

* 观测少、utility 不确定：强 anchor；
* 多视图稳定、真实匹配改善：弱 anchor。

还可以只允许 baseline sampled landmarks 接收 localization gradient，非 landmark Gaussian 保持 baseline feature，减少无必要漂移。

---

# 四、重新定义 utility：从代理统计变成真实定位标签

下一版不要继续手调一组 z-score 权重。先生成真正的监督标签。

对 held-out training query 执行实际 sparse matching。对预测匹配 ((u_q,X_i))，用 GT pose 定义：

[
y_{qi}^{corr}
=============

\mathbb 1
\left[
|\pi(T_q^*X_i)-u_q|<\delta
\right].
]

同时记录：

[
y_{qi}^{inlier}
===============

\mathbb 1[\text{match is a PnP inlier}].
]

每个 Gaussian 的统计改成：

[
R_i=
\frac{\text{correct match count}}
{\text{visible count}+\epsilon},
]

[
I_i=
\frac{\text{PnP inlier count}}
{\text{matched count}+\epsilon}.
]

推荐统计语义：

| 统计量                  | 新定义                                    |
| -------------------- | -------------------------------------- |
| positive probability | full-bank dual-softmax/MNN probability |
| repeatability        | 不同 view bins 中正确匹配的比例                  |
| entropy              | 完整候选分布的归一化熵                            |
| margin               | 正确匹配与最强有效 hard negative 的差             |
| reprojection error   | 预测匹配在 GT pose 下的真实像素误差                 |
| outlier              | 实际 PnP outlier EMA                     |
| information          | 对当前 PnP 信息矩阵的边际 log-det 增益             |
| redundancy           | 3D、descriptor 和 view coverage 的联合冗余    |

可以用一个很小的 logistic model：

[
\hat q_i
========

\sigma
\left(
w^\top z_i+b
\right)
]

预测 landmark 成为正确匹配或 PnP inlier 的概率，而不是手工直接相加。

为防止自我确认，标签应使用：

* support/query cross-fitting；
* 序列块或 pose-cluster 划分；
* 当前模型的 EMA 版本；
* 或上一轮 checkpoint 生成标签。

在 utility 与真实 inlier value 的相关性显著提高之前，不要让它控制 physical topology。

---

# 五、解决“更多 inliers 但 pose 被拉偏”：几何均衡对应选择

这一步属于方法，而不是 RANSAC 调参。

当前 sparse stage 在匹配后直接把所有选中对应交给统一 reprojection threshold 的 pose solver。([GitHub][6])

建议加入一个确定性的 `GeometryBalancedSelector`：

## PnP 前

对候选对应按置信度排序，然后依次施加：

1. **2D grid quota**：每个图像网格最多保留 (k_{2D}) 个；
2. **3D voxel quota**：每个 3D voxel 最多保留 (k_{3D}) 个；
3. **ANMS/FPS**：避免对应集中在局部；
4. **descriptor redundancy suppression**：去掉 descriptor 和 3D 位置都高度冗余的匹配。

## 初始 PnP 后

在第一轮 inliers 上计算 pose Jacobian：

[
J_i
===

\frac{\partial \pi(TX_i)}{\partial\xi}.
]

然后贪心选择能最大化：

[
\log\det
\left(
\sum_iw_iJ_i^\top J_i+\lambda I
\right)
]

的 inlier 子集，再做一次固定参数的 pose refinement。

这一过程直接优化对应集合的几何条件，不需要改变 RANSAC threshold。

最有说服力的实验是一个 (2\times2) 设计：

| Map                 | Correspondence selector |
| ------------------- | ----------------------- |
| baseline descriptor | 原始 selector             |
| LA descriptor       | 原始 selector             |
| baseline descriptor | geometry-balanced       |
| LA descriptor       | geometry-balanced       |

四组使用完全相同的 RANSAC 参数。这样可以拆分：

* descriptor learning 的收益；
* geometry selection 的收益；
* 二者是否互补。

---

# 六、真正把 dense teacher 接回来的方法

这是下一轮最能增强论文核心贡献的部分。

## 6.1 从 dense pixel 对应聚合到 Gaussian

对 dense rendered anchor (j)，需要获得其 top-(K) Gaussian contributors：

[
r_{ji}
======

P(G_i\mid\text{rendered pixel }j).
]

最成熟的方式是让 rasterizer 返回：

```python
topk_gaussian_ids: [M, K]
topk_composition_weights: [M, K]
```

如果暂时不修改 CUDA rasterizer，可以近似：

* 找投影椭圆覆盖该 pixel 的 Gaussian；
* 用 Mahalanobis 距离计算 2D Gaussian density；
* 用深度差过滤；
* 乘 opacity 和近似 transmittance；
* 保留 top-4 或 top-8；
* 归一化为 responsibility。

然后：

[
s_i
===

\frac{
\sum_jr_{ji}s_j
}{
\sum_jr_{ji}+\epsilon
}.
]

必须增加一个归因正确性测试：

> 使用 top-(K) responsibility 重构 anchor feature，与原 rasterized feature 的 cosine 应足够高。

---

## 6.2 把 dense distribution 蒸馏成 sparse distribution

Dense teacher 对 query feature (q) 和 rendered anchors (j) 产生：

[
P_D(j\mid q).
]

聚合为 Gaussian-level teacher distribution：

[
P_T(i\mid q)
============

\sum_jP_D(j\mid q)r_{ji}.
]

Sparse student 为：

[
P_S(i\mid q)
============

\operatorname{softmax}_i
\left(
q^\top f_i/\tau
\right).
]

使用：

[
\mathcal L_{\text{KD}}
======================

D_{\mathrm{KL}}
\left(
\operatorname{sg}(P_T)
\parallel
P_S
\right).
]

只蒸馏以下 dense matches：

* cycle-consistent；
* GT reprojection 正确；
* teacher confidence 高；
* dense refinement 相比 sparse pose 确实改善。

可以定义 episode 权重：

[
w_q
===

\max
\left(
0,
e(T_q^0)-e(T_q^1)
\right),
]

其中 (T_q^0) 是 sparse pose，(T_q^1) 是 dense-refined pose。

这样就真正实现了：

> dense stage 只在建图期存在，并将它能修正的对应关系蒸馏到 sparse map。

---

# 七、topology 应当最后开启

当前 utility 相关性只有 0.106，且生产 `GaussianModel` 中 split/prune mutation 尚未完整实现，因此现在启用 topology 只会引入新的混杂因素。

推荐顺序：

### 先实现 split-only

分裂条件：

[
S_i^{split}
===========

R_i^{repeat}
\cdot
H_i^{dense}
\cdot
M_i^{teacher}
\cdot
r_i^{2D}.
]

也就是：

* 多视图 repeatable；
* dense correspondence entropy 高；
* teacher 对该区域有稳定监督；
* Gaussian footprint 较大。

不要先 prune，也不要 clone。

### Child feature 不能长期复制

根据 dense responsibilities 或 view observation 聚类：

[
m_{i,1}
=======

\frac{\sum_{j\in C_1}r_{ji}q_j}{\sum_{j\in C_1}r_{ji}},
\qquad
m_{i,2}
=======

\frac{\sum_{j\in C_2}r_{ji}q_j}{\sum_{j\in C_2}r_{ji}}.
]

两个 child 分别初始化为不同 prototype。

### 再做 soft prune

只有在 utility 已能预测真实 inlier value 后，才降低 `loc_opacity`。

### 最后做 physical prune

它的贡献最好表述为：

* 在固定 accuracy 下压缩地图；
* 或在固定 Gaussian 数量下提高定位。

否则增加 Gaussian 数量获得提升，会削弱方法性证据。

---

# 八、下一步最强的实验设计

## 8.1 先把当前结果变成“solver-invariant evidence”

不要只报告 6、8、12px 三个点。对 baseline 和 LA 同时评估：

[
{2,4,6,8,10,12,16}\text{ px}.
]

报告：

* AE/TE–threshold curve；
* 2cm 和 5cm recall–threshold curve；
* 曲线下面积；
* 每个相同 threshold 的 paired difference。

最终主表采用：

1. **共享阈值**：在 validation scenes 上选择一次，baseline 与 LA 共用；
2. **各自调优阈值**：两个方法分别只在 validation 上选，作为附加结果；
3. 不允许根据 ShopFacade test 结果选择 6px。

如果当前 ShopFacade 数字来自 test set 上的阈值搜索，那么它可以作为开发诊断，但不能成为最终无偏结论。

---

## 8.2 做 paired per-query 统计

由于 baseline 和 LA 评估的是同一批 query，应对每个 query 记录：

[
\Delta e_q
==========

e_q^{LA}-e_q^{base}.
]

然后报告：

* paired bootstrap 95% CI；
* 改善 query 比例；
* 退化 query 比例；
* translation/rotation scatter；
* recall 的 paired bootstrap 或 McNemar 检验；
* 失败 query 的类别分析。

这比单独比较两个中位数更有证明力。

---

## 8.3 Full-bank mechanism diagnostics

下一版至少报告：

### Descriptor 层

* full-bank retrieval Recall@1/5/10；
* full-bank MNN precision；
* 正确匹配与 top hard negative margin；
* expected calibration error；
* feature drift 与正确率改善的关系。

### Correspondence 层

* GT 2px/4px/6px correct-match precision；
* correct correspondence 数量；
* 每个图像的 2D grid occupancy；
* 2D convex-hull area；
* 3D voxel occupancy；
* 3D point covariance eigenvalues。

### Pose 层

* PnP inlier ratio；
* inlier median reprojection error；
* (J^\top WJ) condition number；
* log-det pose information；
* RANSAC 成功率；
* refinement 前后 pose error。

这能够回答：

> LA 是因为产生了更多相似匹配，还是因为产生了更多几何正确、空间均衡、位姿信息有效的匹配？

---

## 8.4 多场景与多随机种子

当前 ShopFacade 是明确的正向 pilot，但还不足以说明泛化。

建议至少先跑：

* ShopFacade；
* KingsCollege；
* OldHospital；

每个场景三个训练 seed。

最终扩展到 Cambridge 全部场景，并至少选若干 7-Scenes 室内场景。主结果必须使用同一套方法超参数，场景间只允许修改已有的 scene-scale 参数。

support/query split 不应只做随机帧划分，应优先使用：

* 序列块；
* 时间块；
* camera trajectory block；
* pose clustering；

避免相邻视角同时落入 support 和 query。

---

# 九、建议的下一轮实施顺序

| 优先级 | 修改                                                               | 目的                        |
| --- | ---------------------------------------------------------------- | ------------------------- |
| P0  | 修复 phase LR 恢复；完成真实 `GaussianModel.densify_and_split_selected()` | 消除代码隐患                    |
| P0  | 增加 full-bank diagnostics 和 paired query 输出                       | 把现有正向结果证实                 |
| P1  | full-bank bi-MNN loss + hard-negative bank                       | 对齐真实 sparse inference     |
| P1  | view-diverse memory + residual/anchor regularization             | 提升跨视角稳定性并控制 drift         |
| P1  | 2D grid + 3D voxel geometry-balanced selector                    | 解决 inlier 多但 pose 偏的问题    |
| P2  | 用真实 correct-match/PnP-inlier 标签训练 utility                        | 将 Spearman 从弱代理提升为可用预测    |
| P2  | 完成 dense anchor→Gaussian responsibility                          | 回到原始 dense-to-sparse 核心贡献 |
| P3  | dense distribution KL distillation                               | 把 dense 能力真正转入 sparse map |
| P4  | split-only topology，随后 soft prune                                | 验证 topology 的额外收益         |

---

# 最建议立即跑的三个实验

第一，**无需重训练**：对 baseline 和当前 v0.2 checkpoint 做完整 shared-threshold sweep，并输出 paired per-query 结果。这会确认现有提升是否对 solver threshold 稳健。

第二，**无需改 Gaussian 训练**：实现 geometry-balanced correspondence selector，做 baseline/LA × original/balanced 的 (2\times2) 实验，所有组固定同一 RANSAC 参数。这最有可能直接解决当前的 pose 偏移问题。

第三，训练 **v0.3 feature-only**：

* baseline landmarks；
* baseline detector；
* geometry 固定；
* loc opacity 固定；
* full-bank bi-MNN；
* top hard negatives；
* view-diverse memory；
* baseline feature anchor；
* 不使用 utility sampling；
* 不使用 topology。

如果这个版本在共享 RANSAC threshold、多个场景和多个 seed 上仍稳定超过 baseline，就能形成很强的第一阶段结论：

[
\boxed{
\text{Localization-aware descriptor learning alone}
\Rightarrow
\text{better sparse Gaussian localization}
}
]

然后再加入 dense-to-sparse KL distillation 和 topology，分别证明完整方法的第二、第三层贡献。

[1]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_locaware.py "LA-STDLoc/train_locaware.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[2]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/direct_landmark_teacher.py "LA-STDLoc/localization_training/direct_landmark_teacher.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[3]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/descriptor_diagnostics.py "LA-STDLoc/localization_training/descriptor_diagnostics.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[4]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/topology_controller.py "LA-STDLoc/localization_training/topology_controller.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[5]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/scene/gaussian_model.py "LA-STDLoc/scene/gaussian_model.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[6]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/stdloc.py "LA-STDLoc/stdloc.py at LA · Arthurshen926/LA-STDLoc · GitHub"

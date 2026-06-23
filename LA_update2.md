## 结论

需要调整，但不是推翻方向，而是把主张从：

> 一个统一 localization utility 可以普遍指导 Gaussian feature、split 和 prune，从而稳定提高定位。

调整为：

> **定位监督能够改善 sparse Gaussian descriptor；dense teacher 和 topology mutation 只有在其收益可验证时才被选择性采用。**

当前最可靠的正向证据，是固定 landmark、detector 和 sparse pipeline 后的 feature adaptation，以及 ShopFacade 上 matched no-split control 对 topology 的成对改善。ShopFacade 4-event topology 相对同长度 no-split adaptation，平移误差的 paired CI95 为负，说明该场景存在真实的 topology 收益；但 KingsCollege 基本中性，OldHospital 在 aggressive prune 下明显不稳，dense-KL smoke 也尚未形成正向结果。仓库自己的 closure 判断与此一致。([GitHub][1])

因此，下一阶段不应继续围绕一套全局 utility 权重扫描，而应把方法重构为：

[
\boxed{
\text{Inference-aligned descriptor learning}
+
\text{Selective dense supervision}
+
\text{Risk-controlled topology intervention}
}
]

---

# 一、首先修正当前实验结论中的一个重要问题

当前文档里的“三场景三 seed”，严格来说是**三组 query split seed，而不是真正的三个训练随机种子**。

`train_locaware.py` 在参数解析前固定调用了 `seed_everything(2025)`，随后 `safe_state()` 又将 Python、NumPy 和 PyTorch 随机种子统一重置为 0；多场景脚本变化的只是 `V03_QUERY_SPLIT_SEED`。因此当前不同 seed 实验主要反映 support/query 随机划分变化，而不是优化初始化、相机采样、anchor 采样等训练随机性。([GitHub][2])

这不否定 ShopFacade 的 paired positive，但会削弱“多随机种子稳定性”的证据强度。建议立刻改为：

```python
if __name__ == "__main__":
    parser = ...
    args = parser.parse_args()

    safe_state(args.quiet)
    seed_everything(args.train_seed)  # 必须放在 safe_state 后
```

脚本中拆分：

```text
TRAIN_SEEDS="0 1 2"
QUERY_SPLIT_SEEDS="2025 2026 2027"
```

当前结果在论文或实验记录里应改称：

> three query-split repetitions

而不是 three training seeds。

此外，当前 support/query 是对相机列表执行随机 `randperm` 划分，相邻视频帧很可能分别进入 support 和 query。下一轮应增加：

* sequence-block split；
* temporal-block split；
* 或按 camera pose clustering 划分。

这样才能检验真正的视角泛化，而不是邻近帧适配。([GitHub][3])

---

# 二、哪些主张应保留，哪些应收缩

| 原主张                                     | 当前判断      | 建议表述                                                  |
| --------------------------------------- | --------- | ----------------------------------------------------- |
| 定位监督能改善 Gaussian sparse descriptor      | **保留**    | 已有 ShopFacade 正向 pilot，需要扩大跨场景验证                      |
| topology 可以进一步提升定位                      | **条件性保留** | topology 对部分场景有效，但不是统一稳定的增益模块                         |
| 一个统一 utility 可同时控制 landmark、split、prune | **暂时否定**  | split、prune、landmark selection 应采用不同目标                |
| dense stage 已被成功蒸馏到 sparse map          | **暂不成立**  | 当前 direct teacher 正向，dense-KL 尚无正向证据                  |
| physical prune 能提高精度                    | **降级**    | 优先作为地图压缩或效率模块，而非精度主模块                                 |
| 方法在 Cambridge 上跨场景稳定有效                  | **尚不能主张** | 目前只能说 ShopFacade pilot 正向、Kings 边界性、OldHospital mixed |

另一个需要考虑的现实是，最近的 SplitGS-Loc 已经使用 Gaussian splitting 处理 many-to-one correspondence，ULF-Loc 也已经从 alpha-blending feature bias、feature fusion 和 landmark selection 角度改进定位。因此，“定位特征优化 + Gaussian split/prune”本身已经比较拥挤。更有辨识度的贡献应是：

> **根据实际定位风险，有选择地接受或拒绝地图更新，而不是无条件执行 heuristic topology mutation。**

([arXiv][4])

---

# 三、理论方法需要从“静态 utility”改成“操作收益”

当前最大的方法性问题不是 utility 权重没调好，而是：

> 一个 Gaussian 的定位价值并不是独立、静态、可加的。

一个单独看起来匹配率低的 Gaussian，可能是某个稀有视角或空间区域唯一的 landmark；移除它会破坏覆盖。相反，一个匹配率很高的 Gaussian，如果附近已经有大量同类 landmark，继续保留或 split 的边际价值可能很低。

因此不应再定义统一的：

[
U_i=\text{reliability}_i+\text{geometry}_i-\text{redundancy}_i
]

然后让同一个 (U_i) 同时决定 selection、split 和 prune。

应该分别定义操作的反事实收益：

[
\Delta R_i^{split}
==================

R(\mathcal G\oplus split_i)-R(\mathcal G),
]

[
\Delta R_i^{prune}
==================

R(\mathcal G\setminus i)-R(\mathcal G),
]

其中 (R) 是 held-out localization risk，例如：

[
R(\mathcal G,\theta)
====================

\mathbb E_{q\in Q}
\left[
\lambda_t e_t(q)
+
\lambda_r e_r(q)
+
\lambda_f\mathbf 1_{\mathrm{fail}}
\right].
]

总体目标调整为：

[
\min_{\theta,\mathcal G_L}
R(\theta,\mathcal G_L)
+
\lambda_B C(\mathcal G_L)
+
\lambda_D D(\theta,\theta_0),
]

其中：

* (\theta)：localization descriptor；
* (\mathcal G_L)：定位 landmark map；
* (C)：地图大小或推理成本；
* (D)：相对 baseline descriptor/geometry 的漂移约束。

这会把 topology 从“根据分数自动操作”改成：

> 生成候选操作 → 在 held-out episodes 上估计定位风险变化 → 只有收益可信时才提交。

---

# 四、当前 utility 为什么还不足以支持 topology

当前 direct teacher 的统计仍包含明显的代理项：

```python
reproj_error = 0
information = 0
repeatability = positive_prob > 0.25
```

而最终 reliability 是多个 robust z-score 的手工加减；split score 中如果所有候选的 `information` 都为零，还会把 pose effectiveness 整体替换为 1。([GitHub][5])

另外，label-state 路径又把实际 inlier rate 写入 `loc_information_ema`。这意味着同一个字段同时被当作：

* pose information；
* PnP inlier probability。

二者语义不同。前者衡量几何条件：

[
\Delta \log\det H,
]

后者衡量匹配可靠性：

[
P(\text{PnP inlier}\mid\text{matched}).
]

必须拆成不同状态：

```text
loc_match_prob_ema
loc_mnn_correct_ema
loc_inlier_prob_ema
loc_reproj_error_ema
loc_pose_information_ema
loc_view_repeatability_ema
loc_coverage_rarity_ema
```

当前 KingsCollege 的 utility Spearman 约 0.014，校准后仍约 0.029，而且 top quartile inlier rate 低于 bottom quartile，已经说明该 utility 暂时不能用于 aggressive prune。([GitHub][1])

## Utility v2

建议训练两个独立的小型校准器：

[
p_i^{match}
===========

P(\text{GT-correct match}\mid x_i),
]

[
p_i^{inlier}
============

P(\text{PnP inlier}\mid\text{matched},x_i).
]

输入包括：

* full-bank positive probability；
* top-1/top-2 margin；
* normalized entropy；
* MNN 是否正确；
* detector 命中率；
* 独立 view-bin 中的正确匹配率；
* GT reprojection error；
* 3D/descriptor redundancy；
* projected radius；
* opacity；
* Jacobian marginal information；
* 当前 2D grid 和 3D voxel 的稀缺度。

使用 cross-fitting：

1. fold A 生成真实 sparse match/inlier 标签；
2. 在 A 上拟合 logistic/小 MLP calibrator；
3. 在 fold B 上预测 utility 并执行 topology；
4. A/B 交换。

不要在生成标签的同一批 query 上直接训练和执行 topology。

建议设置工程 Go/No-Go 门槛：

* held-out Spearman (>0.25)；
* utility top quartile 的 inlier rate 至少高于 bottom quartile 10 个百分点；
* calibration ECE 和 AUROC 明显优于单独使用 baseline match score。

达不到这些条件时，utility 只能用于诊断，不能用于 physical prune。

---

# 五、descriptor 主路径仍有明确改进空间

当前 v0.3 是最值得继续加强的部分，但还有两个实现细节会造成场景依赖。

## 1. anchor 采样不是空间均衡的

当可见 landmark 超过上限时，当前 `_limit_valid_indices()` 对有效索引顺序执行 `torch.linspace`。这不是随机采样，也不是 2D/3D 分层采样，最终覆盖取决于 landmark 的索引排列。([GitHub][5])

应替换为：

1. 先按 query image 的 (8\times 8) 或 (12\times 12) grid 分桶；
2. 每个格子保留固定 quota；
3. 再按 3D voxel 去冗余；
4. 剩余名额根据 detector repeatability 和 view rarity 分配。

这样不同场景和不同 query 上的 descriptor supervision 才更均衡。

## 2. full-bank hard negative 存在 false negatives

当前 full-bank bi-MNN 把正确 index 之外的所有 Gaussian 都作为负样本，hard-negative top-k 也没有 3D、投影或 source-id ignore mask；相比之下，multiview loss 至少排除了图像中距离过近的 negative。([GitHub][5])

split 后尤其容易出现：

* sibling Gaussians 拥有相同 source id；
* 同一局部表面存在多个近邻高斯；
* GT 投影落在近乎相同的 feature pixel。

这些不应全部作为互斥类别。

建议使用 multi-positive/ignore-set：

[
P(i)=
\left{
j:
d_{3D}(i,j)<\tau_{3D}
\lor
d_{2D}(i,j)<\tau_{2D}
\lor
source(i)=source(j)
\right}.
]

然后采用 multi-positive InfoNCE，而不是单一 index CE。

## 3. drift 控制应自适应

不要为所有 Gaussian 使用相同 anchor weight。定义：

[
\lambda_i^{anchor}
==================

\lambda_0
\exp(-\gamma n_i^{view})
\left(1-\hat p_i^{match}\right).
]

即：

* 多视图证据少、匹配改善不确定：强 anchor；
* 多视图充分、held-out match 明显改善：允许更大更新。

这比全局控制 descriptor drift 更符合当前不同场景的表现。

---

# 六、dense teacher 应改成“选择性教师”，而不是全局 KL

目前 dense responsibility/KL 的代码链路已经完成，但 ShopFacade smoke 中：

* responsibility reconstruction mean cosine 约 0.70；
* p10 约 0.40；
* sparse-only 指标没有受益。

这说明当前 pixel-to-Gaussian attribution 还不够可靠，不能把所有 dense distribution 都当作教师。仓库当前实现会将 dense probability 通过 responsibility 聚合成 Gaussian distribution，再直接对 sparse bank 做 batch-mean KL。([GitHub][6])

建议改成三个 gating 条件。

## 1. Pose-improvement gate

只在 dense stage 实际改善 sparse pose 时使用：

[
w_q^{pose}
==========

\max
\left(
0,
e(T_q^{sparse})-e(T_q^{dense})
\right).
]

如果 dense stage 没有改善，甚至变差，则不蒸馏该 episode。

## 2. Attribution-confidence gate

对 anchor (j)：

[
w_j^{attr}
==========

\mathbf 1[
c_j^{reconstruct}>\tau_c]
\cdot
\mathbf 1[
H(r_j)<\tau_H].
]

初期建议只使用 reconstruction cosine 很高、responsibility entropy 很低的 anchors，而不是追求覆盖率。

## 3. Correspondence correctness gate

要求：

* depth consistency；
* cycle consistency；
* GT reprojection 通过；
* dense MNN confidence 高；
* query 与 rendered anchor 不在遮挡边界。

最终：

[
\mathcal L_{\text{selective-KD}}
================================

\frac{
\sum_{q,j}
w_q^{pose}w_j^{attr}w_j^{corr}
D_{\mathrm{KL}}
(P_T\parallel P_S)
}{
\sum_{q,j}
w_q^{pose}w_j^{attr}w_j^{corr}+\epsilon
}.
]

还可以先不用完整 distribution KL，而仅用 dense teacher 发现的：

* 新 hard positive；
* baseline sparse 漏掉但 dense 正确找到的 match；
* dense 修正成功区域的 ranking margin。

这会比直接蒸馏带噪声的完整分布更稳定。

---

# 七、topology 路线应改为 split-first、prune-last

当前跨场景结果已经说明：

* 保守 split 风险相对可控；
* aggressive physical prune 是主要不稳定来源；
* 固定全局阈值会在不同场景产生数量级不同的 prune 数量。

因此下一轮顺序应当是：

## 1. Split-only 主实验

完全关闭：

```text
physical prune
soft prune
clone
```

只允许 split，且要求：

* projected footprint 大；
* dense correspondence ambiguity 高；
* 多视图 repeatability 高；
* teacher attribution 稳定；
* 该区域存在真实 many-to-one 错配；
* sibling features 能被不同 observation cluster 分开。

SplitGS-Loc 已经表明 many-to-one ambiguity 是关键问题，因此你的差异不能只是“也做 split”，而应是：

> 只有当 held-out localization risk 预测为下降时，才执行 split。

([arXiv][7])

## 2. Soft-mask counterfactual prune

对候选 prune 集合，不要立即删点：

```python
loc_mask[candidates] = 0
```

在一个小型 held-out query buffer 上评估：

[
\Delta R
========

## R(\mathcal G_{\text{masked}})

R(\mathcal G).
]

只有满足：

* 平均 risk 降低；
* recall 不下降；
* 2D/3D coverage 不下降；
* paired bootstrap upper bound 小于容忍值；

才提交 physical prune。

否则回滚。

## 3. 极小的事件预算

在 utility 尚不稳定时，建议每次 physical prune 不超过当前定位点数的约 (0.1%-0.5%)，并保护：

* 每个 2D grid 最小 landmark 数；
* 每个 3D voxel 最小 landmark 数；
* 稀有 view-bin landmark；
* 当前 PnP information 高的 landmark；
* baseline protected source ids。

## 4. 将 physical prune 改成压缩贡献

如果 physical prune 无法跨场景提升精度，但能在几乎不损失精度的情况下减少地图，可以将它的论文目标改成：

> 固定定位精度下压缩地图，而不是依靠 prune 提高定位精度。

这样更符合当前证据，也更容易形成稳定的 Pareto 结果。

---

# 八、建议增加 localization-only overlay map

从当前 OldHospital 的破坏性 prune 看，我建议正式考虑一个结构调整：

[
\mathcal G_{\text{render}}
+
\mathcal G_{\text{loc}}.
]

其中：

* 原始 RGB/feature render Gaussians 保持冻结；
* localization map 从 baseline landmarks 初始化；
* 每个 localization Gaussian 通过 `source_id` 锚定到 render Gaussian；
* feature、loc opacity、split、soft prune 和 physical prune 全部只作用于 (\mathcal G_{\text{loc}})；
* render map 只为 dense teacher 提供 geometry/depth/feature context。

这能解决三个问题：

1. localization topology 不再破坏 photometric map；
2. localization map 可以严格控制数量预算；
3. split/prune 后 sparse descriptor 与 3D point 的语义更清楚。

需要注意，解耦地图本身不是足够的新颖的贡献；新颖点仍应放在：

* selective dense supervision；
* operation-specific benefit prediction；
* held-out risk-based accept/rollback。

---

# 九、下一轮最重要的实验矩阵

所有 topology 实验必须有**同长度 matched continuation**，不能只比较 30500 checkpoint 与 30600 topology checkpoint。

建议每个 scene/split/train-seed 都运行：

| 组别 | 30500→30600 的操作                      |
| -- | ------------------------------------ |
| A  | no-mutation continuation             |
| B  | split-only                           |
| C  | soft-mask prune                      |
| D  | committed physical prune             |
| E  | split + soft-mask                    |
| F  | selective dense teacher + split-only |

要求：

* 相同 query episode 顺序；
* 相同训练 seed；
* 相同训练步数；
* 相同 detector、landmark budget 和 solver；
* physical prune 组从相同 parent checkpoint 分叉。

第一轮先做 Cambridge 五个场景：

* ShopFacade；
* KingsCollege；
* OldHospital；
* GreatCourt；
* StMarysChurch。

完成后再选 7-Scenes 若干室内场景，检查室内遮挡和视角变化。

---

# 十、建议的 Go/No-Go 标准

## Descriptor 主模块

进入论文主方法的条件：

* Cambridge 场景平均正向；
* 绝大多数场景不出现明显退化；
* 三个真实训练 seed 一致；
* shared solver threshold 下仍然正向；
* full-bank Recall@1、MNN precision、PnP inlier rate同步改善。

## Utility

允许控制 topology 的条件：

* held-out Spearman 明显为正，建议至少约 0.25；
* top quartile inlier rate明显高于 bottom quartile；
* calibration 在不同场景上仍有效；
* 不依赖每个场景重新拟合阈值。

## Dense teacher

允许进入主方法的条件：

* eligible episode 中，dense pose 大多数确实优于 sparse pose；
* responsibility reconstruction 足够可靠；
* selective KL 优于 direct teacher，而不是仅增加 loss；
* 多场景 sparse-only 结果正向。

## Topology

成为 accuracy contribution 的条件：

* 相对 matched no-mutation，在多数 Cambridge 场景上改善；
* paired CI 不只是单一场景为负；
* 点数增长受控；
* 不需要每场景单独手工阈值。

若达不到，则将 topology 降级为：

* localization map compression；
* 可视化/诊断分析；
* 或附加模块。

---

# 推荐的主技术路线

下一版可以正式定义为：

## LA-STDLoc v0.4：Risk-Controlled Localization Map Optimization

由三层组成：

### 第一层：主干

**Inference-aligned descriptor adaptation**

* stratified query landmark sampling；
* multi-positive full-bank loss；
* view-diverse memory；
* adaptive baseline anchor；
* fixed sparse inference pipeline。

### 第二层：选择性教师

**Pose-improvement-gated dense-to-sparse distillation**

* dense stage 只有在真正修正 sparse pose 时才监督；
* 只蒸馏 attribution 可靠的 correspondences；
* 不执行无条件全局 KL。

### 第三层：条件性地图更新

**Counterfactual topology controller**

* split、prune 使用不同的收益预测器；
* 先 soft intervention；
* 在 held-out localization episodes 上评估；
* 支持 accept、rollback 和 abstain。

最终最合理的论文主张是：

> LA-STDLoc 将定位视为建图目标，但不假设所有定位反馈都可靠。它通过 inference-aligned descriptor learning 获得基础增益，只蒸馏能够改善姿态的 dense correspondences，并仅在 held-out localization risk 下降时提交 Gaussian map mutation。

这个主张比“定位 utility 指导统一 split/prune”更符合当前实验，也比单纯的 feature fusion 或 Gaussian splitting 更有方法辨识度。

[1]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/LA_update1_closure.md "LA-STDLoc/LA_update1_closure.md at LA · Arthurshen926/LA-STDLoc · GitHub"
[2]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_locaware.py "LA-STDLoc/train_locaware.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[3]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/episode_sampler.py "LA-STDLoc/localization_training/episode_sampler.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[4]: https://arxiv.org/abs/2605.04730?utm_source=chatgpt.com "ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting"
[5]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/direct_landmark_teacher.py "LA-STDLoc/localization_training/direct_landmark_teacher.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[6]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/dense_distill.py "LA-STDLoc/localization_training/dense_distill.py at LA · Arthurshen926/LA-STDLoc · GitHub"
[7]: https://arxiv.org/abs/2605.07351?utm_source=chatgpt.com "Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization"

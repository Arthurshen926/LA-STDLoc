## 结论

当前结果**还不能否定理论方向**。主要问题的优先级是：

1. **实现尚未真正完成“dense-to-sparse、per-Gaussian 定位归因”**；
2. **LA landmark sampling 和 soft detector 破坏了原 sparse stage 的空间覆盖**；
3. **3DGS topology 路径实际上不完整，并存在潜在的 split 严重错误**；
4. 最后才是 utility 权重、loss 权重等调参问题。

所以现在不宜继续在现有完整 Phase1–6 上扫权重。首先要修正因果链，并退回到一个严格受控的 feature-only 正向验证。

从指标看：

* sparse 平移中位误差上升约 **44.9%**；
* 旋转上升约 **20.3%**；
* 5cm/5deg recall 下降 **21.3 个百分点**；
* dense 又把误差恢复到 2.947cm/0.1312°，但 recall 只有 0.699。

这说明 Gaussian dense feature field 没有完全失效；更像是 **sparse landmark、detector 和独立 Gaussian descriptor 没有吸收 dense teacher 的能力**。当前还缺一个关键对照：**原始 baseline dense**。LA dense 应与 baseline dense 比，而不是与 baseline sparse 比。

---

# 一、当前最主要的实现问题

## P0-1：matching statistics 没有归因到具体 Gaussian

当前 dense teacher 得到了每个 anchor 的 `positive_prob`、margin、entropy 和 reprojection error，但最终先求全局均值，再对所有可见 Gaussian 执行 `expand(visible_count)`。因此同一个 episode 中，所有可见 Gaussian 获得完全相同的 matching statistics。`information` 和 `repeatability` 也采用同样处理。([GitHub][1])

目前只有 `loc_loss` 对 `means2d` 的梯度大小是真正 per-Gaussian 的；其他 utility 项并没有建立：

[
\text{dense anchor }j
\longrightarrow
\text{responsible Gaussian }i
]

这一对应关系。随后这些统计被直接 EMA 到 `full_idx`，并组合成最终 utility。([GitHub][2])

这造成两个问题：

* utility 的大部分分量不能区分 Gaussian；
* utility 很可能主要由 `loc_grad` 主导。

而高 gradient 只能说明“该 Gaussian 对当前 loss 敏感或当前误差较大”，不等于“它是可靠 landmark”。现在代码把 gradient 作为正向 utility，可能反而把模糊、错误或不稳定区域选成高价值 landmark。

### 必须修改

不要再使用单一 utility。至少分成三个分数：

[
Q_i^{rel}
=========

R_i+P_i+M_i-H_i-O_i-E_i
]

表示 landmark reliability；

[
Q_i^{split}
===========

G_i^{loc}\cdot H_i\cdot R_i\cdot
\mathbf 1[r_i^{2D}>\tau_r]
]

表示 split necessity；

[
Q_i^{geom}
==========

\Delta\log\det H_i
]

表示 pose geometry value。

用途应严格分离：

* landmark selection：(Q^{rel}+Q^{geom})；
* split：(Q^{split})；
* prune：低 (Q^{rel})、低 (Q^{geom})、充分观测；
* `loc_grad` 不再直接作为 landmark quality。

---

## P0-2：prototype 实际上是 Gaussian 自身 feature

当前代码将：

```python
prototype = gaussians.get_loc_feature[visible_idx].detach()
```

作为 prototype 写入 EMA buffer；之后训练又让当前 Gaussian feature 靠近这个历史 self-feature。([GitHub][1])

这不是 dense-to-sparse distillation，而更接近一个 temporal inertia regularizer。它不能把 query 图像中的描述子知识传给 Gaussian，反而可能阻止 descriptor 向更好的多视图表示移动。

### 必须修改

prototype 必须来自 query observation：

[
m_i
\leftarrow
\beta m_i
+
(1-\beta)
\frac{
\sum_v w_{iv}F_v(\pi(T_vX_i))
}{
\sum_vw_{iv}+\epsilon
}.
]

也就是：

```python
prototype_i = sampled_query_feature.detach()
```

而不是：

```python
prototype_i = gaussian_feature.detach()
```

修复之前应先把：

```text
loc_proto_weight = 0
loc_rank_weight  = 0
```

否则这两个 loss 没有正确语义。

---

## P0-3：LA landmark sampling 完全丢弃了空间覆盖

原始 STDLoc 的 `random_knn_score()` 会先在 3D 空间随机建立局部邻域，再从每个邻域中选择 match score 最好的 Gaussian，从而隐式保持 landmark 的空间分布。([GitHub][3])

LA 版本中：

```python
del xyz, k
sampled = torch.topk(combined, sample_num).indices
```

直接删除了 `xyz` 和 `k`，改成全局 top-k。([GitHub][4])

这非常可能是 sparse recall 下降的最大直接原因。全局 top-k 会把 landmarks 集中在少数高分立面区域，而 PnP 需要的是：

* 全图覆盖；
* 3D 空间覆盖；
* 足够大的基线；
* 良好的 pose information conditioning。

当前 sparse stage 又完全依赖 detector keypoints 与这些 selected Gaussian features 做匹配和 PnP，因此 landmark 聚集会直接转化为定位退化。([GitHub][5])

### 最小修复

先保持原有空间采样，只把 utility 作为局部 tie-break：

```python
combined = (
    base_weight * robust_normalize(base_score, eligible)
    + utility_weight * robust_normalize(reliability, eligible)
)

sampled_idx = random_knn_score(
    xyz,
    num,
    combined,
    k=k,
)
```

更成熟的版本再换成：

* 3D voxel quota；
* image-grid coverage；
* FPS；
* marginal pose-information greedy selection。

第一轮不要再使用全局 top-k。

---

## P0-4：当前 3DGS topology mutation 并没有真正接通

`train_locaware.py` 明确只支持 `gaussian_type=3dgs`，并实例化 `GaussianModel`。([GitHub][6])

但静态检查发现：

* `prune_points`
* `densify_and_split`
* `densification_postfix`
* optimizer tensor mutation

都定义在前面的另一个 Gaussian class 中；这些方法出现在 `class GaussianModel` 声明之前。当前使用的 3DGS `GaussianModel` 从文件第 2920 行左右才开始。([GitHub][2])

与此同时，topology controller 直接调用：

```python
gaussians.prune_points(...)
gaussians.densify_and_split(...)
```

([GitHub][7])

因此按当前分支：

* 如果 physical prune 或 split mask 非空，应触发属性不存在的问题；
* 你的流程能完整跑完，说明这些分支大概率一直没有被真正触发；
* topology phase 很可能主要只执行了 `loc_opacity` soft-prune，而没有真实 split/prune。

也就是说，目前还没有实验到“定位反馈改变 Gaussian topology”这一核心贡献。

### 另一个潜在严重错误

controller 构造了零梯度向量，只给选中的 split 点写入梯度，然后调用：

```python
densify_and_split(
    grads,
    grad_threshold=0.0,
)
```

而现有 split 实现采用：

```python
selected_pts_mask = padded_grad >= grad_threshold
```

零值同样满足 `>= 0`。一旦把该方法直接移植到 3DGS class，所有尺度足够大的 Gaussian 都可能被 split，而不是只有显式选中的点。([GitHub][7])

必须改为显式 mask 接口：

```python
gaussians.densify_and_split_selected(
    selected_mask=split_mask,
    num_children=2,
)
```

不要再通过构造 gradient 和 threshold 间接表达 mask。

---

## P0-5：“feature phase”并不是纯 feature-only

当前 feature phase 训练：

```python
trainable = {"loc_feature", "loc_opacity"}
```

并且 `--use_loc_opacity` 使用 `store_true, default=True`，正常命令行无法将其关闭。([GitHub][6])

所以 Phase1 同时改变了：

* Gaussian descriptor；
* feature rasterization 的 opacity；
* dense teacher 的可见性与混合关系。

这使 Phase1 无法回答最关键问题：

> 仅修改 Gaussian descriptor，能否提高 sparse localization？

### 修复

```python
parser.add_argument(
    "--use_loc_opacity",
    action=argparse.BooleanOptionalAction,
    default=False,
)
```

并新增真正的：

```text
feature_only:
    train loc_feature only
    use RGB opacity for rendering
    loc_opacity lr = 0
```

---

# 二、方法层面目前还缺少的核心环节

## 1. 当前优化的是 rendered dense feature，不是 sparse Gaussian descriptor

当前 dense teacher 从初始 pose 渲染 feature/depth，构造 rendered pixel 到 query pixel 的匹配，再优化 dense descriptor 和 reprojection loss。([GitHub][1])

但 sparse inference 使用的是：

```python
landmark_features = self.landmarks.get_loc_feature
corr = query_features @ landmark_features.T
```

即独立 Gaussian feature。([GitHub][5])

两者之间缺少明确蒸馏路径：

[
\text{dense pixel correspondence}
\not\Rightarrow
\text{specific sparse Gaussian descriptor}.
]

这正好可以解释现在的结果：

* dense stage 尚可；
* sparse stage 退化。

换句话说，目前更像“增强 dense feature rendering”，而不是“把 dense localization 能力蒸馏到 sparse map”。

---

## 2. anchor correspondence 存在错误监督

当前流程将初始 pose 下渲染深度反投影为世界点，再投影到 GT pose；有效性判断主要是正深度、有限值和图像范围。没有检查该点在 GT view 中是否被遮挡。([GitHub][8])

因此可能发生：

* 初始视角看到的是后方表面；
* 投影到 GT view 后，该表面被前景遮挡；
* 代码却在前景 query feature 上建立正对应。

还可能有 expected depth 在前后表面之间混合的问题。

### 修复

同时渲染 GT pose 的 depth 和 alpha，并要求：

[
|z_{q}(X)-D_{GT}(u^*)|
<
\max(\tau_{abs},\tau_{rel}D_{GT}(u^*)).
]

首版可设：

```text
tau_abs = scene_scale × 1e-3
tau_rel = 0.01
```

同时过滤：

* GT alpha 低；
* 深度不一致；
* 投影靠近遮挡边界；
* 多个 anchor 落到同一局部区域。

---

## 3. identity contrastive labels 会产生 false negatives

当前 descriptor loss 把第 (j) 个 rendered anchor 和第 (j) 个 query feature 设为唯一正样本，其他所有 anchor 都作为负样本。([GitHub][1])

对于重复纹理、相邻平面或多个 anchor 投影到相近区域的情况，这会把合理的相似特征错误推开。对 ShopFacade 这类外立面场景尤其危险。

需要改成：

* 同一 Gaussian 的多视图 observation 都是 positives；
* 目标投影距离小于一定阈值的样本不作为 negative；
* 3D 邻域很近的 Gaussian 先进入 ignore set；
* hard negatives 来自空间不同但描述子相似的 Gaussian。

---

# 三、先用现有 checkpoint 定位到底是哪一阶段造成退化

不要马上重跑完整 Phase1–6。先对现有 30k、33k、36k、39k、40k checkpoint 做受控评测。

| 实验 | Gaussian map | Landmark                  | Detector              | 目的                                 |
| -- | ------------ | ------------------------- | --------------------- | ---------------------------------- |
| E0 | baseline 30k | baseline idx              | baseline detector     | 原始基线                               |
| E1 | Phase1 33k   | **baseline idx**          | **baseline detector** | 只检查 descriptor/loc-opacity 更新      |
| E2 | Phase1 33k   | baseline sampling 重算      | hard detector 重训      | 检查 Phase1 map                      |
| E3 | final 40k    | baseline spatial sampling | hard detector         | 检查最终 map，移除 LA sampler/soft target |
| E4 | final 40k    | LA global-topk            | hard detector         | 单独检查 LA sampling                   |
| E5 | final 40k    | baseline spatial sampling | soft detector         | 单独检查 soft target                   |
| E6 | final 40k    | LA sampling               | soft detector         | 当前完整结果                             |

最关键的是 E1 和 E3。

### 预期判读

* **E1 已经下降**：dense teacher/feature optimization 有问题；
* **E1 持平或提升，但 E4 下降**：主要问题是 landmark sampling；
* **E3 正常，E5 下降**：soft detector 是主要问题；
* **36k 才开始下降**：geometry drift；
* **39k 才下降**：soft-prune/topology 逻辑；
* **40k 才下降**：closed-loop pose distribution 或 self-training。

此外补跑：

```text
baseline sparse
baseline dense
LA sparse
LA dense
```

否则无法判断 LA dense field 是否真的比原版好。

---

# 四、下一版应先实现的最小正向证据

我建议暂时把完整方案退回为 **LA-STDLoc-v0.2 Feature Distillation**。

## 固定不变

* Gaussian xyz、scale、rotation；
* RGB/feature opacity；
* Gaussian 数量；
* baseline 的 16384 个 `sampled_idx`；
* baseline detector；
* sparse solver 和所有阈值；
* 不启用 topology；
* 不启用 utility sampling；
* 不启用 soft detector；
* 不启用 prototype/rank。

这样唯一变量就是 Gaussian localization feature。

## 直接优化 sparse landmarks

每个 episode 不再先采样任意 rendered pixels，而是采样 baseline selected landmarks：

```python
landmark_idx = sampled_idx[visible_mask]
xyz = gaussians.get_xyz[landmark_idx]
uv_gt = project(xyz, pose_gt)
query_desc = bilinear_sample(query_feature_map, uv_gt)
gaussian_desc = gaussians.get_loc_feature[landmark_idx]
```

经过 target-depth visibility 检查后，使用：

[
\mathcal L_{\text{direct}}
==========================

\frac{
\sum_iw_i
\left[
1-\cos(f_i,\operatorname{sg}(q_i))
\right]
}{
\sum_iw_i+\epsilon
}.
]

再加入多视图 contrastive：

[
\mathcal L_{\text{mv}}
======================

-\sum_i
\log
\frac{
\sum_{q\in P(i)}
\exp(f_i^\top q/\tau)
}{
\sum_{q\in P(i)}
\exp(f_i^\top q/\tau)
+
\sum_{n\in N(i)}
\exp(f_i^\top n/\tau)
}.
]

此时：

* Gaussian index 是精确已知的；
* prototype 直接使用 `query_desc`；
* matching confidence 精确写回同一个 Gaussian；
* 优化对象与 sparse inference 完全一致。

这是最快能够证明核心假设的版本。

## 初始 loss 配置

```yaml
base_feature_weight: 1.0
loc_direct_weight: 0.1
loc_multiview_weight: 0.05

loc_desc_weight: 0.0
loc_reproj_weight: 0.0
loc_proto_weight: 0.0
loc_rank_weight: 0.0
loc_opacity_weight: 0.0

use_loc_opacity: false
enable_topology: false
```

先从 500、1000、2000 iteration 三个 checkpoint 评估，避免 3000 iteration 一次训练过度漂移。

---

# 五、正向证据不应只看最终 pose

需要建立三级证据。

## Level 1：descriptor 确实改善

固定 baseline landmarks，在 held-out training queries 上统计：

* GT projection 下正对应 cosine；
* positive-negative margin；
* Gaussian-to-query top-1 recall；
* MNN match precision；
* correct match 数量；
* feature drift：

[
1-\cos(f_i^{LA},f_i^{base}).
]

如果 descriptor loss 下降，但 top-1/MNN precision 不升，说明训练目标无效。

## Level 2：相同 sparse pipeline 下定位提升

固定：

* sampled indices；
* detector；
* geometry；
* PnP；

只替换 Gaussian features。

这是最有说服力的第一条正向证据。合理的工程门槛可以设为：

* 5cm/5deg recall 提升至少 2–3 个百分点，或
* median translation 明显下降且 recall 不退化；
* 同时 PnP inlier ratio 提升。

## Level 3：utility 能预测真实定位价值

对每个 Gaussian，在 held-out query 上统计真实：

* visible 次数；
* 被匹配次数；
* 正确匹配次数；
* PnP inlier 次数。

定义真实目标：

[
y_i=
\frac{\text{PnP inlier count}_i}
{\text{visible count}_i+\epsilon}.
]

然后检查：

* Spearman((Q_i^{rel},y_i))；
* utility top quartile 与 bottom quartile 的 inlier rate；
* 相比 baseline match score 是否提供额外预测能力。

如果 utility 与真实 inlier probability 不正相关，就不能让它控制 sampling、prune 或 topology。

---

# 六、feature-only 成立后再逐步恢复完整方案

## Step 1：恢复 localization-aware sampling

不要 global top-k。使用：

[
S_i=
S_i^{base}
+\lambda_qQ_i^{rel}
+\lambda_gQ_i^{geom}
-\lambda_cQ_i^{redundancy}.
]

然后仍在局部空间邻域内选点。

同时记录：

* 3D occupied voxel 数；
* landmark 最近邻距离；
* 每张训练图的 projected grid occupancy；
* (J^\top WJ) condition number；
* log-det pose information。

只有 utility sampling 在**固定 16384 点预算**下提升 sparse pose，才进入下一阶段。

## Step 2：恢复 geometry

只解锁 xyz，暂不解锁 scale/rotation：

```text
xyz_lr = baseline × 0.01
scale_lr = 0
rotation_lr = 0
```

并记录：

* median/95th percentile displacement；
* descriptor improvement；
* RGB、feature rendering 变化。

如果 xyz refinement 没有增益，再考虑 covariance。

## Step 3：正确实现 topology

先只实现 split，不做 prune：

```text
split only
physical prune off
soft loc prune off
clone off
```

必须增加断言：

```python
assert hasattr(gaussians, "densify_and_split_selected")
assert actual_split_count == requested_split_count
assert all_localization_buffers_match_point_count()
```

每次 topology event 打印：

```text
candidate count
requested split count
actual parent removed
actual children added
point count before/after
utility quantiles
```

## Step 4：最后再恢复 detector soft target

当前 soft target 会在每张图中重新 min-max 归一化 utility，并最多只保留 4096 个投影 landmarks。([GitHub][3])

这会使相同 landmark 在不同图像中的监督强度不一致，还可能丢弃大量有价值的空间覆盖。建议先继续使用原始 hard detector target；后续 soft target 改成：

* 全数据集固定 calibration；
* 不做 per-image min-max；
* utility 只作为 loss weight，不改变 GT landmark 是否存在；
* 保留所有可见 landmarks；
* 输出仍保持尖锐局部峰值。

---

# 七、closed loop 也需要修正

当前 `query_mode="mixed"` 并不真正 mixed：只要 cache 中存在成功 sparse pose，就直接使用 sparse pose；只有 cache 缺失或失败才使用 noise。([GitHub][9])

改成：

```python
if mode == "mixed":
    use_sparse = valid_sparse and torch.rand(()) < p_sparse
```

建议课程：

```text
p_sparse: 0.0 → 0.25 → 0.5 → 0.75
```

并从真实误差经验分布采样，不要只取某个 quantile 的固定误差模长。

此外，当前同一训练 camera 同时参与基础重建 loss 和 localization episode。([GitHub][6]) 更可靠的闭环应在 training images 内划分：

```text
support/mapping views
episodic query views
```

周期性交换，但一次 episode 中 query 不参与该步的 base reconstruction loss。

---

# 推荐的实际修改顺序

## 第一批必须修改

1. 修正命令行 Boolean flags；
2. 实现真正的 feature-only；
3. 关闭 loc opacity、prototype、rank；
4. 用 baseline idx 和 baseline detector 做 controlled evaluation；
5. 实现 direct landmark-to-query feature distillation；
6. prototype 改为 query observation；
7. 加 target-view depth consistency。

## 第二批

1. 将 utility 拆成 reliability/split/geometry 三类；
2. 将 per-anchor stats 精确 scatter 到 Gaussian；
3. 恢复 baseline spatial sampling；
4. 做 utility 与真实 PnP inlier 的相关性验证。

## 第三批

1. 把 topology mutation 正确移植到当前 3DGS `GaussianModel`；
2. 使用显式 split mask；
3. split-only 验证；
4. 再加入 soft prune；
5. 最后才做 physical prune、soft detector 和 closed loop。

---

## 最核心判断

当前代码已经证明了：

> 整个训练、状态保存、评测和闭环基础设施可以运行。

但它还没有严格证明：

> dense localization feedback 已经被正确归因并蒸馏到独立 sparse Gaussian landmarks。

当前性能下降最可能来自：

[
\boxed{
\text{错误/弱的 per-Gaussian credit assignment}
+
\text{global-topk landmark clustering}
+
\text{soft detector confound}
}
]

而 topology 核心功能按当前 3DGS 路径还没有真正落地。

下一步最应该追求的正向结果不是完整 Phase6 超 baseline，而是：

[
\boxed{
\text{相同 geometry}
+
\text{相同 landmarks}
+
\text{相同 detector}
+
\text{只更新 Gaussian feature}
\Rightarrow
\text{sparse PnP 改善}
}
]

一旦这条因果链成立，后续 utility、sampling、geometry 和 topology 才有可靠基础。

[1]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/dense_teacher.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/dense_teacher.py"
[2]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/scene/gaussian_model.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/scene/gaussian_model.py"
[3]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_detector.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_detector.py"
[4]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/landmark_distill.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/landmark_distill.py"
[5]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/stdloc.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/stdloc.py"
[6]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_locaware.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/train_locaware.py"
[7]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/topology_controller.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/topology_controller.py"
[8]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/correspondence.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/correspondence.py"
[9]: https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/episode_sampler.py "https://github.com/Arthurshen926/LA-STDLoc/blob/LA/localization_training/episode_sampler.py"

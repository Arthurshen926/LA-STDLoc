可以。基于对 STDLoc 原始代码的梳理，建议把方案正式定义为：

# LA-STDLoc：Localization-Aware Gaussian Mapping for Sparse-Only Relocalization

核心目标不是简单地在原有 3DGS loss 后面再加一个 matching loss，而是：

> 在建图阶段周期性构造完整的定位 episode，用可微 dense matching 评价每个 Gaussian 对定位的贡献，并据此优化 Gaussian feature、定位可见性、几何和拓扑；最终把 dense stage 的能力蒸馏进 sparse landmark map，使测试阶段只运行 sparse stage 也能获得更高精度。

现有 STDLoc 的 `train.py` 只联合优化 RGB L1、D-SSIM 和渲染特征 L1，densification 仍完全由屏幕空间位置梯度驱动；Gaussian 训练结束后才执行 matching-oriented sampling 和 detector training。([GitHub][1])
而 `stdloc.py` 中 sparse/dense 两个定位阶段都处于 `torch.no_grad()` 下，sparse stage 经过 detector、Gaussian feature matching 和 PnP，dense stage 再从初始位姿渲染 feature/depth 并做匹配。([GitHub][2])

因此最合理的实现不是直接修改现有 `STDLoc.loc_dense()`，而是新增一个训练专用的、可微的 dense localization teacher。

---

# 一、完善后的整体系统

最终系统由五个模块组成：

1. **Feature Gaussian 基础地图**
2. **Episodic Dense Localization Teacher**
3. **Gaussian Localization Statistics**
4. **Localization-Aware Topology Controller**
5. **Dense-to-Sparse Landmark/Detector Distillation**

整体训练流程为：

[
\text{基础3DGS训练}
\rightarrow
\text{定位episode微调}
\rightarrow
\text{定位感知拓扑调整}
\rightarrow
\text{landmark蒸馏}
\rightarrow
\text{detector训练}
\rightarrow
\text{sparse-only定位}
]

推理阶段保留两种模式：

* `sparse_only`：论文主结果；
* `sparse_dense`：保留原 STDLoc dense stage，作为性能上界和诊断工具。

---

# 二、首先降低 RGB 与定位目标的冲突

直接让定位 loss 修改所有 RGB Gaussian 参数，容易损害渲染质量。建议采用：

## 共享几何，分离 RGB 与定位属性

每个 Gaussian 表示为：

[
G_i=
{
\mu_i,R_i,s_i,
c_i,\alpha_i^{rgb},
f_i,\alpha_i^{loc}
}.
]

其中：

* (\mu_i,R_i,s_i)：RGB 与定位共享的中心、旋转、尺度；
* (c_i,\alpha_i^{rgb})：RGB 渲染属性；
* (f_i,\alpha_i^{loc})：定位 feature 和定位 opacity。

当前 STDLoc 已经单独执行一次 feature rasterization，但 feature pass 仍使用 RGB opacity 和 scale，而且 feature rasterization 返回的 `meta` 没有传出；当前函数最后返回的是 RGB pass 的 `info["means2d"]`。([GitHub][3])

建议修改为：

[
\text{RGB pass uses }\alpha_i^{rgb},
\qquad
\text{feature pass uses }\alpha_i^{loc}.
]

这样可以形成三级控制：

1. **定位软剪枝**：降低 (\alpha_i^{loc})，但保留 RGB Gaussian；
2. **landmark 排除**：不再把该 Gaussian 选入 sparse map；
3. **物理剪枝**：只有 RGB 和定位都认为无价值时才真正删除。

这是提升技术稳定性的关键设计。它能防止“某个 Gaussian 对 RGB 有价值、但对定位有害”时，定位 loss 直接破坏重建。

初始化时：

[
\alpha_i^{loc}\leftarrow\alpha_i^{rgb},
]

前期冻结或强约束两者，随后逐渐放开。

---

# 三、训练期 Localization Episode

## 3.1 Query 与初始位姿

每个 localization episode 采样一张训练 query：

[
(I_q,F_q,T_q^*,K_q),
]

其中：

* (F_q) 由原 STDLoc 的冻结 feature extractor 提取；
* (T_q^*) 是训练图像真值位姿；
* (T_q^0) 是模拟 sparse stage 得到的初始位姿。

初期使用：

[
T_q^0=\exp(\xi)T_q^*.
]

但不建议手工固定一个噪声范围。更成熟的做法是：

1. 首先运行原始 STDLoc sparse stage；
2. 在训练集内部的 episodic holdout 子集上统计 sparse 位姿误差；
3. 建立平移和旋转误差分布；
4. 训练初期采样误差分布的低—中分位；
5. 后期逐渐覆盖到 90% 分位；
6. 最后混入真实 sparse PnP 输出。

这样训练时的 pose prior 与实际 sparse stage 的误差分布一致。

---

## 3.2 从初始位姿生成 dense 对应监督

从 (T_q^0) 渲染：

* feature map (F_r)；
* expected depth (D_r)；
* alpha map；
* feature-pass projected Gaussian means。

对渲染像素 (x_j)，根据深度反投影：

[
X_j=
(T_q^0)^{-1}
\left[
D_r(x_j)K_q^{-1}\bar{x}_j
\right].
]

再使用真值位姿投影至 query：

[
u_j^*=\pi(K_qT_q^*X_j).
]

于是得到：

[
\left(
F_r(x_j),F_q(u_j^*),X_j
\right)
]

这一组训练期 dense 2D–3D 对应。

为了避免错误监督，需要过滤：

* alpha 过低；
* 投影越界；
* 深度为零或 NaN；
* 动态物体区域；
* 畸变无效区域；
* 在真值视角中发生遮挡的点；
* 多视图 feature 不一致的点。

真值视角的 depth 可以复用该 iteration 的正常训练渲染，因此每个 localization episode 通常只需要额外渲染一次 (T_q^0)。

早期训练中应对 (D_r,X_j,u_j^*) 执行 `detach()`，避免模型通过移动监督目标而“作弊”。Gaussian 几何仍可通过 feature rasterization 和 projected means 的梯度更新。

---

# 四、可微 Dense Teacher

不建议直接复用带有 `@torch.no_grad()`、MNN、top-k 和 Poselib/OpenCV PnP 的 `loc_dense()`。应新增：

```text
localization_training/
    dense_teacher.py
    correspondence.py
    pose_refiner.py
```

## 4.1 可微 coarse matching

每个 episode 不做整张高分辨率 all-pairs correlation，而是：

* 从有效渲染区域分层采样 (M) 个 anchor；
* 室内初始设为 512；
* 室外初始设为 1024；
* 保证 anchor 在图像空间均匀分布；
* 去除距离过近、目标投影重复的样本。

构造：

[
S_{jk}
======

\frac{
F_r(x_j)^\top F_q(u_k^*)
}{\tau}.
]

采用对称交叉熵：

[
\mathcal L_{\text{desc}}
========================

\frac{1}{2}
\left[
\operatorname{CE}(S,I)
+
\operatorname{CE}(S^\top,I)
\right].
]

也可以沿用 STDLoc 的 dual-softmax，但训练时不执行 hard MNN。

## 4.2 可微 fine matching

在 query feature map 上为每个 anchor 建立局部窗口：

[
\Omega_j={u:|u-u_j^*|_\infty\le r}.
]

计算：

[
P_j(u)
======

\operatorname{softmax}
\left(
F_r(x_j)^\top F_q(u)/\tau_f
\right),
]

并通过 soft-argmax 获得：

[
\hat u_j=\sum_{u\in\Omega_j}P_j(u)u.
]

对应损失为：

[
\mathcal L_{\text{reproj}}
==========================

\sum_j c_j
\operatorname{Huber}
(\hat u_j-u_j^*).
]

其中置信度：

[
c_j=
c_j^{alpha}
c_j^{visibility}
c_j^{cycle}
c_j^{entropy}.
]

fine matching 使用课程学习：

* 前期窗口以 (u_j^*) 为中心；
* 中期混合真值中心和 coarse prediction；
* 后期主要以 coarse prediction 为中心。

这样可以防止早期训练因 coarse matching 完全失败而没有有效监督。

---

# 五、Gaussian feature 的定位感知更新

原 STDLoc 的 feature L1 保证渲染 feature map 接近图像 feature，但不能保证每个独立 Gaussian 都是优秀的 sparse descriptor。

建议为每个 Gaussian 建立多视图 prototype：

[
m_i
\leftarrow
\beta m_i
+
(1-\beta)
\frac{
\sum_v c_{iv}F_v(\pi(T_v\mu_i))
}{
\sum_v c_{iv}+\epsilon
}.
]

只使用：

* 可见；
* 非遮挡；
* teacher 匹配置信度高；
* cycle consistency 通过；

的观测更新 prototype。

加入：

[
\mathcal L_{\text{proto}}
=========================

\sum_i
w_i
\left(
1-
\cos(f_i,\operatorname{sg}(m_i))
\right).
]

其中 `sg` 表示 stop-gradient。

再加入局部 hard-negative ranking：

[
\mathcal L_{\text{rank}}
========================

\max
\left(
0,
m+
s(f_i,f_i^-)
------------

s(f_i,m_i)
\right),
]

其中 (f_i^-) 优先从以下 Gaussian 中选择：

* 3D 空间邻近；
* 外观相似；
* 但投影不对应；
* 当前 sparse matching 中容易被误匹配。

这样更新后的 Gaussian feature 才真正面向 sparse 2D–3D matching，而不是只面向 feature rendering。

---

# 六、训练损失

基础版本建议使用：

[
\begin{aligned}
\mathcal L
=&;
\mathcal L_{\text{rgb}}
+\lambda_{\text{ssim}}\mathcal L_{\text{ssim}}
+\lambda_{\text{feat}}\mathcal L_{\text{feat-render}}
\
&+
\lambda_d\mathcal L_{\text{desc}}
+\lambda_r\mathcal L_{\text{reproj}}
+\lambda_p\mathcal L_{\text{proto}}
+\lambda_h\mathcal L_{\text{rank}}
\
&+
\lambda_o\mathcal L_{\text{loc-opacity}}
+\lambda_a\mathcal L_{\text{geometry-anchor}}.
\end{aligned}
]

其中：

### 定位 opacity 正则

[
\mathcal L_{\text{loc-opacity}}
===============================

\lambda_s\sum_i\alpha_i^{loc}
+
\lambda_c
\left|
\frac{1}{N}\sum_i\alpha_i^{loc}
-\rho
\right|.
]

第一项控制定位地图稀疏度，第二项防止所有定位 opacity 一起坍缩为零。

### 几何锚定

保存基础地图参数 (\mu_i^0,s_i^0,R_i^0)，定义：

[
\mathcal L_{\text{geometry-anchor}}
===================================

\sum_i
w_i
\left[
|\mu_i-\mu_i^0|_2^2
+
\lambda_s
|\log s_i-\log s_i^0|_2^2
+
\lambda_R
d(R_i,R_i^0)^2
\right].
]

它只在解锁 geometry 后启用。

首轮工程默认值可以设为：

```yaml
loss:
  desc: 1.0
  reproj: 0.1
  prototype: 0.1
  ranking: 0.05
  loc_opacity: 0.001
  geometry_anchor: 0.01
  pose: 0.0
```

这些只是初始配置，需要通过消融确定。

---

# 七、是否需要可微 PnP

## 主版本：不依赖可微 PnP

原 STDLoc 最终使用 hard matching 和 PnP/RANSAC。由于离散 top-k、MNN、RANSAC 和 NumPy/外部求解过程不适合作为初期反向传播路径，应继续把它们作为：

* 周期性评估器；
* 定位统计生成器；
* 真实 sparse 初始位姿生成器。

训练主体使用 descriptor 和 reprojection surrogate loss。

## 成熟增强版：展开 weighted Gauss–Newton

在 matching 已稳定后，可加入一个纯 PyTorch 的三步 pose refiner：

[
\delta\xi
=========

(J^\top WJ+\lambda I)^{-1}J^\top Wr.
]

迭代：

[
T^{k+1}
=======

\exp(\delta\xi^k)T^k.
]

加入：

[
\mathcal L_{\text{pose}}
========================

\lambda_t
|t-t^*|_1
+
\lambda_R
|\log(RR^{*T})|_1.
]

需要：

* 对 (J^\top WJ) 加 damping；
* 丢弃条件数过高的 episode；
* 只使用 teacher confidence 高的 top-K 对应；
* 初期对 (X_j) detach；
* `lambda_pose` 从 0 缓慢增加。

这个模块应作为后期增强，而不是 MVP 的前置依赖。

---

# 八、定位信息如何归因到 Gaussian

这是整个方案落地的核心。

gsplat rasterization 会返回 `meta["means2d"]`，并支持其反向梯度；新版本接口还支持可选的 `absgrad`。([docs.gsplat.studio][4])

当前 STDLoc feature rasterization 已获得 `meta`，但没有返回它。([GitHub][3])

需要修改 renderer，使其返回：

```python
{
    ...
    "loc_viewspace_points": feat_meta["means2d"],
    "loc_radii": feat_meta["radii"].squeeze(0),
    "loc_visible_idx": visible_idx,
    "loc_alphas": feat_alphas,
    "loc_K": K,
    "loc_viewmat": viewmat,
}
```

然后在训练中计算仅由定位 loss 导致的梯度：

```python
loc_means_grad = torch.autograd.grad(
    outputs=loc_loss,
    inputs=render_pkg["loc_viewspace_points"],
    retain_graph=True,
    allow_unused=True,
)[0]
```

再把 feature pass 中可见子集的梯度 scatter 回全局 Gaussian 索引：

```python
gaussians.add_localization_stats(
    full_idx=render_pkg["loc_visible_idx"],
    means2d_grad=loc_means_grad,
    radii=render_pkg["loc_radii"],
    episode_stats=episode_stats,
)
```

建议首版使用普通 `.grad`。只有确认项目实际使用的 gsplat 版本支持并通过单元测试后，再启用 `absgrad=True`。

---

# 九、每个 Gaussian 需要维护的定位统计

增加以下 EMA buffer：

```text
loc_grad_accum           定位loss对2D中心的梯度
loc_grad_denom
loc_observation_count
loc_repeatability_ema
loc_positive_prob_ema
loc_margin_ema
loc_entropy_ema
loc_outlier_ema
loc_reproj_error_ema
loc_information_ema
loc_redundancy_ema
loc_prototype
loc_prototype_count
birth_iteration
last_topology_iteration
```

其中：

* `repeatability`：可见时能否在不同视角稳定获得正确匹配；
* `positive_prob`：teacher 分给真值邻域的概率质量；
* `margin`：正确对应与最强错误对应的相似度差；
* `entropy`：对应是否模糊；
* `outlier`：周期性 hard PnP 中是否经常成为 outlier；
* `information`：对 pose estimation 的几何信息贡献；
* `redundancy`：附近是否已有大量近似 landmark。

定位效用定义为：

[
U_i=
w_rz(R_i)
+w_mz(M_i)
+w_pz(P_i)
+w_Iz(I_i)
-w_hz(H_i)
-w_oz(O_i)
-w_cz(C_i).
]

这里 (z(\cdot)) 不建议直接使用均值方差标准化，而应使用：

* 中位数；
* MAD；
* 分位数截断；

避免少数异常 Gaussian 破坏统计。

所有拓扑操作必须满足：

[
\text{observation count}\ge n_{\min}.
]

首轮可设 `n_min=8`。

---

# 十、Pose information 的计算

仅依赖 feature distinctiveness 会导致 landmark 集中在一个局部区域，PnP 几何条件仍可能很差。

对 Gaussian (i)：

[
J_i=
\frac{\partial \pi(T\mu_i)}
{\partial \xi}
\in\mathbb R^{2\times6}.
]

当前 landmark 集合的信息矩阵：

[
H=
\sum_iw_iJ_i^\top J_i+\lambda I.
]

Gaussian (i) 的边际信息增益：

[
\Delta I_i
==========

\log\det
(H+w_iJ_i^\top J_i)
-------------------

\log\det H.
]

可以利用矩阵行列式引理，把每个候选点的计算化成 (2\times2) 矩阵：

[
\Delta I_i
==========

\log\det
\left(
I_2+w_iJ_iH^{-1}J_i^\top
\right).
]

实际只对高 utility 的候选池计算，不需要覆盖全部 Gaussian。

它主要用于：

* landmark selection；
* clone 控制；
* prune 保护；
* 空间覆盖约束；

不必在最初版本中作为可微 loss。

---

# 十一、定位感知增长、分裂和剪枝

原始 STDLoc 的 `densify_and_prune()` 只计算累计屏幕梯度，随后执行 clone、split，并按 opacity 和尺寸剪枝。([GitHub][5])
当前 split/clone 还会直接复制父 Gaussian 的 localization feature。([GitHub][5])

应改为混合控制器。

## 11.1 Split

Gaussian 满足以下条件时进行定位驱动 split：

* 定位观测次数充分；
* projected radius 较大；
* `loc_grad` 位于高分位；
* 或 correspondence entropy 持续较高；
* repeatability 较高，说明它不是纯噪声；
* 多视图证据表明同一个 Gaussian 承担多个不同对应。

推荐判据：

[
\begin{aligned}
\text{split}*i =
&;
R_i^{2D}>\tau_r
\
&\land
\left(
G_i^{loc}>Q*{0.95}(G^{loc})
\lor
H_i>Q_{0.90}(H)
\right)
\
&\land
R_i^{repeat}>\tau_{repeat}.
\end{aligned}
]

分裂方向：

1. 首版沿 covariance 最大轴，复用原实现；
2. 后续改为使不同视图 correspondence entropy 下降最大的方向。

## 11.2 Clone

定位驱动 clone 默认关闭。

因为小 Gaussian 的直接复制会产生：

* 几乎相同的 3D 点；
* 相同 descriptor；
* 几何冗余；
* PnP 信息增益接近零。

只有同时满足以下条件才 clone：

* 当前区域 landmark 覆盖不足；
* Gaussian utility 高；
* pose information 边际增益高；
* clone 后允许空间扰动；
* 不是单纯 descriptor duplication。

原有 photometric clone 逻辑可以保留。

## 11.3 Localization-only prune

对于定位不可靠但 RGB 有价值的 Gaussian：

[
\alpha_i^{loc}\rightarrow0,
]

但不删除该 Gaussian。

这是默认的 prune 方式。

## 11.4 Physical prune

只有满足以下条件才物理删除：

[
\left(
\alpha_i^{rgb}<\tau_{rgb}
\land
\alpha_i^{loc}<\tau_{loc}
\land
U_i<\tau_U
\right),
]

或者：

* 在足够多视图中持续被判定为动态/遮挡；
* 长期高 outlier；
* pose information 接近零；
* 与邻近 Gaussian 高度冗余。

不能因为定位残差高就直接剪掉。高残差、大 footprint、但 repeatability 高的 Gaussian 更可能应该 split。

---

# 十二、Split 后 feature 的初始化

不能长期让所有 child 直接复制同一个 feature。

稳妥流程是：

1. split 时先复制父 feature，保证渲染连续性；
2. 为 children 建立 sibling group；
3. 设置 topology cooldown；
4. 在随后若干 feature-only iteration 中，根据不同 child 的多视图 responsibility 分别更新 prototype；
5. 只有当两个 child 对应不同的稳定 image feature 时，才施加 sibling descriptor separation。

不建议在 split 时直接加入大幅随机噪声。

可以加入很小的正交扰动：

[
f_i^{child}
===========

\operatorname{normalize}
(f_i+\epsilon v_i),
\quad
v_i^\top f_i=0,
]

但默认 (\epsilon) 应接近零。

---

# 十三、Dense-to-Sparse Landmark Distillation

当前 `calculate_match_score()` 把 Gaussian 中心投影到训练图像，用 Gaussian feature 与对应图像 feature 的 cosine similarity 作为匹配分数；matching-oriented sampling 再在空间邻域中选择高分点。([GitHub][6])

建议保留原来的空间覆盖框架，只替换评分函数：

[
S_i^{landmark}
==============

z(C_i^{base})
+
\lambda_rz(R_i)
+
\lambda_mz(M_i)
+
\lambda_pz(P_i)
+
\lambda_Iz(I_i)
---------------

## \lambda_hz(H_i)

## \lambda_oz(O_i)

\lambda_cz(C_i^{redundancy}).
]

首版仍输出：

```text
detector/sampled_idx.pkl
```

从而不修改后续 sparse matching 的接口。

同时新增：

```text
detector/landmark_meta.pt
```

其中包含：

```python
{
    "indices": sampled_idx,
    "utility": utility,
    "repeatability": repeatability,
    "margin": margin,
    "information": information,
    "prototype": prototype,
    "version": 1,
}
```

对比实验必须固定：

* landmark 数量仍为 16384；
* 相同 feature extractor；
* 相同 PnP solver；
* 相同 detector top-k；
* 相同 RANSAC 参数。

否则很难证明提升来自地图学习本身。

---

# 十四、Detector 蒸馏

当前 detector GT 是将 sampled Gaussian 中心投影后生成二值 keypoint map，并用 BCE 训练。([GitHub][6])

建议改为 utility-aware soft target：

[
Y(u)
====

\max_i
\left[
\tilde U_i
\exp
\left(
-\frac{|u-\pi(T\mu_i)|^2}
{2\sigma_i^2}
\right)
\right].
]

其中：

* (\tilde U_i) 是归一化 utility；
* (\sigma_i) 可设为 1–2 个 feature pixels；
* 仅投影当前可见且深度一致的 landmark。

损失使用：

[
\mathcal L_{\text{det}}
=======================

\operatorname{FocalBCE}
(\hat Y,Y).
]

第一版不联合训练 detector 与 Gaussian map。仍然保持：

1. 地图优化完成；
2. 固定 landmark；
3. 独立训练 detector。

这与当前 STDLoc 的后处理结构兼容性最高。当前代码也是在 Gaussian 训练完成后调用 30k iteration 的 detector training。([GitHub][1])

---

# 十五、建议的完整代码结构

保留原始 `train.py`，不要直接覆盖，保证 baseline 可复现。

```text
STDLoc/
├── train.py                         # 原始baseline，不修改主逻辑
├── train_locaware.py                # 新训练入口
├── train_detector.py                # 扩展utility sampling与soft target
├── stdloc.py                        # 仅做兼容性和sparse_only修改
│
├── localization_training/
│   ├── __init__.py
│   ├── episode_sampler.py
│   ├── dense_teacher.py
│   ├── correspondence.py
│   ├── losses.py
│   ├── pose_refiner.py
│   ├── pose_information.py
│   ├── gaussian_stats.py
│   ├── topology_controller.py
│   ├── landmark_distill.py
│   └── feature_cache.py
│
├── gaussian_renderer/
│   └── __init__.py                  # 返回feature-pass meta
│
├── scene/
│   └── gaussian_model.py            # loc opacity与统计同步
│
├── configs/
│   └── locaware/
│       ├── 7scenes.yaml
│       └── cambridge.yaml
│
├── scripts/
│   ├── train_locaware_7scenes.sh
│   ├── train_locaware_cambridge.sh
│   └── evaluate_sparse_locaware.sh
│
└── tests/
    ├── test_renderer_loc_grad.py
    ├── test_episode_geometry.py
    ├── test_topology_stats.py
    ├── test_checkpoint_compat.py
    └── test_baseline_regression.py
```

---

# 十六、各文件的具体修改

## `gaussian_renderer/__init__.py`

新增参数：

```python
def render_gsplat(
    ...,
    return_loc_meta: bool = False,
    use_loc_opacity: bool = False,
    loc_absgrad: bool = False,
):
```

新增返回字段：

```python
result.update({
    "loc_viewspace_points": feat_meta["means2d"],
    "loc_radii": feat_meta["radii"].squeeze(0),
    "loc_visible_idx": visible_idx,
    "loc_alphas": feat_alphas,
})
```

`render_from_pose_gsplat()` 同步修改。

必须注意：

* feature rasterization 只使用 RGB pass 中的 visible subset；
* `loc_viewspace_points` 的索引不是全局 Gaussian 索引；
* 必须通过 `loc_visible_idx` scatter 回去；
* `packed=False` 首版保持不变。

## `scene/gaussian_model.py`

新增：

```python
self._loc_opacity
self.loc_grad_accum
self.loc_grad_denom
self.loc_observation_count
self.loc_repeatability_ema
self.loc_positive_prob_ema
self.loc_margin_ema
self.loc_entropy_ema
self.loc_outlier_ema
self.loc_information_ema
self.loc_prototype
self.loc_birth_iteration
```

新增接口：

```python
get_loc_opacity
init_localization_state()
add_localization_stats()
compute_localization_utility()
localization_aware_densify_and_prune()
save_localization_state()
load_localization_state()
```

所有以下操作都必须同步新 buffer：

* `prune_points`
* `densification_postfix`
* `densify_and_clone`
* `densify_and_split`
* checkpoint restore

当前 GaussianModel 的 optimizer 中已有独立 `loc_feature` 参数组，split/clone 也会复制 loc feature，因此可以沿用同一套 optimizer tensor 扩展机制。([GitHub][5])

## `train_locaware.py`

主循环建议为：

```python
for iteration in range(first_iter, total_iter + 1):
    # 1. 原始RGB / feature重建
    base_pkg = render_gsplat(
        viewpoint_cam,
        gaussians,
        background,
        rgb_only=False,
        return_loc_meta=run_loc_episode,
    )
    base_loss = compute_base_loss(base_pkg, viewpoint_cam)

    loc_loss = 0.0
    loc_stats = None

    # 2. 周期性定位episode
    if run_loc_episode:
        episode = episode_sampler.sample(
            query_camera=viewpoint_cam,
            sparse_pose_cache=sparse_pose_cache,
        )

        teacher_out = dense_teacher(
            gaussians=gaussians,
            query_features=episode.query_features,
            pose_init=episode.pose_init,
            pose_gt=episode.pose_gt,
            intrinsics=episode.K,
            gt_depth=base_pkg.get("depth"),
        )

        loc_loss, loc_stats = localization_losses(teacher_out)

        # 3. 只提取定位loss对feature-pass means2d的梯度
        loc_grad = torch.autograd.grad(
            loc_loss,
            teacher_out.loc_viewspace_points,
            retain_graph=True,
            allow_unused=True,
        )[0]

    total_loss = base_loss + loc_weight * loc_loss
    total_loss.backward()

    with torch.no_grad():
        gaussians.add_densification_stats_gsplat(...)
        if loc_stats is not None:
            gaussians.add_localization_stats(
                loc_grad=loc_grad,
                stats=loc_stats,
            )

        if topology_controller.should_update(iteration):
            topology_controller.update(gaussians, iteration)

    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
```

## `train_detector.py`

增加：

```python
sampling_mode = {
    "baseline",
    "localization_aware",
}
```

新增函数：

```python
localization_aware_sample(...)
generate_soft_gt_map(...)
utility_weighted_detector_loss(...)
```

原始模式必须保留，以便做严格消融。

## `stdloc.py`

尽量少改：

* 继续使用原始 detector；
* 继续加载 `sampled_idx.pkl`；
* 可选加载 `landmark_meta.pt`；
* 新增 `sparse_only`；
* 可选把 landmark reliability 作为 feature correlation 的先验；
* 保留原始 hard PnP 和 dense stage。

例如：

```python
if self.config.get("sparse_only", False):
    return {"sparse": sparse_result, "dense": []}
```

主结果不应依赖修改后的 PnP solver。

---

# 十七、Checkpoint 与兼容性

当前 `GaussianModel.capture()` 使用固定 tuple 保存模型参数，并包含 `_loc_feature`。直接向 tuple 中插入新字段会破坏旧 checkpoint。([GitHub][5])

推荐保存为：

```python
{
    "version": 2,
    "iteration": iteration,
    "model_params": gaussians.capture(),
    "localization_state": gaussians.capture_localization_state(),
    "optimizer_state": gaussians.optimizer.state_dict(),
    "config": loc_config,
}
```

加载逻辑：

```python
if isinstance(checkpoint, tuple):
    # 原始STDLoc checkpoint
    gaussians.restore(checkpoint[0], opt)
    gaussians.init_localization_state(from_rgb_opacity=True)
else:
    # 新checkpoint
    gaussians.restore(checkpoint["model_params"], opt)
    gaussians.restore_localization_state(
        checkpoint["localization_state"]
    )
```

PLY 格式首版保持不变。新增状态保存为：

```text
point_cloud/iteration_x/
    point_cloud.ply
    loc_state.pt
```

这样原始 STDLoc 仍能读取 PLY。

---

# 十八、推荐训练阶段

## Phase 0：严格复现原始 STDLoc

使用原始代码完成：

* 30k Feature Gaussian training；
* matching-oriented sampling；
* 30k detector training；
* sparse 和 dense evaluation。

官方仓库明确给出了当前代码版本的训练入口和复现实验结果，因此这一阶段必须作为回归基线。([GitHub][7])

验收条件：

* 原始 checkpoint 可加载；
* sparse/dense 指标与仓库结果处于合理波动范围；
* 保存所有训练 query 的 sparse pose、inlier 和误差分布。

## Phase 1：Renderer 与定位统计基础设施

不改变任何 loss，只实现：

* feature-pass metadata 返回；
* `loc_viewspace_points.grad` 检查；
* loc state 初始化、保存、加载；
* split/prune 后 shape 一致性；
* 原始训练输出不发生变化。

这是第一个必须通过的工程门槛。

## Phase 2：Feature-only localization distillation

从 30k baseline checkpoint 开始：

* 冻结 xyz、scale、rotation、RGB opacity 和 SH；
* 只优化 `loc_feature` 和 `loc_opacity`；
* 不允许 split/prune；
* 使用 descriptor、reprojection、prototype loss；
* localization episode 每 8 个 iteration 执行一次；
* 建议训练 2k–4k iteration。

验收条件：

* 固定 Gaussian topology；
* 固定 16384 landmark 数；
* sparse matching precision、PnP inlier ratio 或 sparse pose 精度提升；
* feature rendering 不退化。

这是整个项目的第一个核心 Go/No-Go 节点。

## Phase 3：低学习率 Geometry refinement

解锁：

* xyz；
* scale；
* rotation。

继续冻结：

* RGB SH；
* RGB opacity，或将其学习率降到接近零。

推荐：

```text
xyz_lr       = baseline当前值的 0.05×
scale_lr     = baseline值的 0.1×
rotation_lr  = baseline值的 0.1×
```

加入 geometry anchor，训练 2k–4k iteration。

验收条件：

* sparse pose 进一步提升；
* RGB PSNR/SSIM/LPIPS 在 baseline 波动范围内；
* 平均 Gaussian displacement 不发生异常增长。

## Phase 4：Localization-aware topology

先累计充分统计，再开启：

* localization split；
* localization-only soft prune；
* joint physical prune；
* 默认关闭 localization clone。

建议默认：

```yaml
topology:
  stats_warmup: 1000
  update_interval: 200
  min_observations: 8
  split_quantile: 0.95
  ambiguity_quantile: 0.90
  growth_cap_per_event: 0.03
  total_point_budget_ratio: 1.25
  cooldown_iterations: 300
  enable_loc_clone: false
```

即：

* 每次最多增长当前点数的 3%；
* 总点数不超过 baseline 的 1.25 倍；
* child 300 iteration 内不再次 split/prune。

## Phase 5：Landmark 与 Detector 蒸馏

冻结 Gaussian map：

1. 计算所有 Gaussian 的最终 utility；
2. 选取固定 16384 landmarks；
3. 保存 `sampled_idx.pkl` 和 `landmark_meta.pt`；
4. 使用 soft GT map 训练 detector；
5. 首次仍沿用原来的 30k detector schedule。

## Phase 6：真实 Sparse Pose 闭环

使用 Phase 5 detector 和 landmark map，在训练 episodic holdout 上实际运行 sparse stage：

* 缓存真实 PnP 初始位姿；
* 缓存 inlier/outlier；
* 缓存失败 query；
* 以这些结果替换部分人工扰动 (T^0)；
* 短周期重新微调 map；
* 再次采样 landmarks 并微调 detector。

这一阶段才构成完整意义上的：

[
\text{sparse localization}
\rightarrow
\text{dense teacher}
\rightarrow
\text{map update}
\rightarrow
\text{sparse localization}.
]

为了控制复杂度，建议最多执行一次完整闭环，先验证边际收益。

---

# 十九、首轮配置模板

```yaml
localization:
  enabled: true
  interval: 8
  query_mode: mixed
  pose_noise_source: sparse_error_distribution

  anchors:
    indoor: 512
    outdoor: 1024
    min_spacing: 3
    alpha_threshold: 0.2

  matching:
    coarse_temperature: 0.07
    fine_temperature: 0.05
    fine_window_radius: 4
    use_dual_softmax: true
    use_hard_mnn: false

  statistics:
    ema_decay: 0.95
    min_observations: 8
    robust_normalization: mad

  optimization:
    feature_only_iterations: 3000
    geometry_iterations: 3000
    topology_iterations: 3000
    episode_interval: 8

  topology:
    update_interval: 200
    split_quantile: 0.95
    ambiguity_quantile: 0.90
    growth_cap: 0.03
    total_budget_ratio: 1.25
    cooldown: 300
    localization_clone: false

  differentiable_pose:
    enabled: false
    num_iterations: 3
    damping: 0.001
```

---

# 二十、实验矩阵

主实验必须逐步增加组件：

| 实验 | Localization loss | Loc opacity | Geometry | Topology | Utility sampling | Soft detector |
| -- | ----------------: | ----------: | -------: | -------: | ---------------: | ------------: |
| A  |                 否 |           否 |       原始 |       原始 |               原始 |             否 |
| B  |                 是 |           否 |       冻结 |       固定 |               原始 |             否 |
| C  |                 是 |           是 |       冻结 |       固定 |               原始 |             否 |
| D  |                 是 |           是 |       解锁 |       固定 |               原始 |             否 |
| E  |                 是 |           是 |       解锁 |     定位感知 |               原始 |             否 |
| F  |                 是 |           是 |       解锁 |     定位感知 |                是 |             否 |
| G  |                 是 |           是 |       解锁 |     定位感知 |                是 |             是 |
| H  |                全部 |           是 |       解锁 |     定位感知 |                是 |  是 + sparse闭环 |

需要报告：

### 定位指标

* sparse-only median translation/rotation；
* 不同误差阈值 recall；
* 定位失败率；
* PnP inlier ratio；
* 参与 PnP 的有效对应数量；
* pose information matrix condition number；
* RANSAC iteration 和时间。

### 地图指标

* Gaussian 总数；
* landmark 数固定为 16384；
* feature/loc-state 存储；
* 训练显存；
* sparse inference 时间；
* dense inference 时间。

### 重建指标

* PSNR；
* SSIM；
* LPIPS；
* feature-map L1；
* 几何漂移；
* floater 数量或深度一致性。

### 公平性

必须固定：

* feature backbone；
* detector 输入分辨率；
* landmark 数；
* sparse top-k；
* PnP solver；
* reprojection threshold；
* query split；
* 随机种子。

---

# 二十一、必须实现的单元测试

## Renderer 梯度测试

验证：

```python
assert loc_viewspace_points.requires_grad
assert loc_grad is not None
assert torch.isfinite(loc_grad).all()
```

## Episode 几何测试

使用两台合成相机和一个已知 3D 点，验证：

[
x\rightarrow X\rightarrow u^*
]

的投影误差小于数值精度阈值。

## Topology 同步测试

每次 split、clone、prune 后验证：

```python
N == xyz.shape[0]
N == loc_feature.shape[0]
N == loc_opacity.shape[0]
N == loc_grad_accum.shape[0]
N == loc_prototype.shape[0]
```

## Checkpoint 兼容测试

验证：

* 原始 tuple checkpoint 能加载；
* 新 dict checkpoint 能加载；
* 缺少 `loc_state.pt` 时自动初始化；
* 保存后重新加载 sparse 结果一致。

## Baseline regression

关闭所有 localization 选项后：

* 原始 loss 数值一致；
* Gaussian 数量变化一致；
* 输出图像和 sparse 结果在浮点误差范围内一致。

---

# 二十二、主要风险和预案

## 1. Dense matching 显存过高

预案：

* 不做 full-resolution all-pairs；
* 使用 512/1024 anchor；
* query candidate chunking；
* fine stage 只做局部窗口；
* feature cache；
* AMP。

## 2. Feature-pass means2d 没有梯度

预案：

* 在 feature rasterization 后立即 `retain_grad()`；
* 单独返回 feature `meta`；
* 确保 loss 确实依赖 feature rasterization；
* 首版不用 `absgrad`；
* 不误用 RGB pass 的 `info["means2d"]`。

## 3. 地图通过移动几何“追随监督”

预案：

* target depth 和 (u^*) detach；
* query backbone 冻结；
* geometry 后期开启；
* geometry anchor；
* 更低 xyz LR；
* 保留 RGB/feature-render losses。

## 4. 拓扑无限增长

预案：

* 最小观测次数；
* 分位数阈值；
* child cooldown；
* 单次增长上限；
* 总点数预算；
* 连续多次满足条件才 split。

## 5. 定位目标损害 RGB

预案：

* 独立 localization opacity；
* localization-only soft prune；
* RGB opacity 与定位 opacity 联合决定物理 prune；
* RGB/geometry loss 始终保留。

## 6. PnP 不可微

预案：

* 主版本使用 descriptor/reprojection surrogate；
* hard PnP 只做评估和统计；
* weighted Gauss–Newton 作为后期增强。

## 7. 训练 query 自我记忆

预案：

* 训练集内部划分 support/query episode；
* 优先按序列或相机轨迹划分；
* pose perturbation；
* feature dropout；
* 视角、尺度和模糊增强；
* 测试集永不参与地图优化。

## 8. 单地图目标仍然冲突

只有在以下现象持续存在时，才升级为双地图：

* 定位提升必然伴随明显 RGB 退化；
* 高 utility landmarks 经常具有低 RGB opacity；
* 定位 split 导致 RGB Gaussian 数量快速膨胀；
* loc opacity 无法充分解耦。

升级方案为：

[
\mathcal G_{\text{rgb}}
+
\mathcal G_{\text{loc}},
]

其中定位地图从 RGB 地图初始化，几何通过 anchor regularization 关联，但 topology 独立。它应是备选路线，不应成为第一版的工程负担。

---

# 二十三、推荐的实际开发顺序

按以下五个独立合并节点实施最稳妥：

### PR-1：基础设施

* renderer 返回 feature metadata；
* loc state；
* checkpoint 兼容；
* 单元测试；
* baseline regression。

### PR-2：Feature-only Dense Teacher

* episode sampler；
* dense correspondence；
* descriptor/reprojection/prototype losses；
* 不改 geometry 和 topology。

这是验证研究假设的最关键版本。

### PR-3：Utility Landmark Distillation

* Gaussian EMA statistics；
* localization utility；
* 新 landmark sampling；
* 保持原 detector 和 topology。

这一版本已经可能提升 sparse stage。

### PR-4：Geometry 与 Topology

* loc opacity；
* geometry anchor；
* loc-gradient densification；
* split、soft prune、joint prune；
* point budget。

### PR-5：Detector 与完整闭环

* utility soft heatmap；
* detector distillation；
* 真实 sparse pose cache；
* 一轮 sparse→dense teacher→map 更新闭环；
* 完整消融与评估。

---

## 最终建议

这个项目最可靠的主技术路线应当是：

[
\boxed{
\text{Baseline map}
+
\text{Differentiable dense matching teacher}
+
\text{Per-Gaussian localization statistics}
+
\text{Soft localization pruning}
+
\text{Conservative topology control}
+
\text{Sparse landmark distillation}
}
]

其中最先需要证明的不是 topology，而是：

> **在 Gaussian 数量、geometry、landmark 数和 detector 都不变时，仅通过 dense localization supervision 更新 Gaussian feature，是否能够稳定提升 sparse matching 和 PnP。**

一旦这个最小闭环成立，再逐步加入 loc opacity、geometry 和 split/prune，项目的技术风险、调试复杂度和论文因果链都会明显更可控。

[1]: https://github.com/zju3dv/STDLoc/blob/main/train.py "https://github.com/zju3dv/STDLoc/blob/main/train.py"
[2]: https://github.com/zju3dv/STDLoc/blob/main/stdloc.py "https://github.com/zju3dv/STDLoc/blob/main/stdloc.py"
[3]: https://github.com/zju3dv/STDLoc/blob/main/gaussian_renderer/__init__.py "https://github.com/zju3dv/STDLoc/blob/main/gaussian_renderer/__init__.py"
[4]: https://docs.gsplat.studio/main/apis/rasterization.html "https://docs.gsplat.studio/main/apis/rasterization.html"
[5]: https://github.com/zju3dv/STDLoc/blob/main/scene/gaussian_model.py "https://github.com/zju3dv/STDLoc/blob/main/scene/gaussian_model.py"
[6]: https://github.com/zju3dv/STDLoc/blob/main/train_detector.py "https://github.com/zju3dv/STDLoc/blob/main/train_detector.py"
[7]: https://github.com/zju3dv/STDLoc "https://github.com/zju3dv/STDLoc"

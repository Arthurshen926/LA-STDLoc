# 冻结方法与输入合同

版本：compact-fast-v1；状态：发布候选重构，算法不新增。这里取代旧 README 的 Base / 可选 R 主线描述。

## 科学主张

1. 现成 Gaussian 可以作为渲染观测源，转换为独立的稀疏定位资产，下游不必保留源建图 RGB 或 Gaussian。
2. Anchor 的身份来自跨视图观测关联；几何来自标定射线约束，而非直接把 Gaussian 中心当地标。
3. mapping-only 紧凑选择与简单的在线置信对应筛选，让存储、匹配成本和定位质量取得实用折中。

“source-image-free”仅指下游转换与部署：上游重建仍使用 RGB，mapping 位姿/内参也仍需要。
冻结通用 SuperPoint 不等于整个方法从未训练；不主张隐私保证、任意先验质量不敏感或全局几何最优。

## 离线流程与实现边界

| 阶段 | 操作 | 输出/注意点 |
| --- | --- | --- |
| 输入审计 | RGB-only prior + mapping cameras，核对像素域和坐标系 | 原始/矫正/渲染域明确；禁止重复去畸变 |
| 渲染观测 | RGB、alpha、depth，原生 SuperPoint | 无真实建图 RGB 特征；render-valid 与 V2 在关联前生效 |
| 关联 | reciprocal、epipolar、cycle 等一致性约束形成 tracks | 关联身份错误不是后面的三角化能必然修复 |
| 几何 | 多视图加权射线最小二乘及冻结的稳健处理 | 依赖视差和相机准确性；协方差是近似，不是误差真值 |
| 描述子 | 一个 Anchor 一个聚合后的 256D 描述子 | 没有多原型检索、训练或 query-conditioned metric |
| 候选补全 | 保留已冻结的 depth-proposed projective completion | 辅助实现，不单独包装主要贡献 |
| 紧凑选择 | 质量排序 → precision core → 逐 mapping 查询匹配覆盖 → 位姿信息/图像空间/深度/世界体素补全 | mapping camera，不是 test query 或在线反馈 |
| 导出 | XYZ + descriptor + ID + 少量来源元数据 | 丢弃在线无用的观测 CSR、协方差、构图缓存；源产物不删除 |

选择器冻结数值：core 每查询目标256、最少1024、检查步长512；matching 每查询256；体素0.5 m；
pixel variance=4、任务尺度0.05 m/5°；位姿补全最多4096、每 source 1、每 voxel 3、相对信息增益至少0.02。
这些是历史 operating point，不宣称单位尺度普适。对新数据不可悄悄按 test 调整。
位姿信息是局部线性化的可观测性代理，不保证新视角覆盖或全局定位精度。保留负结果场景。

整仓重构后，构图由 `python -m anygsloc build-map` 和 `anygsloc/mapping/` 的冻结内核承担。
选择器在compact.py，位姿helper已独立到pose_evidence.py。
主线不依赖研究树、不读旧jobs.json、不生成消融队列；历史代码实际迁入research/legacy。
全场景数值和正式性能验证仍是发布门槛。

## 在线流程

`anygsloc.online.runtime.Localizer`：冻结 SuperPoint → exact global cosine Top-1 → 65% 高分对应 → 一次 PoseLib。
保留数量 `min(N, max(256, ceil(0.65N)))`；恢复原 query 顺序，未开启 PROSAC/重复 anchor 去重。
RANSAC .999 / 250 / 30000；同一批次仅一个查询；不在线更新地图，不运行 second-pass refinement。
匹配 chunk8192、float32，归一化次数与历史零残差 metric 路径一致，坐标保留 `+0.5` pixel center。
复用 CUDA pinned host buffers；CPU PoseLib 因接口需要接收 2D/3D 对应，不搬完整地图到 CPU。
descriptor 检索、筛选、PoseLib 数值内核保持原算法，而非借重构更换匹配器或精度。

关键点预算沿用 mapping 图像面积推导的冻结值（历史限制1024–2048）；重投影门限沿用 mapping 相机标定规则。
新运行时显式接收 resolved 值，不从任意默认12 px 推断旧实验配置。正式 manifest 同时记录配置来源与解析结果。
当前轻量 CLI 提供的是准备好的 tensor 接口，不声称所有原始相机模型都已适配。

## 数据与安全合同

- 原始 RGB、已矫正 RGB、Gaussian render 三种域必须记录；每次重采样/去畸变次数及相机参数可追溯。
- 旧 loader 会自动读取存在的 `masks.pkl`，因此历史结果不能声称没有语义 mask。发布入口仅接受显式 valid mask；使用/来源/生成时间计入实验说明。
- 单独记录 source RGB 是否在下游被读取、mapping camera 数量、上游 prior 实际优化视角数量；二者未必相等。
- legacy `.pt` 导出/建图会读受信任 pickle，禁止加载不可信文件；部署格式使用 weights_only=True，仍只接受受信任资产。
- 导出精确校验 identity config/全零参数、map hash、ID 对齐，并拒绝多原型和 photometric adaptation。
- 不在没有 GT 的部署时用 GT 判定成功；success 只是求解器是否产生 inliers，不是姿态正确性的保证。

## 冻结后的修改纪律

相同输入的 Anchor 行、XYZ、描述子、匹配顺序和 PnP 输入先做回归，再决定是否重跑正式证据。
地图选择或几何变化不能叫纯工程重构；必须新版本、配对实验和新 manifest。
重构通过单元测试不等于全场景输出已经证明完全等价，尤其浮点/硬件/库版本改变时。

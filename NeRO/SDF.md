# NeRO、NeuS、NeRF、Voxel Hashing 与 Neuralangelo：3D 表示与 SDF 计算的比较说明

主要参考：

- NeRF (Mildenhall et al., 2020). [arXiv:2003.08934].
- NeuS (Wang et al., 2021) — SDF → alpha via CDF 差分与 learnable variance. [arXiv:2106.10689]
- Voxel Hashing (Nießner et al., Siggraph Asia 2013) — 稀疏体素哈希用于实时重建与大尺度内存管理.
- Neuralangelo (NVIDIA, 2023) — 多分辨率 Hash Grid + neural surface rendering.
- NeRO repository & project page (liuyuan-pal/NeRO).

## 1. 问题清单

1. NeRO 是如何表示 3D 的？
2. NeRO 在计算 SDF 的过程中是否使用了 Voxel Hashing？
3. NeRO 是否参考或采用了 NeuS 的 SDF→Occupancy/alpha 的方法（先计算 SDF，再推导 Occupancy）？
4. NeRO 是否采用 NeRF（直接拟合体密度 + view-dependent radiance）的表示方法？
5. 与其他方法（Voxel Hashing、NeuS、Neuralangelo）的对比与要点摘录。

## 2. 简短结论

- **NeRO 的 3D 表示**：主体（object）用 **隐式 SDF 网络（SDFNetwork）** 表示并放在 unit-sphere 内；背景/远场用一个单独的 **NeRF/outer_nerf**（NeRFNetwork）表示。材质/着色用独立的 shading MLP（AppShading / Directional MLP）负责 BRDF 参数与视角依赖项.
- **Voxel Hashing**：NeRO **不使用** Voxel Hashing（实现中没有哈希表 / sparse voxel grid 的相关代码），其几何和辐射场均为**连续隐式表示（MLP + positional encoding）**，并用沿射线的分层采样和 SDF→alpha 映射实现渲染。
- **NeuS 方法的借鉴**：NeRO 在 SDF→alpha 的实现中显著借鉴了 **NeuS** 的思想（使用 learnable variance / deviation 将连续 SDF 转换为段不透明度 / alpha，通过 CDF 差分近似分段占据），README/代码中也明确致谢 NeuS。
- **NeRF 的使用情况**：NeRO 在**背景/远场**部分使用 NeRF（outer_nerf）直接拟合体密度与颜色（NeRF++ 风格）；但**对象主体**并非直接用 NeRF，而是用 SDF→alpha（隐式表面）+ shading 的框架来更好处理高光/镜面。

## 3. NeRO 的 3D 表示

### 3.1 表示结构（主体/背景）

- **主体（Inner region, |x| ≤ 1）**：隐式 SDF 网络 `SDFNetwork`（MLP）表示 signed distance s(p). 该 SDF 的零等值面对应显式表面；通过 ∇s 获得表面法线并参与 shading/BRDF 计算. SDF → sigma/alpha 的转换采用 NeuS 风格的可学尺度（SingleVarianceNetwork / inv_s）进行段不透明度推导.
- **背景 / 远场（Outer region, |x| > 1）**：使用 `NeRFNetwork`（即 outer_nerf）来表示远处的辐射场（体密度 + view-dependent radiance），类似 NeRF/NeRF++ 的实现细节：当采样点位于单位球外，渲染调用 outer_nerf 来计算 color & density. 这样设计把远场复杂结构与主体隐式表面分离，便于对反射与遮挡进行物理一致的建模.

### 3.2 渲染路径（内外分支）

- Ray sampling → 判定点 p 是否落在 unit sphere 内 → 若在内走 SDF→alpha→shading → 若在外走 outer_nerf → 最后按 alpha compositing 聚合颜色. 该流程在 repo 的 `network/renderer.py` 和 raytracing 代码中可见.

## 4. NeRO 是否使用 Voxel Hashing？

**Ans：不使用。**

### 4.1 Voxel Hashing 简要说明

- Voxel Hashing（Niessner et al., Siggraph Asia 2013）是一种用于实时大规模 3D 重建的稀疏体素网格哈希结构：只在观测到数据的体素处分配内存，并通过哈希表索引实现高效插入 / 查询与流式管理，常用于基于深度传感器的在线重建与 TSDF Fusion.

### 4.2 为什么 NeRO 不用 Voxel Hashing

- NeRO 的几何与辐射场都是 **连续函数（MLP）**，并结合位置编码进行高频建模；它采用沿射线的层次化采样 + SDF→alpha 的方式而非离散体素查询。代码中没有 voxel hashing 的实现（无 hash table / voxel pool / voxel grid 文件或调用），因此确认 NeRO 不依赖 voxel hashing.

## 5. NeRO 与 NeuS：SDF→Occupancy / alpha 的对应关系

### 5.1 NeuS 的关键思想

- NeuS 提出的 trick：**用连续 SDF 的 CDF 差分来近似体积段的占据概率（alpha）**，并引入一个可学习的尺度（variance / inv_s）来控制 SDF 的“尖锐度”，从而把隐式 SDF 转为体渲染可用的 alpha. 该方法能在没有 mask 的情况下稳定地从多视图图像中恢复表面.

数学上（概念）：

- 对每个段 compute `prev_cdf = sigmoid(s_prev * inv_s)` 与 `next_cdf = sigmoid(s_next * inv_s)`, 两者差作为该段的占据质量（概率密度积分）; 最终得到 alpha = (prev_cdf - next_cdf) / prev_cdf 等形式（实现上有小量数值稳定项与角度退火）.

### 5.2 NeRO 中的 SDF→alpha 实现

- NeRO 在实现里直接借鉴并致谢 NeuS（在 README/Acknowledgements 中可见）。其 `compute_sdf_alpha`（或同名逻辑）与 NeuS 流程高度一致:
  1. SDFNetwork 输出 s(p) 与 feature_vector.
  2. SingleVarianceNetwork（或 deviation_network）输出 inv_s（learnable）.
  3. 用 CDF 差分 `prev_cdf - next_cdf` 近似段的占据质量，并用 prev_cdf 归一化得到 alpha. 实现里还有 `cos_anneal_ratio`（控制与入射角余弦的退火），以稳定早期训练.

## 6. NeRO 和 NeRF 的两者关系（NeRO 是否采用 NeRF？）

### 6.1 NeRF 的定位

- NeRF 用一个 5D（位置 + 视角）连续函数（MLP）直接输出体密度 σ 与 view-dependent radiance c, 并用体渲染沿射线累积颜色，适用于 view synthesis 与场景表示.

### 6.2 NeRO 中 NeRF 的角色

- NeRO **并没有把主体用传统 NeRF 的密度表示**; 主体使用的是 NeuS 风格的 SDF→alpha（隐式表面）。但为了更好地解释远场、背景以及环境光贡献，NeRO 在**外部区域**使用了一个 `outer_nerf`（NeRFNetwork）来拟合远场辐射场，因此技术上它是**复合表示**: SDF（主体）+ NeRF（背景）.

## 7. Neuralangelo 的方法简介与对比

### 7.1 Neuralangelo 核心点

- Neuralangelo 提出把 **多分辨率 3D hash grids（Voxel Hash / dense multi-res hash grids）** 与神经体渲染/表面渲染结合:
  - 使用稀疏/多分辨率哈希网格来存储空间特征（类似 Instant NGP 的 hash grid 思路），加速表示与训练;
  - 通过数值梯度和平滑算子 + coarse-to-fine 优化在 hash grid 上恢复高保真的表面（尤其适合大场景/高细节重建）.

### 7.2 Neuralangelo 与 NeRO / NeuS 的异同

- **与 NeRO 的不同**: Neuralangelo 使用显式的多分辨率哈希网格作为底层空间索引（高性能、可扩展），适合从 RGB Video 恢复大场景; NeRO 使用 MLP（隐式 SDF）+ outer NeRF 的混合模式，更偏向对象级精确 BRDF 恢复.
- **与 NeuS 的不同**: NeuS 强调 SDF→alpha 的几何精确性与体渲染一致性; Neuralangelo 强调用 hash grid 加速并恢复高频几何细节, 两者侧重点不同但可以互补.

## 8. Occupancy Field（占据概率）在这些方法中的差异

- **Voxel Hashing**: 原生就是离散 Occupancy（或 TSDF）在体素网格里存储, 显式、可用阈值直接二值化.
- **NeuS / NeRO**: 并不维护显式离散 Occupancy grid; 它们是**隐式的概率推导**—通过 SDF 的 CDF 差分即时计算一个段的占据概率（alpha），用于体渲染.
- **Neuralangelo (hash grid)**: 底层是稀疏/多分辨率的特征网格, 通常通过神经渲染/后处理提取等值面并得到占据/mesh; 因此它在实现上既可以兼顾显式格网的效率，也能导出显式占据.

## 9. 代码定位（NeRO repo）

- `run_training.py` — 启动脚本（参照 configs 指向 shape/material 配置）。
- `configs/shape/*.yaml`, `configs/material/*.yaml` — 训练阶段与超参配置（foreground_ratio、loss weights、inv_s 初始等）。
- `network/sdf_network.py` / `network/sdf_net.py` — SDFNetwork 类定义（multires、MLP 层、gradient 方法）。
- `network/nerf_network.py` / `network/nerf.py` — outer_nerf / NeRFNetwork（position + view encoding 与 color/density 输出）。
- `network/renderer.py` — 渲染主流程，内/外样本分支（inner_mask/outer_mask）、SDF→alpha 计算与 compositing。
- `network/loss.py` / `train/*` — photometric/eikonal/occ loss 等实现位置.

> 可用 `grep -R "compute_sdf_alpha" -n` 或 `grep -R "inv_s" -n` 搜索 SDF→alpha 实现位置.

## 10. 小结

- **NeRO** = **隐式 SDF（NeuS 风格 SDF→alpha）+ outer NeRF（背景）+ shading MLP（BRDF）**。它没有使用 voxel hashing; 而是把连续隐式表示与体渲染一致性结合起来, 专注于反射物体的几何 + 材质恢复.
- **如果加速或扩展到大场景**, 可以考虑将 SDF 的底层用 multi-resolution hash grid（Neuralangelo / InstantNGP 风格）替代 MLP, 但需处理 SDF→alpha 的数值稳定性与可微性.
- **若目标是单物体且需高质量 BRDF/relighting**, NeRO 的当前架构（隐式表面 + outer NeRF + shading）应该是合理的且受验证的.

---

## 11. 参考文献

- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. Ben Mildenhall et al., 2020. (arXiv:2003.08934).
- NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. Peng Wang et al., 2021. (arXiv:2106.10689).
- Real-time 3D Reconstruction at Scale using Voxel Hashing. M. Nießner et al., Siggraph Asia 2013.
- Neuralangelo: High-Fidelity Neural Surface Reconstruction. Z. Li et al., 2023 (NVIDIA Research).
- NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images — Project page & code (liuyuan-pal/NeRO).

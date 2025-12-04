# 基于统一层状材料模型（Unified Layered Material Model）的神经隐式重建方案

## 摘要 (Abstract)

本研究探讨在神经隐式表面重建（Neural Implicit Reconstruction）框架中，使用**单一、统一的层状 BRDF 模型**替代传统的“多头（Multi-head）”材质网络的可行性。

**结论**：该方案在理论与工程上应该可行。

相比于多头模型，统一层状模型通过参数退化（Parameter Degradation）能更自然地模拟从不透明金属到半透明介质的连续变化，并能基于“有效 BRDF（Effective BRDF）”理论解释复杂基底呈现漫反射外观的物理机制。
然而，该方案面临 **参数可辨识性（Identifiability）** 的挑战。本报告提出了一套基于 **可观测物理信号（如视角方差、空间模糊度、Beer-Lambert 残差）** 的判别机制，结合 **分阶段训练（Staged Training）** 与 **自适应门控（Adaptive Gating）** 策略，以确保模型在复杂光学解空间中的收敛稳定性。

## 1. 设计动机与物理论证 (Motivation & Rationale)

### 1.1 为什么要放弃“多头（Multi-head）”架构？

传统的“分治法”将材质分为 Diffuse、Specular、Transmission 等离散分支（Heads），但这在处理真实世界材质时存在根本性缺陷：

1. **物理连续性缺失**：真实材质很少是单一类别的，往往是混合态。多头架构在材质过渡区域（如金属渐变为氧化层）需依赖硬性分类或人工混合，易产生伪影。
2. **无法表达“有效 BRDF”**：许多看似“漫反射”的材质，本质上是**底层复杂材质被上层微结构（粗糙度/散射层）调制**的结果（例如：带涂层的金属、皮肤、玉石）。多头模型倾向于将其错误归类为“纯 Diffuse”，丢失了底层的物理信息。

### 1.2 核心优势：层状模型的退化能力

**统一层状模型**通过物理方程的叠加与参数控制，具备天然的退化能力：

* **金属**：`trans_weight`=0, `sss_weight`=0, `metallic`=1.
* **塑料**：`metallic`=0, `roughness`中等.
* **漫反射（外观）**：底层 Specular + 高 `roughness` + 厚 SSS 层 $\rightarrow$ **Effective Diffuse**。
* **半透明/液体**：`trans_weight` > 0, 由 SDF 厚度驱动吸收。

**结论**：层状模型不仅覆盖了基础材质，更能自然解释“材质 A 包裹在 B 中呈现出的外观”，这比多头模型更符合物理直觉。

## 2. 数学模型架构 (Mathematical Framework)

我们将表面点 $p$ 处沿视线 $\omega_o$ 的出射辐射 $L_o$ 定义为层状物理分量的加权和：

$$
L_o(p, \omega_o) = L_{\text{diff}} + L_{\text{spec}} + L_{\text{sss}} + L_{\text{trans}}
$$

### 2.1 表面反射项 (Surface Components)

* **漫反射 (Diffuse)**: 基于能量守恒的 Lambertian，受菲涅尔项调制。
    $$ L_{\text{diff}} = \frac{\mathbf{c}_d}{\pi} \cdot (1 - F(\omega_i, \mathbf{n})) \cdot E(p) $$
* **镜面反射 (Specular)**: 标准 Microfacet Cook-Torrance (GGX)。
    $$ L_{\text{spec}} = \text{GGX}(\mathbf{n}, \omega_i, \omega_o, \alpha, F_0) $$
  * *工程注记*：采用 Split-sum 近似或预积分环境贴图以加速训练。

### 2.2 复杂介质项 (Complex Media Components)

* **次表面散射 (SSS)**: 采用表面空间近似（Surface-space Approximation）。
    $$ L_{\text{sss}}(p) \approx w_{\text{sss}} \cdot (\mathbf{c}_{\text{sss}} * K_{\text{diffusion}}(p)) $$
  * 其中 $K$ 为扩散核（高斯或 Normalized Diffusion），模拟光在表皮下的模糊效应。
* **透射与吸收 (Transmission)**: 基于 Beer-Lambert 定律与薄膜近似。
    $$ L_{\text{trans}}(p) \approx w_{\text{trans}} \cdot e^{-\boldsymbol{\sigma}_t \cdot \tau(p)} \cdot L_{\text{env}}(\text{refract}(\omega_o, \mathbf{n}, \eta)) $$
  * $\tau(p)$: **几何厚度**。这是关键耦合点，需通过 SDF 射线追踪计算入射点与出射点的距离获得。
  * $\boldsymbol{\sigma}_t$: 吸收系数 (RGB)。

### 2.3 参数向量 (Parameter Vector)

网络 $F_\Theta(p)$ 输出以下连续场：

* **Base**: `albedo`(3), `roughness`(1), `metallic`(1)
* **Complex**: `sss_weight`(1), `sss_radius`(1), `trans_weight`(1), `absorption`(3), `ior`(1)
* **Control**: `gating_mask`(1, 可选)

## 3. 可辨识性分析与判别信号 (Identifiability & Diagnostics)

为了解决“不同物理过程产生相似外观”的歧义（例如：高粗糙度金属 vs. 灰色漫反射），我们需要利用**可观测的物理信号**进行判别或作为 Loss 的引导。

### 3.1 判别特征矩阵 (Discriminative Feature Matrix)

| 物理机制 | **视角方差 (View Variance)** | **空间模糊度 (Spatial Blur)** | **Beer-Lambert 残差** | **镜面锐度 (Specular Sharpness)** |
| :--- | :--- | :--- | :--- | :--- |
| **分层/表面 (Layered)** | **高** (强菲涅尔/高光) | 低 (纹理清晰) | 高 (不符指数衰减) | **锐利** (若界面光滑) |
| **胶体/SSS (Turbid)** | **低** (各向同性散射) | **高** (光晕/细节丢失) | 高 | 柔和/无 |
| **溶液/透射 (Absorption)** | 中/低 (仅路径变化) | 低 (透射清晰) | **低** (符合 $\exp(-\sigma \tau)$) | 取决于表面 |

### 3.2 判别策略

1. **Beer-Lambert 拟合测试**：利用 SDF 计算厚度 $\tau$，尝试拟合 $\ln(I) \propto -\tau$。若拟合残差极低，则强烈提示为“吸收/溶液”介质。
2. **模糊核测试**：对比像素邻域的颜色协方差。高协方差且低视角依赖性提示为 SSS。
3. **几何一致性**：透射物体的背景会发生折射位移（Snell's Law）。若能观测到背景扭曲，是透射存在的强证据。

## 4. 训练策略：分阶段与自适应 (Staged & Adaptive Training)

为防止模型陷入局部最优（如用“自发光”解释“高光”），必须采用严格的分阶段训练。

### 阶段 I：几何锚定 (Geometry Warm-up)

* **目标**：获得准确的 SDF 表面与厚度估计。
* **模型**：仅启用 `Diffuse` + 简单 `Specular`。关闭 SSS/Trans。
* **Loss**：RGB + Eikonal + Mask。

### 阶段 II：基础材质解耦 (Base Material Optimization)

* **目标**：分离 Albedo 与 Roughness/Metallic。
* **策略**：固定 SDF。利用多视角数据优化 View-dependent specular。
* **正则**：对 Roughness 施加平滑约束，防止高频噪声。

### 阶段 III：复杂效应解锁 (Complex Effects Unlocking)

* **目标**：引入 SSS 和 Transmission。
* **操作**：
    1. 开启 `sss_weight`, `trans_weight` 梯度。
    2. **输入 SDF 厚度 $\tau$**：强制 Transmission 依赖几何厚度。
    3. **稀疏正则**：对 `trans/sss_weight` 施加 L1 正则（Occam's Razor），默认物体是不透明的，除非光度误差迫使模型使用复杂项。

### 阶段 IV：自适应门控与微调 (Adaptive Gating & Fine-tuning)

* **自适应逻辑**：
  * 计算区域的 Photometric Residual。
  * 若 `Residual > Threshold` 且 `Discriminative Features` 暗示复杂介质 $\rightarrow$ 增加该区域 Gating 权重 $g(p)$，甚至分配额外的 Hash-grid 容量。
  * 否则，强制 $g(p) \to 0$，退化回简单模型。

## 5. 损失函数设计 (Loss Functions)

总目标函数：
$$ \mathcal{L} = \lambda_{\text{rgb}} \mathcal{L}_{\text{recon}} + \lambda_{\text{geo}} \mathcal{L}_{\text{eikonal}} + \lambda_{\text{phy}} \mathcal{L}_{\text{physics}} + \lambda_{\text{reg}} \mathcal{L}_{\text{sparse}} $$

1. **重建损失**: $\mathcal{L}_{\text{recon}} = ||C_{ren} - C_{gt}||_1$
2. **几何约束**: $\mathcal{L}_{\text{eikonal}} = (||\nabla \text{SDF}|| - 1)^2$
3. **物理先验 ($\mathcal{L}_{\text{physics}}$)**:
    * **能量守恒**: $\text{ReLU}(w_{\text{diff}} + w_{\text{spec}} + w_{\text{trans}} - 1.0)$
    * **Beer-Lambert 引导** (可选): 对被判别为吸收介质的区域，惩罚偏离指数衰减的预测。
4. **稀疏正则 ($\mathcal{L}_{\text{sparse}}$)**:
    * $\lambda ||w_{\text{sss}}||_1 + \lambda ||w_{\text{trans}}||_1$ (鼓励模型优先使用简单表面解释)
    * $\text{TV}(\text{Roughness})$ (鼓励材质参数的空间连续性)

## 6. 工程实施建议 (Implementation Checklist)

* [ ] **SDF 计算**：确保 Ray Marching 能够准确返回 `entry_point` 和 `exit_point` 以计算 `thickness`。
* [ ] **Split-Sum 积分**：镜面反射不要实时做蒙特卡洛积分，使用预计算的 LUT (Look-Up Table)。
* [ ] **初始化**：
  * Roughness: 0.5
  * Metallic: 0.0
  * Trans/SSS Weight: 0.0 (从不透明漫反射开始)
* [ ] **Debug 可视化**：
  * 必须输出分解图：`Render_Diffuse`, `Render_Spec`, `Render_Trans`, `Map_Thickness`, `Map_SSS_Weight`。
  * 若看到 `Map_Thickness` 出现噪声，需加强 Stage I 的几何训练。

## 7. 结论 (Conclusion)

本方案通过构建**物理完备的层状模型**，结合**SDF 几何信息（厚度）**与**分阶段判别训练**，旨在解决单一模型表达多种复杂介质的难题。
相比多头模型，它在**边界平滑性**、**物理可解释性**以及**对“有效漫反射”现象的表达**上应该具有显著优势。尽管存在可辨识性挑战，但通过引入特征判别与稀疏正则，该方案是当前的优选路线。

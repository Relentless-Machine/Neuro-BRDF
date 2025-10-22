# Integrated Directional Encoding (IDE)

目录

1. 概念与动机
2. 数学定义与符号
3. 详细推导
   - 3.1 vMF 归一化常数 C(κ)
   - 3.2 指数核的 Legendre 系数与 iℓ(κ)
   - 3.3 球谐加法定理：Pℓ(ω·ωr) 到 Yℓm(ω)Yℓm(ωr)
   - 3.4 期望的闭式：E[Yℓm]=Aℓ(κ)Yℓm(ωr)
   - 3.5 改进（球）贝塞尔函数与 Aℓ 的精确表达
   - 3.6 热核近似 Aℓ(κ)≈exp(-ℓ(ℓ+1)/(2κ)) 的来源与适用性
   - 3.7 实/复球谐与代码中 Re/Im 拼接的等价性
4. 工程实现
   - 4.1 Ref‑NeRF/NeRO 的“热核近似”路线（JAX/PyTorch 一致）
   - 4.2 精确 Bessel 比值查表路线（可选）
   - 4.3 反射方向、kappa_inv 与调用约定
   - 4.4 维度与带宽：deg_view、通道数与 72 维的来源
   - 4.5 一致性与单元测试（IDE→DE 退化）
   - 4.6 数值/性能注意事项（dtype、广播、缓存、梯度）
5. 数值稳定与超参数建议
6. 与 NeRO 的结合点（Stage I/Stage II）
7. 局限与扩展方向
8. 公式→代码对照清单（NeRO 与 Ref‑NeRF）
9. 参考文献

## 1. 概念与动机

在微表面 BRDF 框架中，镜面反射会在反射方向附近形成随粗糙度变化的方向 lobe。IDE（Integrated Directional Encoding）通过将方向函数基（球谐）在以反射方向为均值的 von Mises–Fisher（vMF）分布下做期望，实现对“粗糙度”的频谱自适应：粗糙度越大（lobe 越宽，即反射方向附近的高光峰越宽），高阶球谐带衰减越强。

- 物理动机：vMF(ωr, κ)，κ 为集中度（越大越集中）。
- 频谱自适应：Aℓ(κ) 随 κ 变化自动调节各阶带宽能量。
- 工程友好：有限带宽即可表征多尺度镜面；可微可训练。

代码总览：

- NeRO（PyTorch）：[utils/ref_utils.py → generate_ide_fn](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/utils/ref_utils.py#L1-L119)
- Ref‑NeRF（JAX）：[internal/ref_utils.py → generate_ide_fn](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L99-L157)

## 2. 数学定义与符号

- 视向量（出射）$\omega_o\in\mathbb{S}^2$；单位法线 $\mathbf{n}$。
- 反射方向（单位向量）：
  $$
  \omega_r = 2(\mathbf{n}\cdot\omega_o)\,\mathbf{n} - \omega_o.
  $$
- vMF 分布（球面 $\mathbb{S}^2$）：
  $$
  p(\omega\mid\omega_r,\kappa) = C(\kappa)\, e^{\kappa (\omega\cdot\omega_r)},\quad
  C(\kappa)=\frac{\kappa}{4\pi\sinh\kappa}.
  $$
- 实值球谐 $Y_\ell^m(\omega)$（$\ell\ge0,\ -\ell\le m\le\ell$），最大带宽 $L$，通道数 $(L+1)^2$。
- IDE 定义（对 vMF 的期望）：
  $$
  \mathrm{IDE}(\omega_r,\kappa) := \Big\{\mathbb{E}_{\omega\sim\mathrm{vMF}(\omega_r,\kappa)}[\,Y_\ell^m(\omega)\,]\Big\}_{0\le\ell\le L,\ -\ell\le m\le\ell}.
  $$

## 3. 详细推导

### 3.1 vMF 归一化常数 C(κ)

令 $\mu = \cos\theta = \omega\cdot\omega_r$，$d\omega = d\varphi\,d\mu$，其中 $\varphi\in[0,2\pi), \mu\in[-1,1]$。则
$$
\int_{\mathbb{S}^2} e^{\kappa(\omega\cdot\omega_r)}\,d\omega
= \int_0^{2\pi}\!\!\int_{-1}^{1} e^{\kappa\mu}\, d\mu\, d\varphi
= 2\pi \frac{e^{\kappa}-e^{-\kappa}}{\kappa}
= \frac{4\pi\sinh\kappa}{\kappa}.
$$
故 $C(\kappa)$ 满足 $C(\kappa)\cdot \dfrac{4\pi\sinh\kappa}{\kappa}=1$，即
$$
C(\kappa)=\dfrac{\kappa}{4\pi\sinh\kappa}.
$$

### 3.2 指数核的 Legendre 系数与 iℓ(κ)

设
$$
e^{\kappa\mu} = \sum_{\ell=0}^{\infty} a_\ell(\kappa)\, P_\ell(\mu).
$$
利用 Legendre 多项式的正交性
$$
\int_{-1}^1 P_\ell(\mu)P_{\ell'}(\mu)\,d\mu = \frac{2}{2\ell+1}\,\delta_{\ell\ell'},
$$
可得
$$
a_\ell(\kappa)=\frac{2\ell+1}{2}\int_{-1}^{1} e^{\kappa\mu}\,P_\ell(\mu)\,d\mu.
$$

该积分与“修正球贝塞尔函数”（modified spherical Bessel）满足经典恒等式：
$$
\int_{-1}^{1} e^{\kappa\mu}\,P_\ell(\mu)\,d\mu = 2\, i_\ell(\kappa),
$$
从而
$$
a_\ell(\kappa)=(2\ell+1) i_\ell(\kappa).
$$

注：

- $i_\ell(\kappa)$ 与改进贝塞尔 $I_\nu(\kappa)$ 的关系见 3.5 节；
- 该恒等式可由 Rodrigues 公式或母函数法（生成函数）推导。另一常用定义为递推/微分算子形式：
  $$
  i_\ell(\kappa) = (-\kappa)^\ell\left(\frac{1}{\kappa}\frac{d}{d\kappa}\right)^\ell\left(\frac{\sinh\kappa}{\kappa}\right).
  $$

据此得到指数核的 Legendre 展开：
$$
e^{\kappa\mu} = \sum_{\ell=0}^{\infty} (2\ell+1)\, i_\ell(\kappa)\, P_\ell(\mu).
$$

### 3.3 球谐加法定理：Pℓ(ω·ωr) 到 Yℓm(ω)Yℓm(ωr)

球谐加法定理（实或复形式）给出
$$
P_\ell(\omega\cdot\omega_r) = \frac{4\pi}{2\ell+1} \sum_{m=-\ell}^{\ell} Y_\ell^m(\omega)\,Y_\ell^m(\omega_r),
$$
将其代入上式得：
$$
e^{\kappa(\omega\cdot\omega_r)}
= 4\pi \sum_{\ell=0}^\infty i_\ell(\kappa)\sum_{m=-\ell}^{\ell} Y_\ell^m(\omega)\,Y_\ell^m(\omega_r).
$$

### 3.4 期望的闭式：E[Yℓm]=Aℓ(κ)Yℓm(ωr)

两侧乘 $Y_{\ell'}^{m'}(\omega)$ 并在球面上积分，利用球谐正交性
$$
\int_{\mathbb{S}^2} Y_{\ell'}^{m'}(\omega)\,Y_\ell^m(\omega)\, d\omega = \delta_{\ell\ell'}\delta_{mm'},
$$
得到
$$
\int_{\mathbb{S}^2} Y_{\ell'}^{m'}(\omega)\, e^{\kappa(\omega\cdot\omega_r)}\, d\omega
= 4\pi\, i_{\ell'}(\kappa)\, Y_{\ell'}^{m'}(\omega_r).
$$
乘以 vMF 归一化常数即为期望：
$$
\mathbb{E}_{\omega\sim\mathrm{vMF}}[Y_{\ell'}^{m'}(\omega)]
= \frac{\kappa\, i_{\ell'}(\kappa)}{\sinh\kappa}\, Y_{\ell'}^{m'}(\omega_r).
$$

### 3.5 改进（球）贝塞尔函数与 Aℓ 的精确表达

注意 $i_0(\kappa)=\dfrac{\sinh\kappa}{\kappa}$，因此
$$
\boxed{A_\ell(\kappa) = \frac{i_\ell(\kappa)}{i_0(\kappa)}.}
$$
$i_\ell(\kappa)$ 与改进贝塞尔函数 $I_\nu(\kappa)$ 的关系：
$$
i_\ell(\kappa) = \sqrt{\frac{\pi}{2\kappa}}\, I_{\ell+\frac12}(\kappa)
\quad\Rightarrow\quad
A_\ell(\kappa) = \frac{I_{\ell+\frac12}(\kappa)}{I_{\frac12}(\kappa)}.
$$

这提供了精确可数值稳定计算的表达（见第 4.2 节“查表路线”）。

### 3.6 热核近似 Aℓ(κ)≈exp(-ℓ(ℓ+1)/(2κ)) 的来源与适用性

球面热核在频域有形式 $e^{-t\ell(\ell+1)}$。当 vMF 非常集中（大 κ，窄 lobe）时，可近似将 vMF 与热核对应，取 $t \approx 1/(2\kappa)$，得到
$$
\boxed{A_\ell(\kappa)\ \approx\ \exp\!\left(-\frac{\ell(\ell+1)}{2\kappa}\right).}
$$

- 适用性：大 κ（光滑镜面/窄高光）区域误差很小；小 κ（粗糙）时高阶本就衰减强烈。
- 工程优势：避免 Bessel 计算，简单、稳定、可批量高效实现。
- Ref‑NeRF 与 NeRO 实际采用该近似；见 4.1 节代码对照。

### 3.7 实/复球谐与代码中 Re/Im 拼接的等价性

NeRO/Ref‑NeRF 用复数形式构造球谐，再拼接实部与虚部得到实值特征：
$$
\mathrm{IDE}_{\text{real}} = [\Re\,\mathrm{IDE}_{\mathbb{C}}\ ;\ \Im\,\mathrm{IDE}_{\mathbb{C}}].
$$
这与使用“实球谐基”的编码等价，只差一组固定的正交线性变换（相当于复到实基的基变换）。因此不影响表达能力与数值正交性。

## 4. 工程实现

### 4.1 Ref‑NeRF/NeRO 的“热核近似”路线（JAX/PyTorch 一致）

核心实现逻辑（两者一致）：

1) 预构造 ml_array（存放若干 (m, ℓ) 对，即球谐基函数的“次数-带阶”索引对），l_max，及系数矩阵 mat，用于从 z 的幂次和 (x+iy) 的幂次组合出复 SH；
2) 给定方向 xyz=[x,y,z]，构造
   - vmz = [z^0, z^1, ..., z^l_max]（Vandermonde 矩阵）
   - vmxy = concat[(x+iy)^m]（对 ml_array 中每个 m）
3) 得到复球谐“值向量”：
   - sph_harms = vmxy * (vmz @ mat)  // 逐通道乘
4) 按带 ℓ 计算 σℓ = ½ℓ(ℓ+1)，做指数衰减（kappa_inv = 1/κ）：
   - ide = sph_harms * exp(-σℓ · kappa_inv)
5) 返回实特征 concat([Re(ide), Im(ide)]).

对照代码（NeRO，PyTorch）：

- 生成 (m,ℓ) 与系数矩阵：
  - get_ml_array / sph_harm_coeff / assoc_legendre_coeff
  - 链接：[utils/ref_utils.py:1–53](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/utils/ref_utils.py#L1-L53)
- generate_ide_fn（deg_view≤5），生成闭包 integrated_dir_enc_fn：
  - 链接：[utils/ref_utils.py:54–119](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/utils/ref_utils.py#L54-L119)
  - 细化逐步映射（括号内为形状，· 为广播）：
    - 输入：xyz[...,3]（float32，单位方向），kappa_inv[...,1]（float32）
    - 切片：x=xyz[...,0:1]；y=xyz[...,1:2]；z=xyz[...,2:3]
    - vmz = concat_{i=0..l_max} z**i  → [..., l_max+1]（float32）
      - 代码（L100）：

        ```python
        vmz = torch.concat([z**i for i in range(mat.shape[0])], dim=-1)
        ```

    - vmxy = concat_{m∈ml_array[0,:]} (x + 1j*y)**m → [..., T]（complex64），T=ml_array.shape[1]
      - 代码（L103）：

        ```python
        vmxy = torch.concat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)
        ```

      - 注意：这里发生 dtype 提升为复数，PyTorch 会用 complex64。
    - proj = vmz @ mat → [..., T]（float32）
      - 代码（L106）：

        ```python
        sph_harms = vmxy * torch.matmul(vmz, mat)
        ```

      - mat 形状为 [(l_max+1), T]。
    - σ = 0.5 *l* (l+1) → [T]（float32），其中 l=ml_array[1,:]
      - 代码（L110）：

        ```python
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ```

    - ide = sph_harms * exp(-σ · kappa_inv) → [..., T]（complex64）
      - 代码（L111）：

        ```python
        ide = sph_harms * torch.exp(-sigma * kappa_inv)
        ```

      - 广播：σ[T] 与 kappa_inv[...,1] 广播到 [...,T]。
    - 输出：concat([Re(ide), Im(ide)], dim=-1) → [..., 2T]（float32）
      - 代码（L114）：

        ```python
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)
        ```

- deg_view 与 l 的选择：ℓ ∈ {2^0, 2^1, …, 2^{deg_view-1}}，每个 ℓ 取 m=0..ℓ，共（ℓ+1）个通道（复）。
- 退化为 DE（方向编码）：传入 kappa_inv=0（不衰减，即 Aℓ=1），即 Ref‑NeRF 的 generate_dir_enc_fn：
  - 链接：[internal/ref_utils.py:162–177](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L162-L177)

对照代码（Ref‑NeRF，JAX）：

- 同构实现（JAX + jnp + 自定义 math.matmul）：
  - 链接：[internal/ref_utils.py:99–157](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L99-L157)

### 4.2 精确 Bessel 比值查表路线（可选）

若需严格实现 $A_\ell(\kappa)=I_{\ell+\frac12}(\kappa)/I_{\frac12}(\kappa)$，建议：

- 离线预计算 logIv 表（log Iν）以避免溢出；
- 训练/推理时在 GPU 侧按 κ 线性插值；
- 将 Aℓ 扩展到每个（ℓ,m）通道，再与球谐基逐通道相乘（与 4.1 的热核路线仅“系数来源”不同）。

参考预计算与插值代码：

```python
import numpy as np
from scipy.special import logiv

def precompute_A_table(L=4, kappa_min=1e-3, kappa_max=1e2, N=160):
  grid = np.logspace(np.log10(kappa_min), np.log10(kappa_max), N)
  A = np.zeros((N, L+1), dtype=np.float64)
  for i, k in enumerate(grid):
    logI_half = logiv(0.5, k)
    for l in range(L+1):
      A[i, l] = np.exp(logiv(l + 0.5, k) - logI_half)
  return grid, A
```

GPU 侧插值与按带扩展：

```python
import torch

def interp_A_batched(kappa, kappa_grid, A_table):
  k = torch.clamp(kappa, kappa_grid[0].item(), kappa_grid[-1].item())
  idx = torch.searchsorted(kappa_grid, k)
  idx = torch.clamp(idx, 1, len(kappa_grid)-1)
  lo, hi = idx - 1, idx
  t = (k - kappa_grid[lo]) / (kappa_grid[hi] - kappa_grid[lo] + 1e-12)
  return (1 - t.unsqueeze(-1)) * A_table[lo] + t.unsqueeze(-1) * A_table[hi]

def expand_A_to_sh_channels(A_band, L):
  parts = [A_band[:, l:l+1].repeat(1, 2*l+1) for l in range(L+1)]
  return torch.cat(parts, dim=1)
```

对比：

- 查表路线精确但略慢，需要存储 [G, L+1] 的 A 表；
- 热核路线无需表，更快，Ref‑NeRF/NeRO 默认采用。

### 4.3 反射方向、kappa_inv 与调用约定

- 反射方向 $\omega_r=2(n\cdot v)n-v$：
  - Ref‑NeRF：`reflect(viewdirs, normals)`（JAX）
    - [internal/ref_utils.py:19–36](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L19-L36)
  - NeRO：直接在 shader 中计算（PyTorch）
    - [network/field.py: forward（reflective 计算）](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L848-L860)

- kappa_inv（κ 的倒数）：
  - 在 NeRO 中通常直接使用网络预测的 roughness 作为 kappa_inv（单调映射即可，符合“粗糙→更衰减”的直觉）；
  - 也可使用 kappa_inv=roughness^2 或其它单调函数，保持趋势一致。

- 调用示意（直射光编码）：

  ```python
  reflective = normalize(2 * dot(n, v) * n - v)
  kappa_inv = roughness.clamp_min(eps)
  ide_vec = sph_enc(reflective, kappa_inv)   # 72D when deg_view=5
  ```

### 4.4 维度与带宽：deg_view、通道数与 72 维的来源

- 采用 ℓ ∈ {1,2,4,8,16}（deg_view=5），每个 ℓ 取 m=0..ℓ，共（ℓ+1）个“复通道”，拆 Re/Im 后通道翻倍。
- 通道总数：
  $$
  T=\sum_{\ell\in\{1,2,4,8,16\}} (\ell+1)=2+3+5+9+17=36,\quad
  \text{输出维}=2T=72.
  $$
- NeRO 在 outer/inner light 分支中用到 72 维 IDE 向量：
  - [network/field.py: sph_enc/outer_light/inner_light 初始化](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L513-L526)

### 4.5 一致性与单元测试（IDE→DE 退化）

- 性质：当 kappa_inv=0（即 κ→∞），IDE 应退化为“无衰减的方向编码”（DE）。
- Ref‑NeRF 直接提供 `generate_dir_enc_fn`，其内部调用 `generate_ide_fn` 并把 kappa_inv 置零：
  - [internal/ref_utils.py:162–177](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L162-L177)
- 建议测试：
  - 对若干随机方向，比较 `ide(xyz, kappa_inv=0)` 与 `de(xyz)` 的输出，容差在 1e-5～1e-4。
  - 对随机 κ，验证 A0≈1、Aℓ>0 单调随 κ 增大而增大（或随 kappa_inv 减小而增大）。

### 4.6 数值/性能注意事项（dtype、广播、缓存、梯度）

- dtype：
  - vmxy 使用了 `(x + 1j * y)**m` → complex64；与后续实值 proj 相乘得到 complex64；
  - 最终 concat 实部/虚部回到 float32。
- 广播：
  - σ 为 [T]，kappa_inv 为 [...,1]，通过广播到 [...,T]；
  - 请确保 kappa_inv 的维度保留最后一维=1。
- 归一化：
  - 输入 xyz 应为单位向量；如有数值偏差，可先 normalize。
- 预计算缓存：
  - ml_array / mat 应注册为 buffer（NeRO 中已作为 GPU tensor 缓存）；
  - 避免重复 host→device 拷贝。
- 梯度：
  - 指数衰减对 kappa_inv 可导，利于学习粗糙度；
  - 若使用查表路线，插值对 κ 可导（piecewise 线性）；log 域计算更稳。

## 5. 数值稳定与超参数建议

- deg_view：建议 ≤ 5（Ref‑NeRF/NeRO 中明确限制，数值稳定/表达-成本达到折中）。
- kappa_inv/roughness：
  - 使用 softplus/clamp 保证正值；
  - 若采用查表路线，κ 建议对数均匀网格 [1e-3, 1e2]，100～200 点。
- 边界：
  - kappa_inv→0（极小粗糙）时接近 DE；
  - kappa_inv→大（极粗糙）时 A0≈1、其余带→0。
- 法线/反射方向稳定：
  - 建议 Eikonal 正则、充分采样，避免法线噪声经 IDE 放大（NeRO 已含）；
  - 参考：[network/loss.py: Eikonal 权重退火](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/loss.py#L28-L43)。

## 6. 与 NeRO 的结合点（Stage I/Stage II）

- Stage I（AppShadingNetwork；快速、近似）：
  - 方向编码：`self.sph_enc = generate_ide_fn(5)`（72 维）
  - 直射光 outer_light：以 IDE(reflective, roughness)（可选拼接“球面方向”IDE）
  - 间接光 inner_light：concat(pos_enc(points), IDE(reflective, roughness))
  - BRDF 组合：Fresnel/G/GGX + LUT
  - 入口：
    - [network/field.py: AppShadingNetwork.__init__（sph_enc/outer/inner 初始化）](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L513-L526)
    - [network/field.py: predict_specular_lights/predict_diffuse_lights/forward](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L820-L1046)

- Stage II（MCShadingNetwork；采样、精化）：
  - 引入方向采样、可见性估计与重要性权重；
  - 多处用 `self.sph_enc(., 0)`（退化为 DE）表达方向先验；
  - 入口：
    - [network/field.py: MCShadingNetwork.__init__（sph_enc 初始化）](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L714-L819)

- IPE（integrated positional encoding）的人像光编码：
  - NeRO（PyTorch）：`expected_sin/IPE` 实现
    - [network/field.py:369–378](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L369-L378)
  - Ref‑NeRF（JAX）：`integrated_pos_enc`
    - [internal/coord.py:102–137](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/coord.py#L102-L137)

## 7. 局限与扩展方向

- 各向异性/多峰：当前 vMF 假设绕反射轴旋转对称，难表各向异性或多峰高光；可扩展各向异性 vMF 或混合多个分布；
- 近场/遮挡：IDE 更近“远场预滤”；近场需配合显式采样与遮挡（NeRO Stage II 已做）；
- 极窄高光：需更高带宽或依赖采样积分；可探索自适应带宽 L(κ)/学习型带权。

## 8. 公式→代码对照清单（NeRO 与 Ref‑NeRF）

- 反射方向：$\omega_r = 2(n\cdot v)n - v$
  - Ref‑NeRF：`reflect(viewdirs, normals)`（JAX） → [internal/ref_utils.py:19–36](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L19-L36)
  - NeRO：`reflective = sum(v*n)*2*n - v`（PyTorch） → [network/field.py:848–860](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L848-L860)

- IDE 生成（deg_view≤5；vmz/vmxy/mat；exp 衰减）
  - Ref‑NeRF（JAX）：[internal/ref_utils.py:99–157](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L99-L157)
  - NeRO（PyTorch）：[utils/ref_utils.py:54–119](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/utils/ref_utils.py#L54-L119)

- 复球谐系数/Legendre 系数
  - Ref‑NeRF：assoc_legendre_coeff/sph_harm_coeff → [internal/ref_utils.py:54–162](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L54-L162)
  - NeRO：同名函数 → [utils/ref_utils.py:1–53](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/utils/ref_utils.py#L1-L53)

- DE 退化（kappa_inv=0）
  - Ref‑NeRF：`generate_dir_enc_fn` → [internal/ref_utils.py:162–177](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/ref_utils.py#L162-L177)

- IPE（集成位置编码）
  - Ref‑NeRF：`integrated_pos_enc` → [internal/coord.py:102–137](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/coord.py#L102-L137)
  - NeRO：`expected_sin/IPE` → [network/field.py:369–378](https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L369-L378)

## 9. 参考文献

1. Verbin, D., Chou, C., Li, Z., Kholgade, N., & Mildenhall, B. Ref‑NeRF: Structured View‑Dependent Appearance for Neural Radiance Fields. CVPR 2022.（IDE 提出与补充证明）
2. Ref‑NeRF Supplementary Material — Integrated Directional Encoding Proofs（IDE 的完整推导与数值细节）
3. Mardia, K. V., & Jupp, P. E. Directional Statistics. Wiley, 2000.（方向统计与 vMF）
4. Olver, F. W. J., et al. NIST Handbook of Mathematical Functions.（改进/球贝塞尔函数与数值）
5. Liu, Y., et al. NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images. SIGGRAPH 2023.（IDE 在 NeRO 管线的使用）

## 10. 更新日志

- 数学部分：补充从指数核到球谐期望的完整推导，含 Legendre 系数积分、球谐加法定理、vMF 归一化常数、与改进（球）贝塞尔函数的关系；补充“热核近似”的严格来源与适用域。
- 代码部分：逐行映射 NeRO/Ref‑NeRF 的 IDE 实现，解释每个张量的形状/类型、广播机制与数值注意事项；补充维度计算、退化一致性检查（IDE→DE）、以及可选的“精确 Bessel 查表路线”。
- 备注：集成位置编码（Integrated Positional Encoding, IPE）是在输入位置存在不确定性（如射线段/圆锥体上一个高斯分布）时，对标准位置编码的正弦余弦项做期望，得到带“频率自适应衰减”的特征，用于抗混叠与更稳的训练。

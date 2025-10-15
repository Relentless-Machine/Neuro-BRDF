# Integrated Directional Encoding (IDE)

## 目录

1. 概念与动机
2. 数学定义与符号
3. 形态推导（含 Legendre 与贝塞尔关系）
4. 工程实现（Python/PyTorch，含伪码）
5. 数值稳定与超参数建议
6. 与 NeRO 的结合点
7. 局限与扩展方向
8. 参考文献

## 1. 概念与动机

在微表面 BRDF 框架中，镜面反射在反射方向附近形成随粗糙度变化的方向 lobe。IDE（Integrated Directional Encoding）用“以反射方向为中心、由粗糙度控制的旋转对称方向分布”在球面基（实球谐，SH）上的期望作为编码向量，使方向网络能够根据粗糙度自适应地低通/高通方向频谱，兼具物理一致性、可微性与参数高效性。

要点与优势：
- 物理动机明确：以反射方向为均值的 vMF（von Mises–Fisher）族；
- 频谱自适应：粗糙度越大，高阶 SH 带抑制越强；
- 工程简单：仅需少量带宽 L 的 SH 通道即可表达多尺度镜面；
- 可微可训练：与 NeRF/NeRO 的方向 MLP 无缝拼接。

## 2. 数学定义与符号

- 几何与方向：
	- 视向量（出射）$\omega_o\in\mathbb{S}^2$；单位法线 $\mathbf{n}$；
	- 反射方向 $\omega_r = 2(\mathbf{n}\cdot\omega_o)\,\mathbf{n} - \omega_o$（单位）。
- 粗糙度与集中度：粗糙度 $\rho>0$，集中度 $\kappa>0$，常取 $\kappa = 1/\rho$ 或任一单调递减映射 $g(\rho)$。
- vMF（球面 $\mathbb{S}^2$）密度：
	$$
	p(\omega\mid\omega_r,\kappa) = C(\kappa)\, e^{\kappa (\omega\cdot\omega_r)},\quad
	C(\kappa)=\frac{\kappa}{4\pi\sinh\kappa}.
	$$
- 实值球谐 $Y_\ell^m(\omega)$（$\ell\ge0,\ m\in[-\ell,\ell]$），最大带宽 $L$；通道数 $(L+1)^2$。

IDE 定义为在 vMF 下对球谐基的期望：
$$
\mathrm{IDE}(\omega_r,\kappa) := \Big\{\,\mathbb{E}_{\omega\sim\mathrm{vMF}(\omega_r,\kappa)}[\,Y_\ell^m(\omega)\,]\ :\ 0\le\ell\le L,\ -\ell\le m\le\ell\Big\}.
$$

## 3. 形态推导（含 Legendre 与贝塞尔关系）

目标：证明
$$
\mathbb{E}_{\omega\sim\mathrm{vMF}}[Y_\ell^m(\omega)] = A_\ell(\kappa)\, Y_\ell^m(\omega_r),
$$
并给出 $A_\ell(\kappa)$ 的解析形式与数值可行表达。

步骤（凝练自 Ref‑NeRF 补充推导）：

1) 指数核的 Legendre/SH 展开（令 $\mu=\omega\cdot\omega_r$）：

  $$
  e^{\kappa\mu} = \sum_{\ell=0}^\infty (2\ell+1)\, i_\ell(\kappa)\, P_\ell(\mu)
  = 4\pi\sum_{\ell=0}^\infty i_\ell(\kappa) \sum_{m=-\ell}^{\ell} Y_\ell^m(\omega)\,Y_\ell^m(\omega_r).
  $$

2) 乘以 $Y_{\ell'}^{m'}(\omega)$ 对 $\omega$ 积分并用正交性：

  $$
  \int_{\mathbb{S}^2} Y_{\ell'}^{m'}(\omega)\, e^{\kappa(\omega\cdot\omega_r)}\, d\omega
  = 4\pi\, i_{\ell'}(\kappa)\, Y_{\ell'}^{m'}(\omega_r).
  $$

3) 乘以 vMF 归一化常数 $C(\kappa)$ 得期望：

  $$
  \mathbb{E}[Y_{\ell}^{m}(\omega)] = \frac{\kappa\, i_\ell(\kappa)}{\sinh\kappa}\, Y_\ell^{m}(\omega_r).
  $$

4) 由于 $i_0(\kappa)=\sinh\kappa/\kappa$，得到比例式：

  $$
  \boxed{A_\ell(\kappa) = \frac{i_\ell(\kappa)}{i_0(\kappa)}}.
  $$

与改进贝塞尔的关系：
$$
i_\ell(\kappa)=\sqrt{\frac{\pi}{2\kappa}}\, I_{\ell+\tfrac12}(\kappa)
\quad\Rightarrow\quad A_\ell(\kappa) = \dfrac{I_{\ell+\tfrac12}(\kappa)}{I_{\tfrac12}(\kappa)}.
$$

边界行为与近似：

- 大 $\kappa$（窄 lobe/光滑）：$A_\ell(\kappa) \approx \exp\!\left(-\tfrac{\ell(\ell+1)}{2\kappa}\right)$；
- 小 $\kappa$：$A_0\to 1$，$A_{\ell>0}\to 0$。

备注：该近似在工程上高效稳定，通常与查表法混合使用以兼顾精度与速度。

### 3.1 详细推导（逐步）

为满足“逐步推导”的要求，下面从归一化常数、Legendre 系数、加法定理与正交性依次给出完整推导。

步骤 1：vMF 归一化常数 $C(\kappa)$

  在球坐标中令 $\mu=\cos\theta=\omega\cdot\omega_r$，有 $d\omega = d\varphi\, d\mu$ 且 $\varphi\in[0,2\pi)$，$\mu\in[-1,1]$。于是
  $$
  \int_{\mathbb{S}^2} e^{\kappa(\omega\cdot\omega_r)}\,d\omega
  = 2\pi \int_{-1}^{1} e^{\kappa\mu}\, d\mu
  = 2\pi\, \frac{e^{\kappa}-e^{-\kappa}}{\kappa}
  = \frac{4\pi\sinh\kappa}{\kappa}.
  $$
  因此 $C(\kappa)$ 必须满足 $C(\kappa)\cdot \dfrac{4\pi\sinh\kappa}{\kappa}=1$，即
  $$
  \boxed{\ C(\kappa)=\dfrac{\kappa}{4\pi\sinh\kappa}\ }.
  $$

步骤 2：指数核的 Legendre 系数（关于 $\mu\in[-1,1]$）

  设
  $$
  e^{\kappa\mu} = \sum_{\ell=0}^{\infty} a_\ell(\kappa)\, P_\ell(\mu).
  $$
  由 $\int_{-1}^1 P_\ell(\mu)P_{\ell'}(\mu)\,d\mu = \dfrac{2}{2\ell+1}\,\delta_{\ell\ell'}$ 得
  $$
  a_\ell(\kappa) = \frac{2\ell+1}{2}\int_{-1}^{1} e^{\kappa\mu}\, P_\ell(\mu)\, d\mu.
  $$
  这一积分的值与“修正球贝塞尔函数”满足恒等
  $$
  a_\ell(\kappa) = (2\ell+1)\, i_\ell(\kappa),
  $$
  因而得到常用展开式
  $$
  e^{\kappa\mu} = \sum_{\ell=0}^{\infty} (2\ell+1)\, i_\ell(\kappa)\, P_\ell(\mu).
  $$
  注：$i_\ell(\kappa)$ 可由 $I_{\ell+\frac12}(\kappa)$ 定义，见步骤 5）。

步骤 3：由加法定理将 $P_\ell(\omega\cdot\omega_r)$ 写成球谐乘积

  $$
  P_\ell(\omega\cdot\omega_r) = \frac{4\pi}{2\ell+1}\sum_{m=-\ell}^{\ell} Y_\ell^m(\omega)\,Y_\ell^m(\omega_r).
  $$
  将其代入指数核展开，得到
  $$
  e^{\kappa(\omega\cdot\omega_r)}
  = 4\pi \sum_{\ell=0}^{\infty} i_\ell(\kappa) \sum_{m=-\ell}^{\ell} Y_\ell^m(\omega)\,Y_\ell^m(\omega_r).
  $$

步骤 4：乘以 $Y_{\ell'}^{m'}(\omega)$ 并在球面上积分（正交性）

  $$
  \int_{\mathbb{S}^2} Y_{\ell'}^{m'}(\omega)\, e^{\kappa(\omega\cdot\omega_r)}\, d\omega
  = 4\pi\, i_{\ell'}(\kappa)\, Y_{\ell'}^{m'}(\omega_r),
  $$
  其中 $\int Y_{\ell'}^{m'}(\omega)\,Y_\ell^m(\omega)\, d\omega = \delta_{\ell\ell'}\delta_{mm'}$（实球谐）。

步骤 5：引入 vMF 归一化常数得到期望

  $$
  \mathbb{E}_{\omega\sim\mathrm{vMF}}[Y_{\ell'}^{m'}(\omega)]
  = C(\kappa)\cdot 4\pi\, i_{\ell'}(\kappa)\, Y_{\ell'}^{m'}(\omega_r)
  = \frac{\kappa\, i_{\ell'}(\kappa)}{\sinh\kappa}\, Y_{\ell'}^{m'}(\omega_r).
  $$
  注意到 $i_0(\kappa) = \dfrac{\sinh\kappa}{\kappa}$，因此
  $$
  \boxed{\ A_\ell(\kappa) = \dfrac{i_\ell(\kappa)}{i_0(\kappa)}\ }.
  $$

步骤 6：与改进贝塞尔 $I_\nu$ 的关系与边界检验

  $$
  i_\ell(\kappa) = \sqrt{\frac{\pi}{2\kappa}}\, I_{\ell+\tfrac12}(\kappa),\quad
  I_{\tfrac12}(\kappa) = \sqrt{\frac{2}{\pi\kappa}}\, \sinh\kappa.
  $$
  因而 $i_0(\kappa)=\sqrt{\tfrac{\pi}{2\kappa}}\, I_{\tfrac12}(\kappa)=\dfrac{\sinh\kappa}{\kappa}$，与步骤 5) 一致。
  进一步可得
  $$
  A_\ell(\kappa) = \frac{I_{\ell+\tfrac12}(\kappa)}{I_{\tfrac12}(\kappa)}.
  $$

步骤 7：小/大 $\kappa$ 的极限与近似

  小 $\kappa$：利用 $I_\nu(\kappa) \sim \dfrac{1}{\Gamma(\nu+1)}\left(\tfrac{\kappa}{2}\right)^\nu$，得
  $$A_0\to 1,\quad A_{\ell>0}\to 0,$$
  与“均匀分布”直觉一致。

  大 $\kappa$：利用球面热核/局部高斯近似，可得常用工程近似
  $$
  \boxed{\ A_\ell(\kappa)\ \approx\ \exp\!\left(-\tfrac{\ell(\ell+1)}{2\kappa}\right) }.
  $$
  该式可视为对目标方向频谱的二次型衰减，在较大 $\kappa$ 区间内误差很小。

## 4. 工程实现（Python/PyTorch，含伪码）

输入/输出与约定（Batch 形状）：

- 输入：
  - 反射方向 ref_dirs: [B, 3]（单位向量）；
  - 粗糙度 roughness: [B] 或 [B, 1]（正数，建议 softplus 后 clamp）。
- 超参：带宽 L；kappa 查表网格与 A 表；
- 输出：IDE 向量 [B, (L+1)^2]（已按带缩放）。
- 错误模式：粗糙度过小导致 $\kappa$ 过大时数值不稳；通过 clamp 与查表边界插值规避。

步骤（每采样点）：

1) 由 $\mathbf{n}$ 与 $\omega_o$ 计算 $\omega_r$；
2) 由 $\rho$ 得 $\kappa = g(\rho)$（常用 $1/\rho$ 并 clamp）；
3) 评估 $Y_\ell^m(\omega_r)$；
4) 查表插值 $A_\ell(\kappa)$；
5) 按带扩展并逐通道相乘，得到 IDE 向量。

离线预计算（NumPy + SciPy）：

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

GPU 侧按批插值与按带扩展（PyTorch）：

```python
import torch

def interp_A_batched(kappa, kappa_grid, A_table):
  # kappa: [B], kappa_grid: [G] asc, A_table: [G, L+1]
  k = torch.clamp(kappa, kappa_grid[0].item(), kappa_grid[-1].item())
  idx = torch.searchsorted(kappa_grid, k)
  idx = torch.clamp(idx, 1, len(kappa_grid)-1)
  lo, hi = idx - 1, idx
  t = (k - kappa_grid[lo]) / (kappa_grid[hi] - kappa_grid[lo] + 1e-12)
  return (1 - t.unsqueeze(-1)) * A_table[lo] + t.unsqueeze(-1) * A_table[hi]

def expand_A_to_sh_channels(A_band, L):
  # A_band: [B, L+1] -> [B, (L+1)**2]
  parts = [A_band[:, l:l+1].repeat(1, 2*l+1) for l in range(L+1)]
  return torch.cat(parts, dim=1)

class IDEModule(torch.nn.Module):
  def __init__(self, L, kappa_grid, A_table_tensor):
    super().__init__()
    self.L = L
    self.register_buffer('kappa_grid', kappa_grid)    # [G]
    self.register_buffer('A_table', A_table_tensor)   # [G, L+1]

  def forward(self, ref_dirs, roughness, eval_real_sh_batch):
    B = ref_dirs.shape[0]
    rho = torch.clamp(roughness.view(B), 1e-3, 1.0)
    kappa = 1.0 / rho
    A_band = interp_A_batched(kappa, self.kappa_grid, self.A_table)  # [B, L+1]
    Y = eval_real_sh_batch(ref_dirs, self.L)                         # [B, (L+1)**2]
    A_exp = expand_A_to_sh_channels(A_band, self.L)                  # [B, (L+1)**2]
    return A_exp * Y
```

注：若需极致速度且容许微小误差，可直接用 $\exp\!(-\ell(\ell+1)/(2\kappa))$ 近似生成 $A_\ell$。

## 5. 数值稳定与超参数建议

- 粗糙度与集中度：
  - 建议 $\rho\in[10^{-3}, 1.0]$，以 softplus 后 clamp；$\kappa=1/\rho$ 并在表格范围内裁剪。
- A 表与网格：
  - $\kappa$ 取对数均匀网格 $[10^{-3}, 10^2]$，100–200 点效果良好；
  - A 表形状 [G, L+1]，训练前离线生成并存为 tensor 加载至 GPU。
- 近似与边界：
  - $\kappa<\kappa_{\min}$：返回 $A_0\approx1, A_{\ell>0}\approx0$；
  - $\kappa>\kappa_{\max}$：采用指数近似或对数域差分稳定求值；
  - 对高光极窄材料可适当增大 L（3→4/5），但计算与过拟合风险上升。
- 实球谐评估：采用稳定递推或成熟实现，避免高阶数值不稳；
- 法线稳定：对 SDF 法线加入 Eikonal 正则与充分采样，减小 IDE 噪声放大效应。

## 6. 与 NeRO 的结合点

在 NeRO 的 Stage I 中，IDE 与 Split‑sum 近似结合，用于高效表达视角依赖镜面；Stage II 使用蒙特卡洛采样进一步精化光照积分并校正近似误差。

接入流程：

1) 空间 MLP 输出：粗糙度 $\rho$、漫反射 albedo、瓶颈向量 $b(p)$ 等；
2) 由 $\mathbf{n}$ 与 $\omega_o$ 计算 $\omega_r$；
3) 通过 IDE 得到方向编码并与 $(\mathbf{n}\cdot\omega_o)$、$b(p)$ 拼接输入方向 MLP；
4) 输出镜面分量，与漫反射项按 BRDF 组合；
5) 可在 Stage II 用重要性采样/可见性估计替代近似，提升真实性。

示例（伪码）：

```python
# inputs: p, n, omega_o, raw_rho, bottleneck b, ide_module
ref = normalize(2 * (n * omega_o).sum(-1, keepdim=True) * n - omega_o)
rho = softplus(raw_rho).clamp_min(1e-3)
ide_vec = ide_module(ref, rho.view(-1), eval_real_sh_batch)
dir_in = concat(ide_vec, (n * omega_o).sum(-1, keepdim=True), b)
spec_rgb = directional_mlp(dir_in)
color = brdf_combine(diffuse_rgb, spec_rgb, metalness, weights)
```

## 7. 局限与扩展方向

- 各向异性反射与多峰高光：vMF 假设旋转对称；可扩展到各向异性 vMF 或混合多个方向分布以拟合织物等材料；
- 近场光源与强可见性：IDE 更接近远场预滤；近场需结合显式采样与遮挡估计（SIE/捕获模块）；
- 极低粗糙度：需更高带宽 L 才能表示窄高光，训练/推理成本与噪声敏感性上升；
- 自适应带宽：可考虑 L=L(\kappa) 或学习型带权，平衡精度与效率。

## 8. 参考文献

1. Verbin, D., Chou, C., Li, Z., Kholgade, N., & Mildenhall, B. Ref‑NeRF: Structured View‑Dependent Appearance for Neural Radiance Fields. CVPR 2022.（含 IDE 提出与补充证明）
2. Ref‑NeRF Supplementary Material — Integrated Directional Encoding Proofs（IDE 的完整推导与数值细节）
3. Mardia, K. V., & Jupp, P. E. Directional Statistics. Wiley, 2000.（方向统计与 vMF）
4. Olver, F. W. J., et al. NIST Handbook of Mathematical Functions.（改进/球贝塞尔函数性质与数值）
5. Liu, Y., et al. NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images. SIGGRAPH 2023.（IDE 在 NeRO 管线的使用）

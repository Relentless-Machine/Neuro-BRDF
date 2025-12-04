# 综述

## 背景与动机

近年来，基于神经渲染的三维重建方法（如 NeRF）在无监督场景建模与新视角合成方面取得显著进展[^1]。例如，NeRF 使用多层感知机将三维位置和方向映射为颜色与体积密度，再通过体积渲染合成像素[^1]。更近的工作 Neuralangelo 结合多分辨率哈希网格与神经表面渲染，实现了高保真的几何重建[^2] [^3]；NeRO 等方法则针对反射材质重建几何和 BRDF[^4]。然而，这些方法通常假定目标为单层均质表面，其外观仅由单一微表面模型或体积辐射场决定，难以自然刻画具有复杂多层介质结构的物体（如涂层、皮肤、染料层等）。

同时，材料科学和计算机图形学领域早已有大量关于分层介质光学的研究，如 Kubelka–Munk 模型、扩散近似与多重散射模型，以及皮肤渲染中的多层 Dipole 模型等[^5] [^6]。为实现兼具实用性和通用性的复杂表面建模，需要将这些物理理论与神经渲染方法有机结合。

## 问题识别与理论基础

多层介质模型的物理理论基础：多层结构的渲染可以依据光学与散射理论来构建。经典地，Kubelka–Munk 理论针对层状遮光体给出了吸收与散射的传播模型（用于油漆、纸张等多层材料），但该模型通常没有解析闭式解且假设均质各向同性[^5]。Hanrahan 和 Krueger 早在 1993 年提出了分层表面反射模型，将总反射分为表面镜面分量和次表面散射分量，并通过菲涅尔系数对两部分进行调制：$f_r = R\cdot f_{r,s} + (1 - R)\cdot f_{r,v}$[^7]。该表达式物理上明确了：当菲涅尔反射率 $R$ 较高时，表面镜面反射占优，进入介质的光能减少，反之则体散射增强。

在此基础上，现代方法可将任意多层介质视为多个界面和层体的串联，综合应用 Fresnel 反射、微表面模型、体散射理论等来计算综合 BRDF/BTDF。Wenzel Jakob 等人在 SIGGRAPH 2014 提出了一个基于输运理论的分层 BSDF 求解框架，该方法使用各向同性或各向异性散射层的输运方程和边界条件，将层内和层间的多次散射完整考虑在内，从而能够高效地计算任意层结构的 BRDF[^8] [^9]。他们进一步通过与真实多层材料测量数据对比验证了模型的准确性[^10]。此外，在皮肤渲染领域，Donner 和 Jensen 等人引入了多极扩散近似（Multipole model），用于描述薄层皮肤的次表面散射，将多层散射转换到频域简化卷积求解[^6]；D’Eon 等人则用高斯函数基底逼近扩散响应以提高准确性。这些文献都表明：分层模型从物理上是合理的，可以精细控制每层的折射率、吸收和散射系数，以仿真现实中多材质结构的光学行为。

理论局限性：尽管分层模型在理论上更贴近物理，但它们也存在局限性。Kubelka–Munk 等早期模型缺乏封闭解，往往不易直接应用[^5]；Neumann 等人最初的分层 BRDF 模型虽然考虑了光滑透明层和表面层的组合，但无法精确模拟层间吸收和内部多次反射，只使用简化的 Cook–Torrance 模型描述镜面分量[^11]。扩散近似（Dipole/Multipole）在厚层和弱各向异性情况下有效，但当材料散射高度各向异性或具有明显层间异质结构时，其假设被打破，可能出现过度模糊或伪影[^12]。此外，目前大部分物理模型均假定线性光学响应，对于非线性光学效应（如强光下材料折射率变化、光致发光、二次谐波等）尚无通用处理方法。因此，在通用建模策略中需要明确：分层模型可提供物理解释和物理一致性，但应注意其基于近似的前提条件和局限，必要时结合实验数据或更高阶理论加以校正。

## 层状模型中多阶段耦合分析

在一个分层系统中，不同光学过程（表面镜面反射、散射吸收、次表面散射等）本质上是耦合的。传统渲染中常分别处理镜面反射和漫反射/散射，这种分离往往采用诸如镜面微表面反射模型 + 半球漫反射或者简单加权的方式，但未必严格遵循能量守恒或菲涅尔定律。实际上，层状模型通过每个界面的菲涅尔透射/反射来联系表面与体积贡献。例如 Hanrahan–Krueger 模型明确：表面反射分量 $R f_{r,s}$ 与由菲涅尔透射 $(1 - R)$ 调制的次表面散射 $f_{r,v}$ 相加，保证了能量守恒且能解释高镜面反射时体散射减少的现象[^7]。在实现上，可以采用矩阵或谱方法将各层响应级联：Wenzel 的方法通过将每层的散射矩阵展开到球谐基底，实现了层间散射的耦合运算[^8] [^9]，从而获得统一的整体 BRDF。

### 现有模型与算法

一些研究尝试统一处理多阶段交互。SpongeCake 模型将每层视作具有微片散射相位函数（SGGX）的体积散射介质，通过解析推导获得单次散射的精确解，并引入额外神经网络模块拟合多次散射分量[^13]。该模型省略了层间界面的反射/折射，仅用层间导出光传播。这种做法表明，利用物理公式解决单次散射并用神经网络补偿复杂多散射是可行的。另一方向，Neural SSS 等工作使用神经网络直接拟合完整物体的 BSSRDF（8 维光传输）[^14]，在仿真中统一处理不同入射与出射位置间的散射，虽然更难以解释各阶段物理意义，但可捕获异质散射细节。此外，也有工作在神经 BRDF（NBRDF）建模中，将漫反射和镜面分量作为不同网络输出并采用加性分解，以尝试保持物理一致性[^15] [^16]。总的来看，目前虽有多种方法将镜面与散射统一建模（通过显式层合或神经网络分支），但尚无一个端到端的物理严格化方法能同时处理所有阶段及其相互影响。

## 当前神经渲染框架分析

主流神经渲染框架如 NeRO、Neuralangelo、Gaussian Splatting 等在复原几何和表面颜色上表现优异，但它们在物理建模方面存在明显缺失。NeRO 等方法利用分步策略来重建反射物体：先用分割光照近似重建几何，再固定几何拟合环境光和 BRDF[^4]。尽管如此，该方法假设表面为单层光滑体，不考虑层状介质内部的次表面散射或多重折射行为。Neuralangelo 专注于高精度几何恢复，引入多分辨率哈希编码配合神经体渲染，但其本质仍是对 3D 密度场进行优化[^17] [^1]，并不显式区分表面反射与体积散射。Gaussian Splatting 等新技术通过定向高斯点云实现快速渲染，可替代传统 NeRF，但它们的渲染流程依然是对颜色场的插值，不包含物理意义的分层 BRDF 或 BSSRDF 分量。正如 PBR-NeRF 指出的那样，目前大部分 NeRF 及基于体积渲染的方案只建模视见效应而未显式估计材质与照明参数[^18]；其结果常常缺乏物理约束，容易产生能量守恒或互惠性上的偏差[^15] [^16]。尽管最新研究已开始将物理原则融入神经网络——如 PBR-NeRF 在损失函数中加入了物理先验以稳定材质估计[^18]，PBNBRDF 通过解析积分和重参数化保证神经 BRDF 的能量守恒和互惠[^15] [^16]——但这些方法主要聚焦于单层 BRDF，尚未系统性扩展到多层散射场景。因此，当前神经渲染框架普遍缺乏对复杂层状介质与非线性光学效应的建模能力，需要发展新的网络结构和训练策略以补足这些不足。

## 可行的综合建模策略

针对上述挑战，我们建议物理建模与神经表示相结合的方法来实现复杂表面的通用表示。一种思路是将分层介质结构逐层物理建模，并用神经网络拟合不可解析的复杂效应。例如，可为表面层使用菲涅尔 + 微表面模型处理镜面反射，而对下层介质使用散射–吸收模型（如 Kubelka–Munk 扩散、Dipole 模型或 SGGX 体积散射[^13]）描述漫反射成分；接着引入 MLP 来修正层间多次散射和各向异性等高阶效应（SpongeCake 的做法即基于此思路[^13]）。

在网络设计上，可采用多分支结构：一支输出每个表面层的物理参数（折射率、厚度、散射/吸收系数等），另一支输出颜色或光强度，用于拟合难以显式建模的细节。同时，模型可将 NeRF 等隐式场与显式层状 BRDF 结合，例如将隐式场定义为层间传输函数（类似于 Neural BSSRDF 或 Neural SSS 所做[^19] [^14]），并在渲染过程中施加物理约束。物理一致性可通过损失函数约束：例如强制互惠性（$f(\mathbf{i}, \mathbf{o}) = f(\mathbf{o}, \mathbf{i})$）和能量守恒（积分不超过 1）[^15] [^16]，以及保持菲涅尔系数在不同入射角下的变化规律[^7]。此外，可利用可微分渲染（如 Mitsuba 3）对输出结果进行光线追踪验证，将渲染图像误差反向传播至模型参数。通过上述混合策略，一方面保留了可解释的物理层次结构，另一方面借助神经网络的表达力处理复杂和异质效应，有望获得兼具可解释性、物理一致性与拓展性的模型。

## 推荐研究与工程路径

为系统实现这一策略，建议按以下步骤推进：

1. 数据采集：构建多层材料的高质量数据集。可选典型材料如涂层金属、复合塑料、皮肤样本等。采用多视角相机和已知光照（点光源或环境光）拍摄实体样本；结合光度计或扫描设备测量表面法线、反射率和散射参数。对层状介质，还可使用光学断层扫描（OCT）等手段获取内部结构（层厚度、散射谱）作为先验。现有公开数据集亦可参考：如 MERL 提供 100 种均匀材料的 BRDF 数据[^20]，Jensen 皮肤数据集中包含多层皮肤次表面散射数据等。此外，可以利用路径追踪器（Mitsuba、PBRT）合成数据集：按照不同层叠结构和材料参数模拟生成地面真值图像[^21]，以作为网络训练监督。Neural BSSRDF 的做法表明，使用引导光照的路径追踪生成训练样本是可行的[^21]。

2. 网络与模型结构：设计多尺度多分支的隐式表示。可以采用神经 SDF 或 NeRF 为几何基础，并扩展材质场输出（例如输出每点微表面法线、粗略散射系数等参数）。网络主干输出基础几何，分支 1 计算表面微表面模型参数（粗糙度、金属度）；分支 2 计算层间体积散射参数（散射/吸收系数）。也可引入专门的 BSSRDF 网络，类似 Neural BSSRDF 用 MLP 同时处理空间位置、入射和观测方向[^19]。训练时，可使用多损失：像素重投影误差、物理损失（能量守恒、互惠）和正则项（鼓励层次分离）。重要的是将菲涅尔方程、电磁传播等物理公式嵌入损失或网络设计，如通过显式层接口计算反射/透射光比[^7]。

3. 验证与评估：采用多种度量验证模型性能。可以使用前述合成数据或物理引擎渲染的图像作为基准，对比网络渲染结果与真实渲染。评价指标包括重投影误差（PSNR/SSIM）、光能误差（检查所反射光通量是否超过入射）、互惠性违例度等[^15][^16]。对于真实采集的案例，应进行定量主观评估，或者对比细节（如阴影边界、颜色泄露、层间混叠）与参考照片的相符程度。此外，可比较传统模型（分层 BSDF、蒙特卡洛）生成的渲染结果，验证所提混合模型的物理合理性和渲染质量。最终，通过数据驱动与物理验证相结合的方式，迭代改进模型结构和参数，直至达到所需的实用性与通用性平衡。

综上所述，为实现复杂分层表面既物理一致又实用通用的建模，需深入研究并结合经典物理模型与现代神经表示。可行的路径包括构建丰富数据集、设计结合物理先验的神经网络结构、以及采用可微渲染进行端到端优化和物理验证。这一系统化工程将为未来真实感渲染和逆渲染任务提供强大基础。

## 参考文献

1. **Neuralangelo: High-Fidelity Neural Surface Reconstruction** [^1] [^2] [^3] [^17]  
   NeRF 与 Neuralangelo 的高保真表面重建技术。  
   <https://ar5iv.labs.arxiv.org/html/2306.03092>

2. **NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images** [^4]  
   NeRO 对反射物体几何与 BRDF 的重建。  
   <https://liuyuan-pal.github.io/NeRO/>

3. **Exploring the Potential of Layered BRDF Models** [^5] [^11]  
   Weidlich 等对经典分层 BRDF 模型的系统回顾。  
   <https://history.siggraph.org/wp-content/uploads/2025/07/2009-SA_Courses_Weidlich_Exploring-the-Potential-of-Layered-BRDF-Models.pdf>

4. **Light Diffusion in Multi-Layered Translucent Materials** [^6]  
   Donner 等关于多层皮肤散射的近似求解（多层扩散）。  
   <https://www.researchgate.net/publication/220183364_Light_diffusion_in_multi-layered_translucent_materials>

5. **Reflection from Layered Surfaces due to Subsurface Scattering** [^7]  
   Hanrahan 和 Krueger 奠基性的层状散射模型。  
   <https://cseweb.ucsd.edu/~ravir/6998/papers/p165-hanrahan.pdf>

6. **A Comprehensive Framework for Rendering Layered Materials** [^8] [^9] [^10]  
   Wenzel Jakob 等关于分层 BSDF 的通用计算框架。  
   <https://research.cs.cornell.edu/layered-sg14/>

7. **A Spectral BSSRDF for Shading Human Skin** [^12]  
   Donner 等关于多层皮肤散射的研究（Multipole 模型）。  
   <http://www.cs.columbia.edu/cg/pdfs/150_donner08hskin.pdf>

8. **SpongeCake: A Layered Microflake Surface Appearance Model** [^13]  
   SpongeCake 现代分层微片模型。  
   <https://arxiv.org/abs/2110.07145>

9. **Neural SSS: Lightweight Object Appearance Representation** [^14]  
   Neural SSS 对次表面散射的隐式表示。  
   <https://orbit.dtu.dk/en/publications/neural-sss-lightweight-object-appearance-representation/>

10. **Physically Based Neural Bidirectional Reflectance Distribution Function** [^15] [^16]  
    PBNBRDF 关于物理约束（能量守恒、互惠性）的分析。  
    <https://arxiv.org/html/2411.02347v1>

11. **PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields** [^18]  
    PBR-NeRF 对物理约束的讨论。  
    <https://arxiv.org/abs/2412.09680>

12. **Neural BSSRDF: Object Appearance Representation Including Heterogeneous Subsurface Scattering** [^19] [^21]  
    Neural BSSRDF 全对象散射表示，及相关的路径追踪采样说明。  
    <https://arxiv.org/html/2312.15711v1>

13. **MERL BRDF Database** [^20]  
    MERL 数据库，支持数据驱动的训练策略。  
    <https://www.merl.com/brdf/>

<!-- 脚注 -->
[^1]: <https://ar5iv.labs.arxiv.org/html/2306.03092>
[^2]: <https://ar5iv.labs.arxiv.org/html/2306.03092>
[^3]: <https://ar5iv.labs.arxiv.org/html/2306.03092>
[^4]: <https://liuyuan-pal.github.io/NeRO/>
[^5]: <https://history.siggraph.org/wp-content/uploads/2025/07/2009-SA_Courses_Weidlich_Exploring-the-Potential-of-Layered-BRDF-Models.pdf>
[^6]: <https://www.researchgate.net/publication/220183364_Light_diffusion_in_multi-layered_translucent_materials>
[^7]: <https://cseweb.ucsd.edu/~ravir/6998/papers/p165-hanrahan.pdf>
[^8]: <https://research.cs.cornell.edu/layered-sg14/>
[^9]: <https://research.cs.cornell.edu/layered-sg14/>
[^10]: <https://research.cs.cornell.edu/layered-sg14/>
[^11]: <https://history.siggraph.org/wp-content/uploads/2025/07/2009-SA_Courses_Weidlich_Exploring-the-Potential-of-Layered-BRDF-Models.pdf>
[^12]: <http://www.cs.columbia.edu/cg/pdfs/150_donner08hskin.pdf>
[^13]: <https://arxiv.org/abs/2110.07145>
[^14]: <https://orbit.dtu.dk/en/publications/neural-sss-lightweight-object-appearance-representation/>
[^15]: <https://arxiv.org/html/2411.02347v1>
[^16]: <https://arxiv.org/html/2411.02347v1>
[^17]: <https://ar5iv.labs.arxiv.org/html/2306.03092>
[^18]: <https://arxiv.org/abs/2412.09680>
[^19]: <https://arxiv.org/html/2312.15711v1>
[^20]: <https://www.merl.com/brdf/>
[^21]: <https://arxiv.org/html/2312.15711v1>

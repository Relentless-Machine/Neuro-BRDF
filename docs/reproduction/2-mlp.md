# NeRO 中着色（shading）阶段如何获得空间位置信息

## 总览：从相机到颜色的计算路径

- 相机 → 射线：由内参 `K` 与位姿 `pose` 生成像素射线（起点 `rays_o`、方向 `rays_d`）。
  - 代码：`network/renderer.py::_construct_ray_batch`（两处：形状渲染器和材质渲染器分别实现）。
  - 内参 K（intrinsics）：把相机坐标系下的点投影到像素平面的 3×3 矩阵，包含焦距与主点。
    - 典型形式：K = [[fx, s, cx],[0, fy, cy],[0,0,1]]，常见 s=0；fx, fy 以像素计，cx, cy 为主点坐标。
    - 像素→相机射线方向：d_cam ∝ K^{-1}[u, v, 1]^T，再单位化。
  - 位姿 pose（extrinsics/位姿）：描述相机在世界中的位置与朝向，通常是 4×4 刚体变换矩阵，由旋转 R 和平移 t 组成。
    - 两种常见约定：
        1. 世界→相机：x_c = R x_w + t，T_cw = [R t; 0 1]。相机中心 C_w = −R^T t。
        2. 相机→世界：x_w = R x_c + t，T_wc = [R t; 0 1]，且 T_wc = (T_cw)^{-1}。
- 空间采样/相交 → 点与法线：
  - 形状渲染器（NeROShapeRenderer）：在单位球内沿射线分段取样 `z_vals`，得到空间点 `points = rays_o + rays_d * z`，并用 `SDFNetwork.gradient(points)` 得法线；用于着色的“空间位置”就是这些 `points`。
  - 材质渲染器（NeROMaterialRenderer）：用网格光线追踪获取相交点 `inters` 与法线 `normals`；`inters` 即着色使用的“空间位置”。
  - 代码：
    - 形状：`renderer.py::render_core` 中构造 `points`
    - 材质：`renderer.py::trace/_construct_ray_batch` 调用 `raytracing.RayTracer.trace` 获得 `inters`
- 着色网络：
  - 形状渲染器使用 `AppShadingNetwork`（`network/field.py`）
  - 材质渲染器使用 `MCShadingNetwork`（`network/field.py`）
  - 两者都直接接收空间点（`points`/`pts`）并在 MLP 内部将其与特征或编码组合，从而加入空间位置信息。

## 几何与空间位置的来源

### 1) 形状渲染路径（SDF 体渲染）

- 代码位置：`network/renderer.py::NeROShapeRenderer`
- 射线采样：
  - `sample_ray` 依据近远界（单位球相交近似）在 [near, far] 均匀取样，渐进式细分，得到 `z_vals`。
  - `render_core` 中通过
    - `dists = z_vals[...,1:] - z_vals[...,:-1]`
    - `mid_z_vals = z_vals + dists * 0.5`
    - `points = rays_o[...,None,:] + rays_d[...,None,:] * mid_z_vals[...,None]`
    得到每条射线在体内的空间采样点 `points`。
- 法线与特征：
  - 法线来自 `SDFNetwork.gradient(points)`。
  - 几何隐特征来自 `SDFNetwork(points)[..., 1:]`，即 SDF MLP 的非 SDF 输出部分，记为 `feature_vector`。
- 将“空间位置信息”加入着色：`AppShadingNetwork.forward(points, normals, view_dirs, feature_vectors, ...)`。

### 2) 材质渲染路径（显式网格）

- 代码位置：`network/renderer.py::NeROMaterialRenderer`
- 相交点：`raytracing/RayTracer.trace(rays_o, rays_d)` 返回：
  - `inters`（相交点，pn×3）→ 直接作为 `pts` 输入着色网络。
  - `normals`（面法线，经单位化与翻转以符合 NeuS 约定）。
- 将“空间位置信息”加入着色：`MCShadingNetwork.forward(pts, view_dirs, normals, ...)`。

## 着色网络如何使用空间位置

### AppShadingNetwork（形状渲染）

- 入口：`network/field.py::AppShadingNetwork.forward(points, normals, view_dirs, feature_vectors, human_poses, ...)`
- 直接使用点坐标：
  - 物性预测头直接拼接原始点坐标 `points` 与几何特征 `feature_vectors`：
    - `metallic = MLP([feature_vectors, points])`
    - `roughness = MLP([feature_vectors, points])`
    - `albedo = MLP([feature_vectors, points])`
  - 这些 MLP 由 `make_predictor(feats_dim+3, out)` 构建，三层 ReLU，支持 weight_norm；最后层激活根据任务选择（默认为 sigmoid / exp / none 等）。
- 间接光（inner light）/遮挡（occlusion）对位置的使用：
  - 位置编码 `pos_enc = get_embedder(self.cfg['light_pos_freq'], 3)`，对 `points` 做 NeRF 风格多频率正余弦编码。
  - `inner_light = MLP([pos_enc(points), IDE(reflective, roughness)])`
  - `inner_weight(≈ occlusion prob) = MLP([pos_enc(points).detach(), dir_enc(reflective).detach()])`
    - `dir_enc = get_embedder(6, 3)`，方向的多频率编码
    - `sph_enc = generate_ide_fn(5)`，集成方向编码（IDE，基于球谐和 vMF 抑制项）
- 直接光（outer light）：
  - 默认只依赖方向 IDE：`outer_light = MLP(IDE(direction))`
  - 当 `sphere_direction=True` 时还会构造“反射方向与单位球的交点”的球面位置编码：先用 `get_sphere_intersection(sph_points, reflective)` 计算从点出发沿反射方向击中单位球的距离，再取交点并做 IDE，和 `IDE(reflective)` 拼接送入 MLP。
- 人体反光（human light，可选）：
  - 通过 `get_camera_plane_intersection(points, reflective, human_poses)` 将点沿反射方向投影到摄影者所在相机平面，做 IDE/IPE 编码，进而估计“人光”强度与权重。

小结：AppShadingNetwork 既直接用“原始点坐标”参与物性回归，也通过“对点坐标做位置编码”影响间接光/遮挡估计；方向相关项则通过 IDE/PE 进入。由此，空间位置信息多路径注入到着色 MLP 中。

### MCShadingNetwork（材质渲染）

- 入口：`network/field.py::MCShadingNetwork.forward(pts, view_dirs, normals, human_poses, step, is_train)`
- 材质特征/物性预测：
  - `MaterialFeatsNetwork` 先对 `pts` 做 8 频位置编码（`get_embedder(8, 3)`），两段残差样式 MLP 得到中间特征 `feats`。
  - 物性头为 `metallic/roughness/albedo = MLP([feats, pts])`。
  - 其中 `roughness` 会被映射到合法区间（训练内部以“平方粗糙度”形式使用）。
- 光照估计与位置：
  - 间接光（inner）使用 `inner_light = MLP([pos_enc(pts), IDE(reflections)])`。
  - 直接光（outer）默认仅依赖方向 IDE；若配置为 `sphere_direction` 会使用“方向 + 球面交点的 IDE 拼接”。
  - 人体反光（可选）同 `AppShadingNetwork` 思路，利用 `get_camera_plane_intersection(pts, directions, human_poses)`。
- 蒙特卡洛/重要性采样：
  - 从法线/反射方向在切平面内采样一组 `directions`，对每个 `(pts, direction)` 查询光照：
    - 若从 `pts` 沿 `direction` 发射的“光线”命中物体（通过 `ray_trace_fun` → `RayTracer.trace`），则取“inner lights”（`get_inner_lights`），其本质是“从交点出发沿入射方向与视线方向的 BRDF 组合 + MLP 预测的间接光”。
    - 未命中则取“outer lights”（环境）和可选的人体反光；两者本身不依赖 `pts`，但人光权重与相交运算依赖 `pts`。

小结：MCShadingNetwork 对点坐标进行位置编码（以及与方向编码拼接）来建模间接光；物性与 `pts`（及其编码）直接相关。点坐标同时决定“光线与几何是否相交”的可见性，从而改变光照组合。

## 位置/方向编码与 MLP 结构

- NeRF 风格位置编码：`network/field.py::get_embedder(multires, input_dims=3)`
  - 将输入坐标 x 映射为 `[x, sin(2^k x), cos(2^k x)]` 的拼接（k=0..multires-1），扩大感受野与频率覆盖。
- 集成方向编码 IDE：`utils/ref_utils.py::generate_ide_fn(deg_view)`
  - 依据球谐基展开与 vMF 分布的方差抑制项，对方向进行多阶稳健编码，默认 `deg_view=5` 生成 72 维特征。
- `make_predictor(in_dim, out_dim, activation)`
  - 统一的 MLP 构建器：线性+ReLU×3，最后一层按任务用 `sigmoid/exp/none/relu`。
  - 大量着色子头（材质、光照、可见性）均通过它搭建。
- SDFNetwork（几何 MLP）：`network/field.py::SDFNetwork`
  - 输入为三维坐标（带或不带位置编码），输出第一维是签名距离 `sdf`，其余维度作为几何隐特征 `feature_vector` 提供给着色网络。

## 可见性/遮挡与位置

- 形状渲染器的“遮挡监督”：`renderer.py::compute_occ_loss`
  - 在表面附近（`|sdf| < 阈值`）从点 `points` 沿镜面反射方向调用 `get_intersection(sdf_fun, inv_fun, points, reflective)`
  - 该函数内部在单位球内基于 `SDF` 沿射线做分层重要性采样，得到“命中概率”近似 `occ_prob_gt`，用来监督 `AppShadingNetwork` 里基于 `points` 的 `inner_weight` 预测。
- 材质渲染器的可见性依赖显式网格：`raytracing/RayTracer.trace` 返回是否命中，用于区分内外光照路径（`get_lights`）。

## 数据形状与接口约定（简要概括）

- `points/pts`：`[pn, 3]` 或 `[rn, sn, 3]`（射线数 rn × 采样数 sn），单位球内。
- `normals`：与 `points` 对齐的 `[*, 3]` 单位法线。
- `view_dirs/directions/reflective`：单位方向向量。
- 着色输出：
  - 颜色 `rgb_pr`/`color`（线性空间经 `linear_to_srgb` 裁剪至 [0,1]）。
  - 中间量：`diffuse_light/specular_light/specular_ref/occ_prob/indirect_light/human_light` 等。

## 坐标系与边界注意点

- 绝大多数体渲染逻辑在单位球内进行；`near_far_from_sphere`/`get_sphere_intersection` 以单位球包围场景。
- `offset_points_to_sphere` 在点模长接近 1 时做轻微收缩，避免数值不稳定。
- 法线方向与 NeuS 兼容性：材质渲染里对法线做了翻转提示（见 `NeROMaterialRenderer.trace`）。

## 结论：着色中“空间位置信息”的获得与使用方式

- 形状渲染器：空间位置来自“相机射线在单位球内的分段采样点 `points`”；法线来自 `SDF` 梯度；位置既直接进入物性 MLP（与几何特征拼接），也通过位置编码参与“间接光/遮挡”估计。
- 材质渲染器：空间位置来自“射线与网格的相交点 `inters`”；`pts` 经位置编码用于材质与间接光估计，同时决定与几何的相交关系（从而影响取 inner 还是 outer light 以及人光投影）。
- 两条路径都将“方向信息”与“空间位置信息”同时送入着色 MLP，通过 NeRF 风格的位置/方向编码和 IDE 编码，使网络具备丰富的空间与角度响应能力。

---

参考代码入口与关键符号（部分）：

- `network/renderer.py`：`NeROShapeRenderer`, `NeROMaterialRenderer`, `render_core`, `_construct_ray_batch`, `near_far_from_sphere`
- `network/field.py`：`SDFNetwork`, `AppShadingNetwork`, `MCShadingNetwork`, `MaterialFeatsNetwork`, `make_predictor`, `get_embedder`, `generate_ide_fn`, `get_intersection`
- `raytracing/raytracer.py`：`RayTracer.trace`
- `utils/ref_utils.py`：IDE/IPE 编码实现
- `utils/base_utils.py`：辅助函数（采样、颜色空间等）
- ...

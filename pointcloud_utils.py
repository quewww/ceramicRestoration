import numpy as np
import open3d as o3d
import os
import struct


def _read_ply_fallback(filepath):
    """兼容 Blender / binary PLY：在 Open3D 读取失败时手动解析 x/y/z 和可选 rgb/normal。"""
    with open(filepath, 'rb') as f:
        data = f.read()

    if not data:
        raise ValueError('PLY 文件为空')

    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        text = data.decode('latin1')

    lines = text.splitlines()
    header_lines = []
    end_header_idx = None
    for idx, line in enumerate(lines):
        header_lines.append(line.strip())
        if line.strip() == 'end_header':
            end_header_idx = idx
            break

    if end_header_idx is None:
        raise ValueError('无法解析 PLY 头部信息')

    header = header_lines[:end_header_idx]
    format_line = next((h for h in header if h.startswith('format ')), '')
    format_name = format_line.split()[1] if len(format_line.split()) >= 2 else 'ascii'
    vertex_count = None
    properties = []
    element_vertices = False

    for line in header:
        if line.startswith('element vertex'):
            try:
                vertex_count = int(line.split()[-1])
                element_vertices = True
            except Exception:
                pass
        elif element_vertices and line.startswith('property '):
            parts = line.split()
            if len(parts) >= 3:
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_type, prop_name))

    if vertex_count is None or not properties:
        raise ValueError('PLY 头部缺少 vertex/property 定义')

    property_names = [p[1] for p in properties]
    xyz_idx = [property_names.index(k) for k in ['x', 'y', 'z'] if k in property_names]
    if len(xyz_idx) != 3:
        raise ValueError('PLY 头部中未找到 x/y/z 坐标属性')

    if format_name == 'ascii':
        body = text.split('end_header', 1)[1].strip()
        rows = [line.strip() for line in body.splitlines() if line.strip()]
        if len(rows) < vertex_count:
            raise ValueError('ASCII PLY 顶点数与头部声明不一致')
        xyz = []
        rgb = []
        for row in rows[:vertex_count]:
            vals = row.split()
            if len(vals) < 3:
                continue
            xyz.append([float(vals[0]), float(vals[1]), float(vals[2])])
            if len(vals) >= 6 and property_names[:3] == ['x', 'y', 'z']:
                pass
            if 'red' in property_names or 'green' in property_names or 'blue' in property_names:
                idx_r = property_names.index('red') if 'red' in property_names else None
                idx_g = property_names.index('green') if 'green' in property_names else None
                idx_b = property_names.index('blue') if 'blue' in property_names else None
                if idx_r is not None and idx_g is not None and idx_b is not None:
                    rgb.append([
                        float(vals[idx_r]) / 255.0,
                        float(vals[idx_g]) / 255.0,
                        float(vals[idx_b]) / 255.0
                    ])
        points = np.asarray(xyz, dtype=np.float32)
        if len(rgb) == len(points):
            return points, np.asarray(rgb, dtype=np.float32)
        return points, None

    # binary little/big endian
    body = data.split(b'end_header\n', 1)[1] if b'end_header\n' in data else data.split(b'end_header\r\n', 1)[1]
    total_bytes = len(body)
    if total_bytes == 0:
        raise ValueError('二进制 PLY 体数据为空')

    type_map = {
        'char': 'b', 'uchar': 'B', 'short': 'h', 'ushort': 'H',
        'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd'
    }
    fmt = '<' if 'little' in format_name else '>'
    struct_fmt = []
    for prop_type, _ in properties:
        if prop_type not in type_map:
            raise ValueError(f'暂不支持的 PLY property 类型: {prop_type}')
        struct_fmt.append(type_map[prop_type])
    item_size = struct.calcsize(fmt + ''.join(struct_fmt))
    if item_size <= 0 or vertex_count <= 0:
        raise ValueError('无法计算 PLY 二进制数据大小')

    points = []
    colors = []
    offset = 0
    for i in range(vertex_count):
        if offset + item_size > total_bytes:
            break
        item = struct.unpack(fmt + ''.join(struct_fmt), body[offset:offset + item_size])
        offset += item_size
        values = dict(zip(property_names, item))
        if 'x' in values and 'y' in values and 'z' in values:
            points.append([float(values['x']), float(values['y']), float(values['z'])])
        if 'red' in values and 'green' in values and 'blue' in values:
            colors.append([
                float(values['red']) / 255.0,
                float(values['green']) / 255.0,
                float(values['blue']) / 255.0
            ])

    if not points:
        raise ValueError('二进制 PLY 未解析出有效顶点')

    points = np.asarray(points, dtype=np.float32)
    if colors:
        colors = np.asarray(colors, dtype=np.float32)
        if len(colors) == len(points):
            return points, colors
    return points, None


# ==================== 统一归一化 API（全工程强制使用） ====================
# 策略：包围盒中心 + 最大维度半宽（PoinTr 训练标准，坐标范围 [-1, 1]）
# 禁止在其他文件中重复实现 normalize/denormalize 逻辑


def preprocess_points(points):
    """
    统一归一化：包围盒中心化 + 尺度归一化到 [-1, 1]

    这是 PoinTr 训练时使用的标准归一化方式。
    所有模块必须使用此函数，禁止各自实现。

    Args:
        points: (N, 3) numpy 数组，原始点云
    Returns:
        normalized: (N, 3) float32 归一化点云
        center: (3,) float32 包围盒中心
        scale: float32 最大维度半宽
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), np.float32(1.0)

    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)

    center = ((min_coord + max_coord) / 2.0).astype(np.float32)
    extent = (max_coord - min_coord).astype(np.float32)
    scale = np.max(extent) / 2.0
    if scale < 1e-8:
        scale = np.float32(1.0)

    normalized = ((points - center) / scale).astype(np.float32)

    print(f"[统一归一化] 中心点: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), 缩放: {scale:.4f}")
    print(f"[统一归一化] 归一后范围: x=[{normalized[:,0].min():.4f},{normalized[:,0].max():.4f}],"
          f" y=[{normalized[:,1].min():.4f},{normalized[:,1].max():.4f}],"
          f" z=[{normalized[:,2].min():.4f},{normalized[:,2].max():.4f}]")

    return normalized, center, scale


def postprocess_points(points, center, scale):
    """
    统一逆归一化：将模型输出点还原到原始坐标系

    所有模块必须使用此函数，禁止各自实现。

    Args:
        points: (N, 3) numpy 数组，归一化空间的点
        center: (3,) 包围盒中心（preprocess_points 返回值）
        scale: float32 缩放因子（preprocess_points 返回值）
    Returns:
        denormalized: (N, 3) float32 原始坐标点
    """
    points = np.asarray(points, dtype=np.float32)
    result = (points * scale + center).astype(np.float32)

    print(f"[统一逆归一化] 还原后范围: x=[{result[:,0].min():.4f},{result[:,0].max():.4f}],"
          f" y=[{result[:,1].min():.4f},{result[:,1].max():.4f}],"
          f" z=[{result[:,2].min():.4f},{result[:,2].max():.4f}]")

    return result


def read_point_cloud(filepath):
    """读取点云文件，支持 .ply, .xyz, .txt, .pcd 格式"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.ply':
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if pcd.is_empty():
                raise ValueError('Open3D 读取为空')
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            if len(points) == 0:
                raise ValueError('Open3D 读取到空点云')
            return points, colors
        except Exception as e:
            print(f"[点云读取] Open3D 读取失败，切换到兼容 PLY 解析: {e}")
            points, colors = _read_ply_fallback(filepath)
            return points, colors
    elif ext == '.xyz':
        try:
            points = np.loadtxt(filepath, dtype=np.float32, usecols=(0, 1, 2))
            if len(points) == 0:
                raise ValueError("空点云")
            return points, None
        except Exception as e:
            print(f"np.loadtxt 失败，尝试手动读取: {e}")
            points = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except:
                                pass
            if len(points) == 0:
                raise ValueError("读取到空点云")
            return np.array(points, dtype=np.float32), None
    elif ext == '.txt':
        try:
            points = np.loadtxt(filepath, dtype=np.float32, usecols=(0, 1, 2))
            if len(points) == 0:
                raise ValueError("空点云")
            return points, None
        except Exception as e:
            print(f"np.loadtxt 失败，尝试手动读取: {e}")
            points = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except:
                                pass
            if len(points) == 0:
                raise ValueError("读取到空点云")
            return np.array(points, dtype=np.float32), None
    elif ext == '.pcd':
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def write_point_cloud(points, filepath):
    """将点云写入 .ply 文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)


def downsample_point_cloud(points, target_points=20000):
    """体素降采样，将点数减少到 target_points 以内"""
    if len(points) <= target_points:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_size = 0.01
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    while len(np.asarray(downsampled_pcd.points)) > target_points and voxel_size < 1.0:
        voxel_size *= 1.5
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    if len(np.asarray(downsampled_pcd.points)) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]

    return np.asarray(downsampled_pcd.points)


def convert_to_ply(input_path, output_path):
    """将 .xyz/.txt 格式转换为 .ply 格式"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in ['.xyz', '.txt']:
        raise ValueError(f"不支持转换为 PLY 的格式: {ext}")

    points, _ = read_point_cloud(input_path)
    write_point_cloud(points, output_path)


def count_points(filepath):
    """统计点云文件中的点数"""
    try:
        points, _ = read_point_cloud(filepath)
        return len(points)
    except Exception as e:
        print(f"计数失败: {e}")
        return 0


def estimate_chamfer_distance(pc1_path, pc2_path):
    """估算两个点云之间的 Chamfer Distance（用于计算修复精度）"""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PoinTr'))
        from extensions.chamfer_dist import ChamferDistanceL2

        points1, _ = read_point_cloud(pc1_path)
        points2, _ = read_point_cloud(pc2_path)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)

        if len(np.asarray(pcd1.points)) > 4096:
            pcd1 = pcd1.voxel_down_sample(0.02)
        if len(np.asarray(pcd2.points)) > 4096:
            pcd2 = pcd2.voxel_down_sample(0.02)

        xyz1 = np.asarray(pcd1.points).reshape(1, -1, 3).astype(np.float32)
        xyz2 = np.asarray(pcd2.points).reshape(1, -1, 3).astype(np.float32)

        import torch
        chamfer_dist = ChamferDistanceL2()
        distance = chamfer_dist(
            torch.from_numpy(xyz1),
            torch.from_numpy(xyz2)
        ).item()

        return distance
    except Exception as e:
        print(f"Chamfer Distance 计算失败: {e}")
        return None


# ==================== 轴对称检测 API ====================

def detect_axisymmetry(points, z_layers=20, theta_bins=12, threshold=0.12):
    """
    检测点云是否为轴对称物体（如陶瓷罐、花瓶等旋转体）。

    算法：
      1. PCA 求主方向，取 PC1 和 PC2 作为候选旋转轴
      2. 对每个候选轴：将点投影到轴向坐标 z 和径向 r
      3. 沿 z 轴分层，每层做角度分箱，计算径向分布的方差
      4. 汇总得到 axisymmetry score，score 越小越接近轴对称
      5. 取两个候选轴中 score 更小的作为最终轴

    Args:
        points: (N, 3) numpy 数组
        z_layers: 沿轴方向的分层数（默认 20）
        theta_bins: 每层的角度分箱数（默认 12，即每 30° 一个箱）
        threshold: 判定为轴对称的阈值（默认 0.12，越小越严格）

    Returns:
        dict: {
            is_axisymmetric: bool,
            score: float,              # 轴对称分数（越小越轴对称）
            axis_vector: (3,) float,   # 旋转轴方向向量（单位向量）
            axis_origin: (3,) float,   # 旋转轴通过的点（物体重心）
            pc1_score: float,          # PC1 作为轴时的分数
            pc2_score: float,          # PC2 作为轴时的分数
            best_pc: int               # 哪个主方向作为轴更好（0=PC1, 1=PC2）
        }
    """
    points = np.asarray(points, dtype=np.float32)
    n_pts = len(points)
    if n_pts < 50:
        print(f"[轴对称检测] 点云过少 ({n_pts} < 50)，无法检测")
        return {
            'is_axisymmetric': False,
            'score': 1.0,
            'axis_vector': np.array([0, 1, 0], dtype=np.float32),
            'axis_origin': np.mean(points, axis=0),
            'pc1_score': 1.0,
            'pc2_score': 1.0,
            'best_pc': -1
        }

    # PCA：中心化后对协方差矩阵做 SVD
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = centered.T @ centered / (n_pts - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh 返回升序，反转为降序
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # 取前两个主方向作为候选旋转轴
    candidates = [eigenvectors[:, 0], eigenvectors[:, 1]]  # PC1, PC2
    candidate_scores = []

    for axis_vec in candidates:
        axis_vec = axis_vec.astype(np.float32)

        # --- 投影：z = 轴向坐标, perp = 垂直于轴的分量 ---
        z_coords = centered @ axis_vec                   # (N,)
        perp_coords = centered - np.outer(z_coords, axis_vec)  # (N, 3)
        r_coords = np.linalg.norm(perp_coords, axis=1)   # (N,)
        theta_coords = np.arctan2(perp_coords[:, 1], perp_coords[:, 0])  # (N,)

        # --- 沿 z 轴分层（等距分层） ---
        z_min, z_max = z_coords.min(), z_coords.max()
        if z_max - z_min < 1e-8:
            candidate_scores.append(1.0)
            continue

        z_bin_edges = np.linspace(z_min, z_max, z_layers + 1)
        layer_scores = []

        for li in range(z_layers):
            mask = (z_coords >= z_bin_edges[li]) & (z_coords < z_bin_edges[li + 1])
            if li == z_layers - 1:  # 包含最后一个点
                mask = (z_coords >= z_bin_edges[li]) & (z_coords <= z_bin_edges[li + 1])
            if mask.sum() < 5:
                continue

            r_layer = r_coords[mask]
            theta_layer = theta_coords[mask]

            # --- 角度分箱 ---
            theta_min, theta_max = -np.pi, np.pi
            theta_bin_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
            bin_mean_r = []

            for bi in range(theta_bins):
                t_mask = (theta_layer >= theta_bin_edges[bi]) & (theta_layer < theta_bin_edges[bi + 1])
                if bi == theta_bins - 1:
                    t_mask = (theta_layer >= theta_bin_edges[bi]) & (theta_layer <= theta_bin_edges[bi + 1])
                if t_mask.sum() >= 2:
                    bin_mean_r.append(r_layer[t_mask].mean())

            if len(bin_mean_r) >= 3:
                bin_mean_r = np.array(bin_mean_r)
                mean_r = bin_mean_r.mean()
                if mean_r > 1e-8:
                    # 归一化方差 = 标准差 / 均值（变异系数）
                    cv = bin_mean_r.std() / mean_r
                    layer_scores.append(cv)

        if len(layer_scores) >= 3:
            candidate_scores.append(float(np.median(layer_scores)))
        else:
            candidate_scores.append(1.0)

    # 选取分数更低的候选轴
    pc1_score = candidate_scores[0]
    pc2_score = candidate_scores[1]

    if pc1_score <= pc2_score:
        best_idx = 0
        best_score = pc1_score
    else:
        best_idx = 1
        best_score = pc2_score

    best_axis = candidates[best_idx].astype(np.float32)
    is_axi = best_score < threshold

    result = {
        'is_axisymmetric': bool(is_axi),
        'score': float(best_score),
        'axis_vector': best_axis,
        'axis_origin': centroid.astype(np.float32),
        'pc1_score': float(pc1_score),
        'pc2_score': float(pc2_score),
        'best_pc': best_idx
    }

    print(f"[轴对称检测] score={best_score:.4f} (PC1={pc1_score:.4f}, PC2={pc2_score:.4f}), "
          f"best={'PC1' if best_idx == 0 else 'PC2'}, "
          f"axisymmetric={is_axi} (threshold={threshold})")

    return result


# ==================== 旋转轮廓重建 ====================

def _build_local_frame(axis_vector):
    """
    构建以 axis_vector 为 Z 轴的局部正交坐标系。

    Returns:
        x_axis, y_axis, z_axis: 三个 (3,) 单位向量，构成右手坐标系
        R: (3, 3) 旋转矩阵，列向量为 x_axis, y_axis, z_axis
    """
    z_axis = axis_vector / np.linalg.norm(axis_vector)

    # 选一个不平行的向量构造叉积
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(tmp, z_axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    x_axis = np.cross(z_axis, tmp)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])  # 3x3，列为轴向量
    return x_axis, y_axis, z_axis, R


def _smooth_profile(r_theta, window=5):
    """
    对 r(theta) 做环形平滑（首尾相接）。
    优先使用 Savitzky-Golay 滤波器（scipy.signal.savgol_filter），
    若不可用则回退到环形移动平均。

    Args:
        r_theta: (n_theta,) float 数组
        window: 平滑窗口大小（奇数）
    Returns:
        smoothed: (n_theta,) float 数组
    """
    n = len(r_theta)
    if window < 2 or n < window:
        return r_theta.copy()

    # 环形填充：首尾相接
    pad = window // 2
    padded = np.concatenate([r_theta[-pad:], r_theta, r_theta[:pad]])

    try:
        from scipy.signal import savgol_filter
        # Savitzky-Golay：window=11, polyorder=3（参考实现）
        sg_window = min(11, n if n % 2 == 1 else n - 1)
        if sg_window < 5:
            sg_window = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
        sg_window = max(3, sg_window)
        if sg_window % 2 == 0:
            sg_window -= 1
        polyorder = min(3, sg_window - 1)
        smoothed = savgol_filter(padded, sg_window, polyorder, mode='valid')
        # valid 模式输出长度 = len(padded) - sg_window + 1
        # 需要截取与原始 n 等长
        if len(smoothed) >= n:
            start = (len(smoothed) - n) // 2
            smoothed = smoothed[start:start + n]
        return smoothed.astype(np.float32)
    except Exception:
        # 回退：环形移动平均
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(padded, kernel, mode='valid')
        if len(smoothed) >= n:
            start = (len(smoothed) - n) // 2
            smoothed = smoothed[start:start + n]
        return smoothed.astype(np.float32)


def _interpolate_profiles(profiles, z_vals, valid_mask):
    """
    对缺失层的径向轮廓做线性插值补全。

    Args:
        profiles: list of (n_theta,) 数组，缺失层为 None
        z_vals: (n_z,) z 值
        valid_mask: (n_z,) bool，有效层为 True
    Returns:
        filled_profiles: list of (n_theta,) 数组，全部有效
    """
    n_z = len(profiles)
    if valid_mask.sum() >= n_z - 1:
        return profiles

    valid_idx = np.where(valid_mask)[0]
    filled = [p.copy() if p is not None else None for p in profiles]

    for i in range(n_z):
        if filled[i] is not None:
            continue

        left_idx = valid_idx[valid_idx < i]
        right_idx = valid_idx[valid_idx > i]

        if len(left_idx) == 0 and len(right_idx) > 0:
            nearest = right_idx[0]
            filled[i] = filled[nearest].copy()
        elif len(right_idx) == 0 and len(left_idx) > 0:
            nearest = left_idx[-1]
            filled[i] = filled[nearest].copy()
        elif len(left_idx) > 0 and len(right_idx) > 0:
            li = left_idx[-1]
            ri = right_idx[0]
            t = (i - li) / (ri - li) if ri != li else 0.5
            filled[i] = (1.0 - t) * filled[li] + t * filled[ri]
        else:
            filled[i] = np.full(len(profiles[0]), 0.1, dtype=np.float32)

    return filled


def revolve_reconstruction(points, axis_vector, axis_origin,
                           n_z=128, n_theta=72,
                           smooth_window=5, min_pts_per_layer=10,
                           seal_bottom=True, seal_top=True):
    """
    旋转轮廓重建：沿轴分层提取径向轮廓，生成旋转体三角网格。

    流程：
      1. 以 axis_vector 为 Z 轴构建局部坐标系
      2. 将点云变换到局部坐标
      3. 沿 Z 轴等距分层，每层提取 r(theta) 径向轮廓
      4. 对轮廓做平滑 + 缺失层插值
      5. 按 theta 生成环点，连接相邻环构建三角网格
      6. 可选：闭合顶部和底部

    Args:
        points: (N, 3) numpy 数组，原始点云
        axis_vector: (3,) 旋转轴方向单位向量
        axis_origin: (3,) 旋转轴通过点
        n_z: 沿轴方向的分层数（默认 128）
        n_theta: 每层的角度采样数（默认 72，即每 5° 一个采样点）
        smooth_window: 径向轮廓平滑窗口（默认 5）
        min_pts_per_layer: 每层最少点数，低于此数视为缺失（默认 10）
        seal_bottom: 是否闭合底面（默认 True）
        seal_top: 是否闭合顶面（默认 True）

    Returns:
        dict: {
            'mesh': open3d.geometry.TriangleMesh,  # 重建的旋转体网格
            'z_vals': (n_z,) float32,              # 每层的 z 坐标（局部系）
            'profiles': list of (n_theta,) float32,  # 每层的 r(theta)
            'rings': (n_z, n_theta, 3) float32,    # 每层环的 3D 坐标（世界系）
            'valid_layers': (n_z,) bool,            # 原始有效层
            'interpolated': int,                    # 插值补全的层数
            'success': bool
        }
    """
    points = np.asarray(points, dtype=np.float32)
    axis_vector = np.asarray(axis_vector, dtype=np.float32)
    axis_origin = np.asarray(axis_origin, dtype=np.float32)
    n_pts = len(points)

    if n_pts < 20:
        print("[旋转重建] 点云过少，无法重建")
        return {'mesh': None, 'success': False, 'profiles': [], 'rings': None,
                'z_vals': None, 'valid_layers': None, 'interpolated': 0}

    # ---- 1. 构建局部坐标系 ----
    x_axis, y_axis, z_axis, R = _build_local_frame(axis_vector)
    # 平移 + 旋转到局部坐标
    pts_local = (points - axis_origin) @ R  # (N, 3)，列为 x_axis, y_axis, z_axis

    # ---- 2. 沿 Z 轴分层 ----
    z_coords = pts_local[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    if z_max - z_min < 1e-6:
        print("[旋转重建] Z 方向范围过小，无法分层")
        return {'mesh': None, 'success': False, 'profiles': [], 'rings': None,
                'z_vals': None, 'valid_layers': None, 'interpolated': 0}

    z_vals = np.linspace(z_min, z_max, n_z).astype(np.float32)
    z_layer_edges = np.linspace(z_min, z_max, n_z + 1)

    # ---- 3. 逐层提取径向轮廓 ----
    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2  # (n_theta,)
    profiles = [None] * n_z
    valid_mask = np.zeros(n_z, dtype=bool)

    for li in range(n_z):
        z0, z1 = z_layer_edges[li], z_layer_edges[li + 1]
        if li == n_z - 1:
            mask = (z_coords >= z0) & (z_coords <= z1)
        else:
            mask = (z_coords >= z0) & (z_coords < z1)

        if mask.sum() < min_pts_per_layer:
            continue

        layer = pts_local[mask]
        x = layer[:, 0]
        y = layer[:, 1]
        thetas = np.arctan2(y, x)
        rs = np.sqrt(x * x + y * y)

        # 按角度分箱，每箱取中位数；空箱用环形邻居线性插值
        bin_idx = np.digitize(thetas, theta_edges) - 1
        r_bin = np.zeros(n_theta, dtype=np.float32)
        bin_has_data = np.zeros(n_theta, dtype=bool)

        for b in range(n_theta):
            sel = rs[bin_idx == b]
            if sel.size == 0:
                r_bin[b] = 0.0
            else:
                r_bin[b] = float(np.median(sel))
                bin_has_data[b] = True

        # 对空箱做环形线性插值（避免 r=0 导致退化三角形）
        if not np.all(bin_has_data) and np.any(bin_has_data):
            for b in range(n_theta):
                if not bin_has_data[b]:
                    # 找左右最近的非空箱（环形）
                    left_b = (b - 1) % n_theta
                    right_b = (b + 1) % n_theta
                    # 向左找
                    li_found = None
                    for step in range(1, n_theta):
                        idx = (b - step) % n_theta
                        if bin_has_data[idx]:
                            li_found = idx
                            break
                    # 向右找
                    ri_found = None
                    for step in range(1, n_theta):
                        idx = (b + step) % n_theta
                        if bin_has_data[idx]:
                            ri_found = idx
                            break
                    if li_found is not None and ri_found is not None:
                        # 线性插值
                        t = (b - li_found) / ((ri_found - li_found) % n_theta) if ri_found != li_found else 0.5
                        r_bin[b] = (1.0 - t) * r_bin[li_found] + t * r_bin[ri_found]
                    elif li_found is not None:
                        r_bin[b] = r_bin[li_found]
                    elif ri_found is not None:
                        r_bin[b] = r_bin[ri_found]
                    else:
                        r_bin[b] = 0.0

        profiles[li] = r_bin
        valid_mask[li] = True

    n_valid = valid_mask.sum()
    print(f"[旋转重建] 有效层: {n_valid}/{n_z}")

    if n_valid < max(3, n_z // 4):
        print("[旋转重建] 有效层过少，放弃重建")
        return {'mesh': None, 'success': False, 'profiles': profiles,
                'rings': None, 'z_vals': z_vals, 'valid_layers': valid_mask,
                'interpolated': 0}

    # ---- 4. 缺失层插值 ----
    interpolated_count = int((~valid_mask).sum())
    if interpolated_count > 0 and n_valid >= 2:
        profiles = _interpolate_profiles(profiles, z_vals, valid_mask)
        print(f"[旋转重建] 插值补全 {interpolated_count} 层")

    # ---- 5. 轮廓平滑 ----
    for li in range(n_z):
        if profiles[li] is not None:
            profiles[li] = _smooth_profile(profiles[li], smooth_window)

    # ---- 6. 生成环点（世界坐标） ----
    cos_t = np.cos(theta_centers)  # (n_theta,)
    sin_t = np.sin(theta_centers)  # (n_theta,)

    rings = np.zeros((n_z, n_theta, 3), dtype=np.float32)
    for li in range(n_z):
        r = profiles[li]
        # 局部坐标 (x, y, z)
        x_local = r * cos_t  # (n_theta,)
        y_local = r * sin_t  # (n_theta,)
        z_local = z_vals[li]

        # 变换回世界坐标
        rings[li, :, 0] = x_local * x_axis[0] + y_local * y_axis[0] + z_local * z_axis[0] + axis_origin[0]
        rings[li, :, 1] = x_local * x_axis[1] + y_local * y_axis[1] + z_local * z_axis[1] + axis_origin[1]
        rings[li, :, 2] = x_local * x_axis[2] + y_local * y_axis[2] + z_local * z_axis[2] + axis_origin[2]

    # ---- 7. 构建三角网格 ----
    vertices = rings.reshape(-1, 3)  # (n_z * n_theta, 3)

    faces = []
    for li in range(n_z - 1):
        for ti in range(n_theta):
            ti_next = (ti + 1) % n_theta
            i00 = li * n_theta + ti
            i10 = (li + 1) * n_theta + ti
            i01 = li * n_theta + ti_next
            i11 = (li + 1) * n_theta + ti_next

            # 两个三角形构成一个四边形
            faces.append([i00, i10, i01])
            faces.append([i01, i10, i11])

    # ---- 8. 闭合底面 ----
    if seal_bottom and n_valid > 0:
        bottom_center = np.array([0.0, 0.0, z_vals[0]], dtype=np.float32)
        bc_world = bottom_center[0] * x_axis + bottom_center[1] * y_axis + bottom_center[2] * z_axis + axis_origin
        center_idx = len(vertices)
        vertices = np.vstack([vertices, bc_world.reshape(1, 3)])

        for ti in range(n_theta):
            ti_next = (ti + 1) % n_theta
            i0 = 0 * n_theta + ti
            i1 = 0 * n_theta + ti_next
            faces.append([center_idx, i0, i1])

    # ---- 9. 闭合顶面 ----
    if seal_top and n_valid > 0:
        top_center_local = np.array([0.0, 0.0, z_vals[-1]], dtype=np.float32)
        tc_world = top_center_local[0] * x_axis + top_center_local[1] * y_axis + top_center_local[2] * z_axis + axis_origin
        center_idx = len(vertices)
        vertices = np.vstack([vertices, tc_world.reshape(1, 3)])

        for ti in range(n_theta):
            ti_next = (ti + 1) % n_theta
            i0 = (n_z - 1) * n_theta + ti
            i1 = (n_z - 1) * n_theta + ti_next
            faces.append([center_idx, i1, i0])

    # ---- 10. 创建 open3d 网格 ----
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    mesh.compute_vertex_normals()

    n_verts = len(vertices)
    n_faces = len(faces)
    print(f"[旋转重建] 完成: {n_verts} 顶点, {n_faces} 三角面")

    return {
        'mesh': mesh,
        'z_vals': z_vals,
        'profiles': profiles,
        'rings': rings,
        'valid_layers': valid_mask,
        'interpolated': interpolated_count,
        'success': True
    }
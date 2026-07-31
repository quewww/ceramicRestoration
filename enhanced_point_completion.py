# enhanced_point_completion.py - 增强版点云补全模块

import sys
import os

try:
    import numpy as np
    print("NumPy 导入成功")
except ImportError as e:
    print(f"NumPy 导入失败: {e}")
    raise

import open3d as o3d

# 统一归一化 API（强制使用，禁止自行实现）
from pointcloud_utils import preprocess_points as unified_preprocess
from pointcloud_utils import postprocess_points as unified_postprocess

try:
    import torch
    print(f"PyTorch 导入成功，版本: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch 导入失败: {e}")
    raise

import numpy as np

base_dir = os.path.dirname(__file__)
pointr_dir = os.path.join(base_dir, 'PoinTr')
sys.path.insert(0, pointr_dir)

try:
    from pointnet2_ops import pointnet2_utils
    print("使用 CUDA 加速的 pointnet2_ops")
except ImportError:
    print("使用纯 PyTorch 替代方案")
    import pointnet2_utils_pytorch as pointnet2_utils
    sys.modules['pointnet2_ops'] = type(sys)('pointnet2_ops')
    sys.modules['pointnet2_ops.pointnet2_utils'] = pointnet2_utils

_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        try:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            _device = torch.device('cpu')
    return _device


def load_model():
    global _model
    if _model is not None:
        return _model

    device = get_device()
    ckpt_path = os.path.join(pointr_dir, 'ckpts', 'pointr_training_from_scratch_c55_best.pth')
    config_path = os.path.join(pointr_dir, 'cfgs', 'ShapeNet55_models', 'PoinTr.yaml')

    if not os.path.exists(ckpt_path):
        print(f"错误: 模型文件不存在: {ckpt_path}")
        return None

    try:
        original_cwd = os.getcwd()
        os.chdir(pointr_dir)

        from utils.config import cfg_from_yaml_file
        config = cfg_from_yaml_file('cfgs/ShapeNet55_models/PoinTr.yaml')

        from models.PoinTr import PoinTr
        model = PoinTr(config.model)

        os.chdir(original_cwd)

        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'base_model' in checkpoint:
                state_dict = checkpoint['base_model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(device)
        model.eval()

        _model = model
        print(f"模型加载成功! 设备: {device}")
        return model

    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def preprocess_points(points, target_num=2048):
    """预处理流程：严格按照用户规范实现
    
    处理流程：
    1. 将网格均匀采样为点云，初始采样点数量设为10000个
    2. 对点云进行数据层面的归一化：计算质心并平移到原点，再等比例缩放使最远点落在单位球面上
    3. 使用体素降采样将点云数量减少到2048个点，匹配PoinTr模型的标准输入要求
    """
    import numpy as np
    
    if len(points) == 0:
        return np.zeros((target_num, 3)), 1.0, np.zeros(3)
    
    # 步骤1：网格均匀采样到10000个点
    if len(points) != 10000:
        print(f"📊 网格均匀采样 ({len(points)} → 10000)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        try:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8)
            pcd = mesh.sample_points_uniformly(10000)
            points = np.asarray(pcd.points)
            print(f"  ✅ 网格均匀采样成功，生成 {len(points)} 个点")
        except Exception as e:
            print(f"  ⚠️ 网格采样失败 ({e})，使用简单采样")
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]
            else:
                remaining = 10000 - len(points)
                indices = np.random.choice(len(points), remaining, replace=True)
                points = np.vstack([points, points[indices]])
    
    # 步骤2：统一归一化（包围盒中心 + 最大维度半宽，PoinTr 训练标准）
    normalized, center, scale = unified_preprocess(points)
    
    # 步骤3：体素降采样到2048个点
    if len(normalized) > target_num:
        print(f"📉 体素降采样 ({len(normalized)} → {target_num})...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(normalized)
        
        voxel_size = 0.02
        pcd = pcd.voxel_down_sample(voxel_size)
        while len(np.asarray(pcd.points)) > target_num:
            voxel_size *= 1.1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(normalized)
            pcd = pcd.voxel_down_sample(voxel_size)
        normalized = np.asarray(pcd.points)
        
        if len(normalized) < target_num:
            indices = np.random.choice(len(normalized), target_num - len(normalized), replace=True)
            normalized = np.vstack([normalized, normalized[indices]])
    
    print(f"  ✅ 预处理完成，输出点数: {len(normalized)}")
    # 保持返回签名与调用方一致: (normalized, scale, center)
    return normalized, scale, center


def uniform_mesh_sampling(points, target_num):
    """将点云进行均匀采样，模拟网格采样效果"""
    import numpy as np
    
    if len(points) >= target_num:
        return points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        
        pcd = mesh.sample_points_uniformly(target_num)
        result = np.asarray(pcd.points)
        print(f"  ✅ 网格均匀采样成功，生成 {len(result)} 个点")
        return result
    except Exception as e:
        print(f"  ⚠️ 网格采样失败 ({e})，使用KNN插值")
        return knn_interpolation_upsample(points, target_num)


def poisson_reconstruction_upsample(points, target_num):
    """使用泊松重建从稀疏点云重建表面，再均匀采样"""
    import numpy as np
    
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        
        pcd = mesh.sample_points_uniformly(target_num)
        result = np.asarray(pcd.points)
        print("  ✅ 使用泊松重建上采样")
        return result
    except Exception as e:
        print(f"  ⚠️ 泊松重建失败 ({e})，使用KNN插值")
        return knn_interpolation_upsample(points, target_num)


def knn_interpolation_upsample(points, target_num):
    """使用KNN插值在现有点之间生成新点"""
    import numpy as np
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    current_points = np.asarray(pcd.points)
    
    new_points = []
    needed = target_num - len(current_points)
    
    for i in range(len(current_points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(current_points[i], 6)
        if k >= 3:
            neighbors = current_points[idx[1:]]
            for j in range(min(len(neighbors), 3)):
                if len(new_points) >= needed:
                    break
                t = np.random.rand() * 0.6
                new_point = current_points[i] + (neighbors[j] - current_points[i]) * t
                new_points.append(new_point)
    
    combined = np.vstack([current_points, np.array(new_points)])
    
    if len(combined) < target_num:
        remaining = target_num - len(combined)
        indices = np.random.choice(len(combined), remaining, replace=True)
        combined = np.vstack([combined, combined[indices]])
    
    print("  ✅ 使用KNN插值上采样")
    return combined[:target_num]


# postprocess_points 已统一到 pointcloud_utils.py
# 禁止在此文件中自行实现逆归一化逻辑


def run_point_completion(points, target_points=None):
    """
    运行PoinTr模型进行点云补全
    
    参数：
        points: 输入点云
        target_points: 目标生成的补全点数（不是总点数）
    
    返回：
        生成的补全点（不包含原始点）
    """
    model = load_model()
    if model is None:
        print("错误: 模型不可用")
        return None

    device = get_device()
    
    # 使用模型默认的2048作为预处理目标，这是模型输入要求
    normalized, scale, center = preprocess_points(points, target_num=2048)

    input_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            ret = model(input_tensor)

            if isinstance(ret, tuple) and len(ret) == 2:
                coarse_points, fine_points = ret
                if isinstance(fine_points, tuple):
                    fine_points = fine_points[0]
                try:
                    completed = fine_points.squeeze(0).cpu().numpy()
                except Exception as np_error:
                    print(f"使用 numpy() 失败，尝试 tolist(): {np_error}")
                    completed = fine_points.squeeze(0).cpu().tolist()
                    import numpy as np
                    completed = np.array(completed)
            else:
                try:
                    completed = ret.squeeze(0).cpu().numpy()
                except Exception as np_error:
                    print(f"使用 numpy() 失败，尝试 tolist(): {np_error}")
                    completed = ret.squeeze(0).cpu().tolist()
                    import numpy as np
                    completed = np.array(completed)

            completed = unified_postprocess(completed, center, scale)

            # 根据target_points调整输出数量
            if target_points is not None and target_points > 0:
                import numpy as np
                if len(completed) > target_points:
                    indices = np.random.choice(len(completed), target_points, replace=False)
                    completed = completed[indices]
                elif len(completed) < target_points:
                    remaining = target_points - len(completed)
                    indices = np.random.choice(len(completed), remaining, replace=True)
                    completed = np.vstack([completed, completed[indices]])

            return completed

        except Exception as e:
            print(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def merge_original_with_completion(original_points, completion_points, threshold_factor=2.0):
    """
    直接返回模型输出点云（PoinTr是全局重建模型，网络输出本身就是完整物体）
    
    参数：
        original_points: 原始点云（残缺的，不再使用）
        completion_points: PoinTr生成的完整点云
        threshold_factor: 距离阈值因子（保留参数但不再使用）
    
    返回：
        模型输出的完整点云（不与原始点云合并）
    """
    print(f"   原始点数: {len(original_points)}")
    print(f"   模型输出点数: {len(completion_points)}")
    print(f"   重要：PoinTr是全局重建模型，直接使用模型输出，不合并原始点云")
    
    return completion_points


def repair_point_cloud(input_points, target_points=None):
    """
    点云修复核心函数：保持原始点不变，补充缺失部分
    
    参数：
        input_points: 输入点云（残缺的）
        target_points: 目标总点数（默认根据输入自动计算，确保足够补全）
    
    策略选择：
        - 点多（>= 500点）：使用PoinTr模型（效果最好）
        - 点少且形状规则：使用形状参数化补全（圆柱、球体等）
        - 其他情况：使用对称性补全
    
    关键逻辑：
        - 保持原始点完全不变
        - 只添加新点来补全缺失部分
        - 输出点数 >= 输入点数（补全后更完整）
    """
    import numpy as np
    
    original_count = len(input_points)
    
    # 默认目标点数：至少是输入的2倍，确保有足够点补全缺失部分
    if target_points is None:
        target_points = max(original_count * 2, 2048)
    
    # 确保目标点数不小于原始点数
    if target_points < original_count:
        target_points = original_count + 1000
    
    # 需要生成的补全点数
    completion_count = max(target_points - original_count, 1000)
    
    print(f"\n点云修复开始...")
    print(f"输入点数: {original_count}, 目标点数: {target_points}, 需要补全: {completion_count}")
    print(f"输入范围 - X: [{input_points[:,0].min():.3f}, {input_points[:,0].max():.3f}]")
    print(f"输入范围 - Y: [{input_points[:,1].min():.3f}, {input_points[:,1].max():.3f}]")
    print(f"输入范围 - Z: [{input_points[:,2].min():.3f}, {input_points[:,2].max():.3f}]")
    
    # 点多时优先使用PoinTr
    if original_count >= 500:
        print("🚀 点云密度足够，使用PoinTr模型...")
        result = run_point_completion(input_points, completion_count)
        
        if result is not None and len(result) > 0:
            print(f"✅ PoinTr补全成功! 生成了 {len(result)} 个补全点")
            # 使用区域基于合并：保留原始点不变，只添加缺失区域的点
            combined = merge_original_with_completion(input_points, result)
            print(f"✅ 合并后总点数: {len(combined)}")
            return combined
        else:
            print("❌ PoinTr失败，尝试形状补全...")
    
    # 点少时使用形状检测和对称补全
    shape_type = detect_shape_type(input_points)
    print(f"📐 检测到形状类型: {shape_type}")
    
    if shape_type == 'cylinder':
        result = cylinder_completion(input_points, target_points)
    elif shape_type == 'sphere':
        result = sphere_completion(input_points, target_points)
    else:
        result = symmetry_based_completion(input_points, target_points)
    
    if result is not None and len(result) > 0:
        print(f"✅ 补全成功! 输出点数: {len(result)}")
        return result
    else:
        print("❌ 所有方案失败，使用基础补全")
        return basic_completion(input_points, target_points)


def sphere_completion(points, target_points):
    """针对球体的补全方法"""
    import numpy as np
    
    print("🔧 球体补全...")
    
    if len(points) < 10:
        return generate_sphere(target_points)
    
    try:
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.mean(distances)
        
        print(f"   球体参数: 中心={center}, 半径={radius:.3f}")
        
        # 生成完整的球体表面点
        theta = np.random.rand(target_points) * np.pi
        phi = np.random.rand(target_points) * 2 * np.pi
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        sphere_points = np.column_stack([x, y, z]) + center
        
        print(f"   生成了 {len(sphere_points)} 个球体表面点")
        return sphere_points
        
    except Exception as e:
        print(f"   ⚠️ 球体补全失败: {e}")
        return basic_completion(points, target_points)


def generate_sphere(target_points):
    """生成标准球体"""
    import numpy as np
    
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 40)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    x = 0.5 * np.sin(theta_grid) * np.cos(phi_grid)
    y = 0.5 * np.sin(theta_grid) * np.sin(phi_grid)
    z = 0.5 * np.cos(theta_grid)
    
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    if len(points) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        points = points[indices]
    elif len(points) < target_points:
        remaining = target_points - len(points)
        indices = np.random.choice(len(points), remaining, replace=True)
        points = np.vstack([points, points[indices]])
    
    return points


def detect_shape_type(points):
    """检测点云的形状类型"""
    import numpy as np
    
    if len(points) < 20:
        return 'unknown'
    
    cov_matrix = np.cov(points.T)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = sorted(eigenvalues, reverse=True)
    
    ratio1 = eigenvalues[0] / eigenvalues[1]
    ratio2 = eigenvalues[1] / eigenvalues[2]
    
    if ratio1 > 2 and ratio2 < 1.5:
        return 'cylinder'
    elif ratio1 < 1.5 and ratio2 < 1.5:
        return 'sphere'
    else:
        return 'symmetric'


def cylinder_completion(points, target_points):
    """针对圆柱形物体的补全方法"""
    import numpy as np
    
    print("🔧 圆柱形补全...")
    original_count = len(points)
    
    if original_count < 10:
        return generate_cylinder(target_points)
    
    try:
        center = np.mean(points, axis=0)
        
        cov_matrix = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        axis_dir = eigenvectors[:, sorted_indices[0]]
        axis_dir = axis_dir / np.linalg.norm(axis_dir)
        
        radius = estimate_radius(points, center, axis_dir)
        height_range = estimate_height_range(points, center, axis_dir)
        
        print(f"   圆柱参数: 半径={radius:.3f}, 高度范围={height_range}")
        
        # 生成目标点数的圆柱点
        new_points = generate_cylinder_points(center, axis_dir, radius, height_range, target_points)
        
        # 返回完整的圆柱点云（不保留原始点，因为原始点可能不完整）
        print(f"   生成了 {len(new_points)} 个圆柱表面点")
        return new_points
        
    except Exception as e:
        print(f"   ⚠️ 圆柱补全失败: {e}")
        return basic_completion(points, target_points)


def estimate_radius(points, center, axis_dir):
    """估计圆柱体半径"""
    import numpy as np
    
    radius_sum = 0
    count = 0
    
    for p in points:
        vec = p - center
        proj_len = np.dot(vec, axis_dir)
        proj_vec = proj_len * axis_dir
        radial_vec = vec - proj_vec
        radius_sum += np.linalg.norm(radial_vec)
        count += 1
    
    return radius_sum / count if count > 0 else 0.5


def estimate_height_range(points, center, axis_dir):
    """估计圆柱体高度范围"""
    import numpy as np
    
    projections = np.dot(points - center, axis_dir)
    return (projections.min(), projections.max())


def generate_cylinder_points(center, axis_dir, radius, height_range, num_points):
    """生成圆柱表面点"""
    import numpy as np
    
    if num_points <= 0:
        # 返回一个包含单个点的数组，避免拼接错误
        return np.array([center])
    
    theta = np.random.rand(num_points) * 2 * np.pi
    z = np.random.rand(num_points) * (height_range[1] - height_range[0]) + height_range[0]
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    circle_points = np.column_stack([x, y, z])
    
    rotation_matrix = rotation_matrix_from_axis(axis_dir)
    
    rotated_points = circle_points @ rotation_matrix.T
    
    translated_points = rotated_points + center
    
    return translated_points


def rotation_matrix_from_axis(axis):
    """从轴向量生成旋转矩阵"""
    import numpy as np
    
    axis = axis / np.linalg.norm(axis)
    
    if np.abs(axis[2] - 1) < 1e-6:
        return np.eye(3)
    
    if np.abs(axis[2] + 1) < 1e-6:
        return np.diag([1, -1, -1])
    
    a = axis[0]
    b = axis[1]
    c = axis[2]
    
    s = np.sqrt(a**2 + b**2)
    if s < 1e-6:
        return np.eye(3)
    
    cx = a / s
    sx = b / s
    
    R = np.array([
        [cx*c, sx*c, -s],
        [-sx, cx, 0],
        [cx*s, sx*s, c]
    ])
    
    return R


def generate_cylinder(target_points):
    """生成标准圆柱体"""
    import numpy as np
    
    theta = np.linspace(0, 2*np.pi, 60)
    z = np.linspace(0, 1, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = 0.5 * np.cos(theta_grid)
    y = 0.5 * np.sin(theta_grid)
    
    points = np.column_stack([x.flatten(), z_grid.flatten(), y.flatten()])
    
    if len(points) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        points = points[indices]
    elif len(points) < target_points:
        remaining = target_points - len(points)
        indices = np.random.choice(len(points), remaining, replace=True)
        points = np.vstack([points, points[indices]])
    
    return points


def basic_completion(points, target_points):
    """基础补全方法：保持原始点，只添加必要的点"""
    import numpy as np
    
    if len(points) >= target_points:
        return points[:target_points]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    current_points = np.asarray(pcd.points)
    
    needed = target_points - len(current_points)
    new_points = []
    
    for i in range(len(current_points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(current_points[i], 6)
        if k >= 3:
            neighbors = current_points[idx[1:]]
            for j in range(min(len(neighbors), 3)):
                if len(new_points) >= needed:
                    break
                t = np.random.rand() * 0.5
                new_point = current_points[i] + (neighbors[j] - current_points[i]) * t
                new_points.append(new_point)
    
    combined = np.vstack([current_points, np.array(new_points)])
    
    if len(combined) < target_points:
        remaining = target_points - len(combined)
        indices = np.random.choice(len(combined), remaining, replace=True)
        combined = np.vstack([combined, combined[indices]])
    
    return combined[:target_points]


def symmetry_based_completion(points, target_points):
    """
    基于对称性的点云补全：
    1. 保持原始点不变
    2. 检测对称平面
    3. 根据对称性生成缺失点
    4. 只在空白区域添加点
    """
    import numpy as np
    
    original_points = points.copy()
    print(f"🔍 对称性分析开始，原始点数: {len(original_points)}")
    
    if len(points) < 10:
        print("   ⚠️ 点太少，无法进行对称性分析")
        return None
    
    try:
        symmetry_plane = detect_symmetry_plane(points)
        print(f"   检测到对称平面")
        
        mirrored_points = mirror_points(points, symmetry_plane)
        
        # 保持原始点，添加镜像点
        combined = np.vstack([original_points, mirrored_points])
        
        # 不进行去重，保留所有点
        if len(combined) < target_points:
            combined = fill_sparse_regions(combined, target_points)
        
        print(f"   补全后点数: {len(combined)}")
        
        return combined  # 直接返回，不做删减
        
    except Exception as e:
        print(f"   ⚠️ 对称性分析失败: {e}")
        return None


def detect_symmetry_plane(points):
    """检测点云的主要对称平面"""
    import numpy as np
    
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    normal_vector = eigenvectors[:, sorted_indices[2]]
    
    center = np.mean(points, axis=0)
    
    return {'normal': normal_vector, 'point': center}


def mirror_points(points, plane):
    """将点云沿对称平面镜像"""
    import numpy as np
    
    normal = plane['normal']
    point = plane['point']
    
    normal = normal / np.linalg.norm(normal)
    
    mirrored = []
    for p in points:
        vector = p - point
        distance = np.dot(vector, normal)
        mirrored_point = p - 2 * distance * normal
        mirrored.append(mirrored_point)
    
    return np.array(mirrored)


def remove_duplicate_points(points, threshold=None):
    """移除重复或非常接近的点
    
    参数：
        points: 点云
        threshold: 距离阈值（默认根据点云范围自动计算）
    """
    import numpy as np
    
    if len(points) == 0:
        return points
    
    # 自动计算阈值：点云对角线长度的0.1%
    if threshold is None:
        if len(points) >= 2:
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            diagonal = np.linalg.norm(max_bound - min_bound)
            threshold = diagonal * 0.001  # 0.1%的对角线长度
        else:
            threshold = 0.001
    
    unique_points = [points[0]]
    
    for p in points[1:]:
        distances = np.linalg.norm(unique_points - p, axis=1)
        if np.min(distances) > threshold:
            unique_points.append(p)
    
    return np.array(unique_points)


def fill_sparse_regions(points, target_points):
    """在稀疏区域填充点"""
    import numpy as np
    
    if len(points) >= target_points:
        return points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    current_points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    needed = target_points - len(current_points)
    new_points = []
    
    for i in range(len(current_points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(current_points[i], 6)
        if k >= 4:
            neighbors = current_points[idx[1:]]
            for j in range(min(len(neighbors), 3)):
                if len(new_points) >= needed:
                    break
                t = np.random.rand() * 0.5
                new_point = current_points[i] + (neighbors[j] - current_points[i]) * t
                new_points.append(new_point)
    
    combined = np.vstack([current_points, np.array(new_points)])
    
    if len(combined) < target_points:
        remaining = target_points - len(combined)
        indices = np.random.choice(len(combined), remaining, replace=True)
        combined = np.vstack([combined, combined[indices]])
    
    return combined[:target_points]


def shape_based_completion(points, target_points=8192):
    """基于形状重建的补全方法：先重建表面网格，再均匀采样"""
    import numpy as np
    
    print("🔨 使用形状重建方法...")
    
    if len(points) < 10:
        print("   ⚠️ 点太少，生成基础形状")
        return generate_basic_shape(target_points)
    
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, scale=1.1)
        
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        mesh = mesh.filter_smooth_simple(number_of_iterations=2)
        
        pcd = mesh.sample_points_uniformly(target_points)
        result = np.asarray(pcd.points)
        
        print("   ✅ 形状重建成功")
        return result
        
    except Exception as e:
        print(f"   ⚠️ 泊松重建失败 ({e})，使用扩展方法")
        return extend_and_smooth(points, target_points)


def generate_basic_shape(target_points):
    """生成基础几何体作为默认补全"""
    import numpy as np
    
    print("   📦 生成圆柱形状...")
    
    theta = np.linspace(0, 2*np.pi, 60)
    z = np.linspace(0, 1, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = 0.5 * np.cos(theta_grid)
    y = 0.5 * np.sin(theta_grid)
    
    points = np.column_stack([x.flatten(), z_grid.flatten(), y.flatten()])
    
    if len(points) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        points = points[indices]
    elif len(points) < target_points:
        remaining = target_points - len(points)
        indices = np.random.choice(len(points), remaining, replace=True)
        points = np.vstack([points, points[indices]])
    
    return points


def extend_and_smooth(points, target_points):
    """扩展点云并进行平滑处理"""
    import numpy as np
    
    print("   🔧 使用扩展和平滑方法...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    
    for _ in range(3):
        pcd = pcd.filter_smooth_simple(number_of_iterations=2, lambda_filter=0.5)
    
    current_points = np.asarray(pcd.points)
    
    if len(current_points) >= target_points:
        indices = np.random.choice(len(current_points), target_points, replace=False)
        return current_points[indices]
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    new_points = []
    needed = target_points - len(current_points)
    
    for i in range(len(current_points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(current_points[i], 8)
        if k >= 4:
            neighbors = current_points[idx[1:]]
            for j in range(min(len(neighbors), 4)):
                if len(new_points) >= needed:
                    break
                for t in [0.3, 0.7]:
                    new_point = current_points[i] + (neighbors[j] - current_points[i]) * t
                    new_points.append(new_point)
    
    combined = np.vstack([current_points, np.array(new_points)])
    
    if len(combined) < target_points:
        remaining = target_points - len(combined)
        indices = np.random.choice(len(combined), remaining, replace=True)
        combined = np.vstack([combined, combined[indices]])
    
    return combined[:target_points]


def smooth_and_refine(points, target_points):
    """对模型输出进行平滑和细化处理"""
    import numpy as np
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
    
    pcd = pcd.filter_smooth_simple(number_of_iterations=2, lambda_filter=0.4)
    pcd = pcd.filter_smooth_taubin(number_of_iterations=2)
    
    result = np.asarray(pcd.points)
    
    if len(result) != target_points:
        if len(result) > target_points:
            indices = np.random.choice(len(result), target_points, replace=False)
            result = result[indices]
        else:
            indices = np.random.choice(len(result), target_points - len(result), replace=True)
            result = np.vstack([result, result[indices]])
    
    return result


def enhance_point_cloud(points, preserve_shape=True):
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    pcd = pcd.filter_smooth_simple(number_of_iterations=2, lambda_filter=0.5)
    
    pcd = pcd.filter_smooth_taubin(number_of_iterations=3)

    return np.asarray(pcd.points)


def light_enhance_point_cloud(points):
    """轻量级后处理：几乎不去噪，保留模型生成的完整形状"""
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5.0)

    return np.asarray(pcd.points)


def densify_point_cloud(points, target_points=20000):
    """通过插值增加点云密度"""
    if len(points) >= target_points:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    pcd = pcd.uniform_down_sample(every_k_points=1)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"平均邻距: {avg_dist:.4f}")
    
    new_points = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    current_points = np.asarray(pcd.points)
    
    for i in range(len(current_points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(current_points[i], 6)
        if k >= 3:
            neighbors = current_points[idx[1:]]
            for j in range(len(neighbors)):
                t = np.random.rand() * 0.5
                new_point = current_points[i] + (neighbors[j] - current_points[i]) * t
                new_points.append(new_point)
    
    combined = np.vstack([current_points, np.array(new_points)])
    
    if len(combined) < target_points:
        needed = target_points - len(combined)
        indices = np.random.choice(len(combined), needed, replace=True)
        combined = np.vstack([combined, combined[indices]])
    elif len(combined) > target_points:
        indices = np.random.choice(len(combined), target_points, replace=False)
        combined = combined[indices]
    
    return combined


def improved_rule_based_completion(points, target_points=8192):
    if len(points) == 0:
        return np.random.randn(target_points, 3) * 0.5

    if len(points) >= target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    scale = max_bound - min_bound

    needed_points = target_points - len(points)
    new_points = []

    num_copy = min(needed_points // 3, len(points))
    indices = np.random.choice(len(points), num_copy, replace=True)
    copied_points = points[indices] + np.random.randn(num_copy, 3) * (scale.mean() * 0.01)
    new_points.extend(copied_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=scale.mean()*0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    remaining = needed_points - num_copy
    for i in range(remaining):
        idx = np.random.randint(len(points))
        base_point = points[idx]
        normal = normals[idx] if len(normals) > 0 else np.array([0, 1, 0])
        offset = np.random.rand() * scale.mean() * 0.05
        new_point = base_point + normal * offset + np.random.randn(3) * (scale.mean() * 0.005)
        
        dists = np.linalg.norm(points - new_point, axis=1)
        if np.min(dists) < scale.mean() * 0.03:
            continue
        
        new_points.append(new_point)

    completed = np.vstack([points, np.array(new_points)])

    if len(completed) < target_points:
        indices = np.random.choice(len(completed), target_points - len(completed), replace=True)
        completed = np.vstack([completed, completed[indices]])
    elif len(completed) > target_points:
        indices = np.random.choice(len(completed), target_points, replace=False)
        completed = completed[indices]

    return completed


if __name__ == "__main__":
    print("=" * 70)
    print("增强版点云补全测试")
    print("=" * 70)

    def create_half_cube():
        points = []
        for x in np.linspace(-0.5, 0.5, 20):
            for y in np.linspace(0, 0.5, 10):
                for z in np.linspace(-0.5, 0.5, 20):
                    points.append([x, y, z])
        return np.array(points)

    input_pts = create_half_cube()
    print(f"\n输入点数: {len(input_pts)}")

    result = repair_point_cloud(input_pts, target_points=8192)

    if result is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result)
        o3d.io.write_point_cloud("enhanced_repaired.ply", pcd)
        
        print(f"\n修复结果已保存: enhanced_repaired.ply")
        print(f"输出点数: {len(result)}")
        print(f"输出范围:")
        print(f"  X: [{result[:,0].min():.3f}, {result[:,0].max():.3f}]")
        print(f"  Y: [{result[:,1].min():.3f}, {result[:,1].max():.3f}]")
        print(f"  Z: [{result[:,2].min():.3f}, {result[:,2].max():.3f}]")
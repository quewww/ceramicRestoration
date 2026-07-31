# point_cloud_completion.py - PoinTr 预训练模型使用
import numpy as np
import open3d as o3d
import torch
import sys
import os

# 统一归一化 API（强制使用，禁止自行实现）
from pointcloud_utils import preprocess_points, postprocess_points

# 第一步：设置路径
base_dir = os.path.dirname(__file__)
pointr_dir = os.path.join(base_dir, 'PoinTr')
sys.path.insert(0, pointr_dir)

# 第二步：关键 - 在导入任何 PoinTr 模块之前创建并替换模块
# 创建一个替代 pointnet2_ops 的模块
class Pointnet2OpsModule:
    def __init__(self):
        # 导入我们的 PyTorch 实现
        import pointnet2_utils_pytorch
        self.pointnet2_utils = pointnet2_utils_pytorch

# 现在注册这个模块到 sys.modules 中
# 这样当 PoinTr 尝试导入 pointnet2_ops.pointnet2_utils 时，会看到我们的版本
sys.modules['pointnet2_ops'] = Pointnet2OpsModule()

# 现在尝试导入以验证
try:
    from pointnet2_ops import pointnet2_utils
    # 验证是否有我们需要的函数
    required_funcs = ['furthest_point_sample', 'gather_operation', 'grouping_operation', 
                     'query_ball_point', 'knn_point', 'index_points', 'square_distance']
    
    missing_funcs = [f for f in required_funcs if not hasattr(pointnet2_utils, f)]
    
    if missing_funcs:
        print(f"警告: 缺失函数: {missing_funcs}")
    else:
        print("成功: PyTorch 替代方案加载完成")
        
except Exception as e:
    print(f"替代方案加载失败: {e}")
    import traceback
    traceback.print_exc()

_model = None
_device = None


def get_device():
    """获取可用设备"""
    global _device
    if _device is None:
        try:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {_device}")
        except:
            _device = torch.device('cpu')
            print("使用设备: cpu")
    return _device


def load_model():
    """加载 PoinTr 预训练模型"""
    global _model

    if _model is not None:
        return _model

    device = get_device()

    ckpt_path = os.path.join(pointr_dir, 'ckpts', 'pointr_training_from_scratch_c55_best.pth')
    config_path = os.path.join(pointr_dir, 'cfgs', 'ShapeNet55_models', 'PoinTr.yaml')

    print("\n" + "="*70)
    print("正在加载 PoinTr 模型...")
    print(f"模型文件: {ckpt_path}")
    print(f"配置文件: {config_path}")
    print("="*70 + "\n")

    if not os.path.exists(ckpt_path):
        print("\n错误: 模型文件不存在!")
        print(f"请确认文件路径: {ckpt_path}")
        return None

    if not os.path.exists(config_path):
        print("\n错误: 配置文件不存在!")
        print(f"请确认文件路径: {config_path}")
        return None

    try:
        # 保存原始工作目录
        original_cwd = os.getcwd()
        # 切换到 PoinTr 目录下加载配置
        os.chdir(pointr_dir)
        
        from utils.config import cfg_from_yaml_file
        
        # 使用相对路径加载配置
        config_relative = os.path.join('cfgs', 'ShapeNet55_models', 'PoinTr.yaml')
        config = cfg_from_yaml_file(config_relative)

        print("模型配置:")
        print(f"  - 输出点数: {config.model.num_pred}")
        print(f"  - 查询点数: {config.model.num_query}")

        # 现在导入 PoinTr 模型（这会尝试使用我们的替代模块）
        from models.PoinTr import PoinTr
        model = PoinTr(config.model)

        # 切换回原始工作目录
        os.chdir(original_cwd)

        print("\n正在加载模型权重...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ['base_model', 'model', 'state_dict']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"  - 使用键: {key}")
                    break
            if state_dict is None:
                state_dict = checkpoint
        
        if state_dict is None:
            state_dict = checkpoint

        # 移除 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        print("  - 加载状态字典...")
        model.load_state_dict(new_state_dict, strict=False)

        model = model.to(device)
        model.eval()

        _model = model

        print("\nPoinTr 模型加载成功!")
        return model

    except Exception as e:
        # 确保恢复原始工作目录
        try:
            os.chdir(original_cwd)
        except:
            pass
            
        print(f"\n错误: 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# preprocess_points / postprocess_points 已统一到 pointcloud_utils.py
# 禁止在此文件中自行实现归一化逻辑
# 如需采样到固定点数，单独使用 sample_to_target() 函数


def sample_to_target(points, target_num=2048):
    """将点云采样到固定点数（随机采样 + 补足）"""
    if len(points) > target_num:
        indices = np.random.choice(len(points), target_num, replace=False)
        return points[indices]
    elif len(points) < target_num:
        remaining = target_num - len(points)
        indices = np.random.choice(len(points), remaining, replace=True)
        return np.vstack([points, points[indices]])
    return points


def run_point_completion(points, target_points=None):
    """
    使用 PoinTr 模型进行点云补全
    """
    model = load_model()
    if model is None:
        print("错误: PoinTr 模型不可用")
        return None

    device = get_device()

    # 统一归一化（包围盒中心 + 半宽）
    normalized, center, scale = preprocess_points(points)
    # 采样到模型需要的 2048 点
    model_input = sample_to_target(normalized, target_num=2048)

    print("\n预处理完成:")
    print(f"  - 输入点数: {len(points)}")
    print(f"  - 归一化后点数: {len(normalized)}")
    print(f"  - 模型输入点数: {len(model_input)}")

    input_tensor = torch.FloatTensor(model_input).unsqueeze(0).to(device)

    print("\n正在进行点云补全推理...")

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
                    print(f"numpy() 失败，尝试 tolist(): {np_error}")
                    completed = fine_points.squeeze(0).cpu().tolist()
                    completed = np.array(completed)
            else:
                try:
                    completed = ret.squeeze(0).cpu().numpy()
                except Exception as np_error:
                    print(f"numpy() 失败，尝试 tolist(): {np_error}")
                    completed = ret.squeeze(0).cpu().tolist()
                    completed = np.array(completed)

            completed = postprocess_points(completed, scale, center)

            print("\n点云补全成功!")
            print(f"  - 输出点数: {len(completed)}")

            if target_points and len(completed) != target_points:
                completed = resample_points(completed, target_points)

            return completed

        except Exception as e:
            print(f"\n错误: PoinTr 推理出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def resample_points(points, target_num):
    """重采样到目标点数"""
    current_num = len(points)

    if current_num == target_num:
        return points

    if current_num > target_num:
        indices = np.random.choice(current_num, target_num, replace=False)
        return points[indices]
    else:
        indices = np.random.choice(current_num, target_num, replace=True)
        return points[indices]


def rule_based_completion(points, target_points=15000):
    """基于规则的补全算法（备用方案）"""
    print("\n使用规则补全方法...")

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

    num_copy = min(needed_points // 2, len(points))
    indices = np.random.choice(len(points), num_copy, replace=True)
    copied_points = points[indices] + np.random.randn(num_copy, 3) * (scale.mean() * 0.02)
    new_points.extend(copied_points)

    remaining = needed_points - num_copy
    for _ in range(remaining):
        t = np.random.rand(3)
        new_point = min_bound + t * scale

        if len(points) > 0:
            dists = np.linalg.norm(points - new_point, axis=1)
            nearest_dist = np.min(dists)
            if nearest_dist > scale.mean() * 0.3:
                nearest_point = points[np.argmin(dists)]
                new_point = nearest_point + (new_point - nearest_point) * 0.3

        new_points.append(new_point)

    completed = np.vstack([points, np.array(new_points)])

    if len(completed) < target_points:
        indices = np.random.choice(len(completed), target_points - len(completed), replace=True)
        completed = np.vstack([completed, completed[indices]])
    elif len(completed) > target_points:
        indices = np.random.choice(len(completed), target_points, replace=False)
        completed = completed[indices]

    return completed


def simple_point_cloud_completion(points, target_points=15000):
    """
    点云补全主函数
    """
    try:
        result = run_point_completion(points, target_points)
        if result is not None and len(result) > 0:
            return result
    except Exception as e:
        print(f"错误: PoinTr 调用失败: {e}")

    return rule_based_completion(points, target_points)


def enhance_point_cloud(points):
    """增强点云质量（去噪）"""
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    try:
        pcd = pcd.filter_smooth_simple(number_of_iterations=3)
    except AttributeError:
        try:
            pcd = pcd.filter_smooth_taubin(number_of_iterations=3)
        except:
            print("跳过平滑处理")

    return np.asarray(pcd.points)

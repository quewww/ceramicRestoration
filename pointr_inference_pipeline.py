import os
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
import sys

pointr_root = os.path.join(os.path.dirname(__file__), 'PoinTr')
sys.path.insert(0, pointr_root)

from tools.builder import model_builder, load_model
from utils.config import cfg_from_yaml_file


# ==================== 阶段1：输入预处理模块 ====================

def load_point_cloud(filepath):
    """加载PLY点云文件"""
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"无法加载点云文件: {filepath}")
    points = np.asarray(pcd.points, dtype=np.float32)
    print(f"[预处理] 原始点云加载完成，点数: {points.shape[0]}")
    print(f"[预处理] 原始点云坐标范围: x=[{points[:,0].min():.4f}, {points[:,0].max():.4f}], "
          f"y=[{points[:,1].min():.4f}, {points[:,1].max():.4f}], z=[{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
    return pcd, points


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """统计滤波去除离群点（SOR）"""
    print(f"[预处理] 执行统计滤波SOR...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    filtered_points = np.asarray(filtered_pcd.points, dtype=np.float32)
    print(f"[预处理] SOR滤波完成，剩余点数: {filtered_points.shape[0]}")
    return filtered_pcd, filtered_points


def remove_nan_inf(points):
    """清除NaN和Inf无效坐标"""
    mask = np.isfinite(points).all(axis=1)
    cleaned_points = points[mask]
    removed_count = len(points) - len(cleaned_points)
    if removed_count > 0:
        print(f"[预处理] 清除无效坐标 {removed_count} 个")
    return cleaned_points


def fps_sample(points, num_samples=2048):
    """标准FPS最远点采样（固定采样至目标点数，禁止随机采样和重复采样）"""
    print(f"[预处理] 执行FPS采样，目标点数: {num_samples}")
    
    if len(points) < num_samples:
        raise ValueError(f"[错误] FPS采样失败：可用点数({len(points)}点) < 目标采样点数({num_samples}点)，禁止使用重复采样方式补足")
    
    points_tensor = torch.from_numpy(points).unsqueeze(0).contiguous()
    
    try:
        from pointnet2_ops import pointnet2_utils
        fps_idx = pointnet2_utils.furthest_point_sample(points_tensor, num_samples)
        sampled_points = pointnet2_utils.gather_operation(
            points_tensor.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).squeeze(0).numpy()
    except ImportError:
        print(f"[警告] pointnet2_ops不可用，使用纯Python FPS实现")
        sampled_points = fps_python(points, num_samples)
    
    print(f"[预处理] FPS采样完成，采样点数: {sampled_points.shape[0]}")
    return sampled_points


def fps_python(points, num_samples):
    """纯Python实现的FPS采样（确定性实现，保证可复现）"""
    n_points = len(points)
    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_indices[0] = 0
    dists = np.zeros(n_points)
    
    for i in range(1, num_samples):
        last_idx = sampled_indices[i-1]
        dist = np.linalg.norm(points - points[last_idx], axis=1)
        dists = np.maximum(dists, dist)
        sampled_indices[i] = np.argmax(dists)
    
    return points[sampled_indices]


def normalize_points(points):
    """包围盒中心化 + 归一化到[-1,1]（保存center和scale用于逆归一化）"""
    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)
    
    center = (min_coord + max_coord) / 2.0
    translated = points - center
    
    extent = max_coord - min_coord
    scale = np.max(extent) / 2.0
    if scale <= 0:
        scale = 1.0
    normalized = translated / scale
    
    print(f"[预处理] 归一化完成，中心点: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"[预处理] 缩放因子: {scale:.4f}")
    print(f"[预处理] 归一后坐标范围: x=[{normalized[:,0].min():.4f}, {normalized[:,0].max():.4f}], "
          f"y=[{normalized[:,1].min():.4f}, {normalized[:,1].max():.4f}], z=[{normalized[:,2].min():.4f}, {normalized[:,2].max():.4f}]")
    
    return normalized, center, scale


# ==================== 阶段2：模型推理模块 ====================

def denormalize_points(points_norm, center, scale):
    """逆归一化：恢复原始坐标（防止点云炸开的关键步骤）"""
    denormalized = points_norm * scale + center
    print(f"[推理] 逆归一化完成，坐标范围: x=[{denormalized[:,0].min():.4f}, {denormalized[:,0].max():.4f}], "
          f"y=[{denormalized[:,1].min():.4f}, {denormalized[:,1].max():.4f}], z=[{denormalized[:,2].min():.4f}, {denormalized[:,2].max():.4f}]")
    return denormalized


def init_pointr_model(config_path, checkpoint_path, device='cuda'):
    """初始化PoinTr模型（严格对齐C55训练配置）"""
    print(f"[推理] 加载配置文件: {config_path}")
    config = cfg_from_yaml_file(config_path)
    
    num_pred = config.model.num_pred
    num_query = config.model.num_query
    print(f"[推理] 初始化模型，num_pred={num_pred}, num_query={num_query}")
    print(f"[推理] 模型输入点数: 2048, 模型生成点数: {num_pred}")
    
    model = model_builder(config.model)
    
    print(f"[推理] 加载权重: {checkpoint_path}")
    load_model(model, checkpoint_path)
    
    model = model.to(device)
    model.eval()
    print(f"[推理] 模型加载完成，设备: {device}")
    
    return model, config


def run_inference(model, points_norm, device='cuda'):
    """执行模型推理（仅取fine分支输出）"""
    points_tensor = torch.from_numpy(points_norm).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ret = model(points_tensor)
    
    coarse_output = ret[0].squeeze(0).detach().cpu().numpy()
    fine_output = ret[1].squeeze(0).detach().cpu().numpy()
    
    print(f"[推理] 推理完成，coarse输出点数: {coarse_output.shape[0]}, fine输出点数: {fine_output.shape[0]}")
    print(f"[推理] 其中模型生成点数: {fine_output.shape[0] - points_norm.shape[0]}")
    
    return coarse_output, fine_output


# ==================== 阶段3：后处理模块 ====================

def post_process_denoising(points, nb_neighbors=30, std_ratio=3.0):
    """统计滤波去噪（去除推理生成的漂浮离群点）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"[后处理] 执行SOR滤波去除漂浮离群点...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    filtered_points = np.asarray(filtered_pcd.points, dtype=np.float32)
    
    removed = len(points) - len(filtered_points)
    print(f"[后处理] SOR滤波完成，去除 {removed} 个离群点，剩余点数: {filtered_points.shape[0]}")
    
    return filtered_points


def dbscan_keep_largest_component(points, eps=0.25, min_samples=3):
    """DBSCAN聚类，仅保留最大连通域（删除零散碎片）"""
    print(f"[后处理] 执行DBSCAN聚类分析，eps={eps}, min_samples={min_samples}...")
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    print(f"[后处理] DBSCAN完成，检测到 {n_clusters} 个连通组件")
    
    if n_clusters == 0:
        print(f"[警告] 未检测到有效聚类，返回原始点云")
        return points
    
    if n_clusters == 1:
        print(f"[后处理] 只有一个连通组件，无需过滤")
        return points
    
    max_cluster_size = 0
    max_cluster_label = -1
    
    for label in unique_labels:
        if label == -1:
            continue
        cluster_size = np.sum(labels == label)
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            max_cluster_label = label
    
    mask = labels == max_cluster_label
    filtered_points = points[mask]
    
    removed = len(points) - len(filtered_points)
    print(f"[后处理] 保留最大连通组件，去除 {removed} 个孤立点，剩余点数: {filtered_points.shape[0]}")
    
    return filtered_points


def fps_resample(points, target_points=6144):
    """均匀FPS重采样（改善凹凸杂乱观感，获得分布均匀的点云）"""
    print(f"[后处理] 执行均匀FPS重采样，目标点数: {target_points}")
    
    if len(points) <= target_points:
        repeats = (target_points // len(points)) + 1
        repeated_points = np.tile(points, (repeats, 1))
        resampled_points = repeated_points[:target_points]
    else:
        resampled_points = fps_sample(points, num_samples=target_points)
    
    print(f"[后处理] FPS重采样完成，采样点数: {resampled_points.shape[0]}")
    return resampled_points


# ==================== 阶段4：输出保存模块 ====================

def save_point_cloud(points, filepath):
    """保存PLY点云文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"[后处理] 点云保存完成: {filepath}")


def point_cloud_completion_pipeline(
    input_ply_path,
    output_ply_path,
    config_path='PoinTr/cfgs/ShapeNet55_models/PoinTr.yaml',
    checkpoint_path='PoinTr/ckpts/pointr_training_from_scratch_c55_best.pth',
    device='cuda'
):
    """
    PoinTr点云补全完整推理流水线（重构版）
    
    设计原则：
    1. PoinTr是全局重建模型，网络输出本身就是完整物体，不叠加原始残缺点云
    2. 严格对齐C55训练配置：输入2048点，num_pred=6144，num_query=96
    3. 完整的预处理→推理→后处理流程，确保坐标正确还原
    
    Args:
        input_ply_path: 输入残缺点云PLY文件路径
        output_ply_path: 输出补全后点云PLY文件路径
        config_path: PoinTr配置文件路径
        checkpoint_path: 预训练权重文件路径
        device: 推理设备 ('cuda' 或 'cpu')
    """
    print("=" * 60)
    print("PoinTr点云补全推理流水线启动（重构版）")
    print("=" * 60)
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("[警告] CUDA不可用，切换到CPU模式")
        device = 'cpu'
    
    # ---------------------- 阶段1：输入预处理模块 ----------------------
    print("\n[阶段1] 输入预处理")
    print("-" * 40)
    
    pcd, raw_points = load_point_cloud(input_ply_path)
    
    if len(raw_points) < 2048:
        raise ValueError(f"[错误] 原始点云点数不足({len(raw_points)}点)，无法进行PoinTr推理。PoinTr模型需要至少2048点才能正常工作，禁止使用重复采样方式补足。")
    
    pcd_filtered, sor_points = remove_outliers(pcd)
    
    clean_points = remove_nan_inf(sor_points)
    
    if len(clean_points) < 2048:
        raise ValueError(f"[错误] SOR滤波后点数不足({len(clean_points)}点)，无法进行PoinTr推理。请检查输入点云质量。")
    
    sampled_points = fps_sample(clean_points, num_samples=2048)
    
    normalized_points, center, scale = normalize_points(sampled_points)
    
    # ---------------------- 阶段2：模型推理模块 ----------------------
    print("\n[阶段2] 模型推理")
    print("-" * 40)
    
    model, config = init_pointr_model(config_path, checkpoint_path, device)
    
    coarse_output, fine_output = run_inference(model, normalized_points, device)
    
    denormalized_points = denormalize_points(fine_output, center, scale)
    
    # ---------------------- 阶段3：后处理模块（仅作用于模型输出） ----------------------
    print("\n[阶段3] 后处理")
    print("-" * 40)
    
    print("[后处理] 注意：后处理仅作用于模型输出点云，不混入原始残缺点云")
    print(f"[后处理] 模型原生输出点数: {denormalized_points.shape[0]}")
    print(f"[后处理] 配置num_pred={config.model.num_pred}，num_query={config.model.num_query}")
    print(f"[后处理] 计算: fold_step = sqrt(num_pred/num_query) = sqrt({config.model.num_pred}/{config.model.num_query}) = {int(pow(config.model.num_pred//config.model.num_query, 0.5) + 0.5)}")
    print("[后处理] 重要：num_pred是训练时固定的网络输出维度（FoldingNet生成新点数量），修改它会导致权重失效、形状崩坏")
    print("[后处理] 训练时fine输出 = num_pred + 输入点数 = 6144 + 2048 = 8192点，与权重严格匹配")
    print("[后处理] 点数精简只能放在后处理阶段，禁止在模型推理阶段限制输出点数")
    
    denoised_points = post_process_denoising(denormalized_points)
    
    connected_points = dbscan_keep_largest_component(denoised_points)
    
    target_points = 6144
    if len(connected_points) > target_points:
        print(f"[后处理] 执行FPS重采样，从 {len(connected_points)} 点精简到 {target_points} 点")
        final_points = fps_resample(connected_points, target_points=target_points)
    else:
        print(f"[后处理] 点数不足 {target_points}，跳过FPS重采样")
        final_points = connected_points
    
    final_points = remove_nan_inf(final_points)
    
    # ---------------------- 阶段4：输出保存模块 ----------------------
    print("\n[阶段4] 保存结果")
    print("-" * 40)
    
    save_point_cloud(final_points, output_ply_path)
    
    print("\n" + "=" * 60)
    print("PoinTr点云补全推理流水线完成")
    print(f"原始点数: {raw_points.shape[0]}")
    print(f"模型输出点数: {fine_output.shape[0]}")
    print(f"补全后最终点数: {final_points.shape[0]}")
    print("=" * 60)
    
    return final_points


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PoinTr点云补全推理（重构版）')
    parser.add_argument('input_ply', type=str, help='输入残缺点云PLY文件路径')
    parser.add_argument('output_ply', type=str, help='输出补全后点云PLY文件路径')
    parser.add_argument('--config', type=str, default='PoinTr/cfgs/ShapeNet55_models/PoinTr.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='PoinTr/ckpts/pointr_training_from_scratch_c55_best.pth', help='权重文件路径')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='推理设备')
    
    args = parser.parse_args()
    
    print(f"\n调用示例: python pointr_inference_pipeline.py {args.input_ply} {args.output_ply} --device {args.device}")
    print("=" * 60 + "\n")
    
    point_cloud_completion_pipeline(
        input_ply_path=args.input_ply,
        output_ply_path=args.output_ply,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

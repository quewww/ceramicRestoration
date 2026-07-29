#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PF-Net (Point-to-Point Network) 独立推理脚本
基于 CVPR2020 PF-Net / PoinTr 生成器网络结构实现
输入：残缺点云 PLY 文件
输出：完整补全点云 PLY 文件

使用方法：
    from pfnet_inference import pfnet_completion
    pfnet_completion("input.ply", "output.ply", device="cuda")

或命令行：
    python pfnet_inference.py input.ply output.ply
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

# PCA 调试开关：设为 False 可跳过 PCA 对齐，用于排查是否为 PCA 导致坐标漂移
PCA_ENABLED = True


# ==================== 特征提取器（3D→512维特征） ====================

class PoinTrFeatureExtractor(nn.Module):
    """将原始3D点坐标投影为512维特征（用于Generator输入）"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 512, kernel_size=1)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) 原始点坐标
        Returns:
            features: (B, 512, N) 特征图
        """
        x = xyz.permute(0, 2, 1)  # (B, 3, N)
        x = self.conv(x)           # (B, 512, N)
        return x


# ==================== PoinTr Generator 网络结构 ====================

class LatentConvLayer(nn.Module):
    """Latent特征提取器中的单个Convlayers块
    同时返回中间特征(128通道)和最终特征(1024通道)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) 输入
        Returns:
            tuple: (intermediate, final)
                intermediate: (B, 128, H, W) conv3后特征
                final: (B, 1024, H, W) conv6后特征
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        intermediate = x  # (B, 128, H, W)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        final = x  # (B, 1024, H, W)
        return intermediate, final


class PoinTrGenerator(nn.Module):
    """
    PoinTr 生成器 (Generator)
    完全匹配 point_netG90.pth checkpoint 结构
    键名格式: Convlayers1.0.conv1.weight (Sequential包装产生.0.索引)
    """
    def __init__(self):
        super().__init__()

        # Stem 卷积层（处理512维特征）
        self.conv1_1 = nn.Conv1d(512, 512, kernel_size=1)
        self.conv1_2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv1_3 = nn.Conv1d(256, 12, kernel_size=1)
        self.conv2_1 = nn.Conv1d(128, 6, kernel_size=1)

        # Latent 特征提取器 (Sequential包装使键名带.0.索引)
        self.Convlayers1 = nn.Sequential(LatentConvLayer())
        self.Convlayers2 = nn.Sequential(LatentConvLayer())
        self.Convlayers3 = nn.Sequential(LatentConvLayer())
        self.conv1 = nn.Conv1d(3, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)

        # FC 解码器分支
        self.fc1 = nn.Linear(1920, 1024)
        self.fc1_1 = nn.Linear(1024, 65536)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2_1 = nn.Linear(512, 8192)

        self.fc3 = nn.Linear(512, 256)
        self.fc3_1 = nn.Linear(256, 192)

    def forward(self, features):
        """
        Args:
            features: (B, 512, N) 输入特征图
        Returns:
            pred_points: (B, num_pred, 3) 预测补全点
        """
        B, C, N = features.shape  # B, 512, 1024

        # ============ Stem 卷积分支 ============
        x1 = F.relu(self.conv1_1(features))   # (B, 512, N)
        x2 = F.relu(self.conv1_2(x1))          # (B, 256, N)
        x3 = F.relu(self.conv1_3(x2))          # (B, 12, N)

        # ============ Latent 特征分支 ============
        # 从512D特征中提取3通道表示供conv1使用
        raw_proxy = features[:, :3, :]         # (B, 3, N)

        # conv1(3→1) + bn1: 生成1通道潜表示
        latent_lp = self.conv1(raw_proxy)      # (B, 1, N)
        # Convlayers中conv1的kernel_size=(1,3)，需要N在宽度维: (B, 1, 1, N)
        latent_lp = latent_lp.unsqueeze(2).permute(0, 1, 3, 2)
        latent_lp = self.bn1(latent_lp)
        latent_lp = F.relu(latent_lp)         # (B, 1, 1, N)

        # Convlayers处理: 三个并行块，返回 (intermediate, final) 元组
        lat1_int, lat1_fin = self.Convlayers1(latent_lp)  # (B,128,1,N), (B,1024,1,N)
        lat2_int, lat2_fin = self.Convlayers2(latent_lp)  # (B,128,1,N), (B,1024,1,N)
        lat3_int, lat3_fin = self.Convlayers3(latent_lp)  # (B,128,1,N), (B,1024,1,N)

        # 融合三个分支的中间特征和最终特征
        lat_int_fused = (lat1_int + lat2_int + lat3_int) / 3.0  # (B, 128, 1, N)
        lat_fin_fused = (lat1_fin + lat2_fin + lat3_fin) / 3.0  # (B, 1024, 1, N)

        # ============ 全局平均池化得到固定维度向量 ============
        # stem: 对conv1_1和conv1_2输出做全局池化
        x1_pooled = x1.mean(dim=-1)   # (B, 512)
        x2_pooled = x2.mean(dim=-1)   # (B, 256)

        # Convlayers: 对中间(128ch)和最终(1024ch)特征做全局池化
        # 输入形状: (B, C, 1, N) → 对后两个维度求均值
        lat_int_pooled = lat_int_fused.mean(dim=(2, 3))   # (B, 128)
        lat_fin_pooled = lat_fin_fused.mean(dim=(2, 3))   # (B, 1024)

        # ============ 拼接得到1920维特征 ============
        # 512 + 256 + 128 + 1024 = 1920
        combined = torch.cat([x1_pooled, x2_pooled, lat_int_pooled, lat_fin_pooled], dim=-1)
        # combined: (B, 1920)

        # ============ FC 解码分支 ============
        fc1_out = F.relu(self.fc1(combined))         # (B, 1024)
        fc1_pred = self.fc1_1(fc1_out)               # (B, 65536)

        fc2_out = F.relu(self.fc2(fc1_out))          # (B, 512)
        fc2_pred = self.fc2_1(fc2_out)               # (B, 8192)

        fc3_out = F.relu(self.fc3(fc2_out))          # (B, 256)
        fc3_pred = self.fc3_1(fc3_out)               # (B, 192) → 64 points

        # ============ 输出组合 ============
        fine_points = fc3_pred.reshape(B, -1, 3)    # (B, 64, 3)

        # fc2_pred: 8192 → 取前8190 = 2730 * 3 点
        medium_points = fc2_pred[:, :8190].reshape(B, -1, 3)  # (B, 2730, 3)

        all_points = torch.cat([fine_points, medium_points], dim=1)  # (B, 2794, 3)

        return all_points


# ==================== 迭代最远点采样 (IFPS) ====================

def iterative_farthest_point_sampling(points, num_samples=1024, num_iters=10):
    """
    迭代最远点采样 (IFPS)

    原理：在每次迭代中，使用FPS采样得到一个子集，然后对剩余点重复此过程，
    确保采样点覆盖整个点云的不同区域。

    Args:
        points: (N, 3) numpy数组
        num_samples: 目标采样点数
        num_iters: 迭代次数

    Returns:
        sampled_points: (num_samples, 3) numpy数组
    """
    n_points = len(points)

    if n_points <= num_samples:
        print(f"[IFPS] 警告: 点数({n_points}) <= 目标点数({num_samples})，直接返回")
        return points

    subset_size = min(num_samples // num_iters, n_points // num_iters)
    if subset_size < 1:
        subset_size = 1

    sampled_indices = set()
    remaining_indices = set(range(n_points))

    for iteration in range(num_iters):
        if len(sampled_indices) >= num_samples:
            break

        remaining_list = list(remaining_indices)
        remaining_points = points[remaining_list]

        need = num_samples - len(sampled_indices)
        current_sample = min(subset_size, need, len(remaining_list))

        if current_sample <= 0:
            break

        fps_idx = _fps_python(remaining_points, current_sample)

        for idx in fps_idx:
            original_idx = remaining_list[idx]
            if original_idx not in sampled_indices:
                sampled_indices.add(original_idx)
                remaining_indices.discard(original_idx)

    if len(sampled_indices) < num_samples:
        remaining_list = list(remaining_indices)
        remaining_points = points[remaining_list]
        need = num_samples - len(sampled_indices)

        if len(remaining_list) >= need:
            fps_idx = _fps_python(remaining_points, need)
            for idx in fps_idx:
                sampled_indices.add(remaining_list[idx])
        else:
            for idx in remaining_list[:need]:
                sampled_indices.add(idx)

    final_indices = list(sampled_indices)[:num_samples]
    return points[final_indices]


def _fps_python(points, num_samples):
    """纯Python实现的FPS采样"""
    n_points = len(points)
    if n_points <= num_samples:
        return list(range(n_points))

    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_indices[0] = 0
    dists = np.full(n_points, np.inf)

    for i in range(1, num_samples):
        last_idx = sampled_indices[i-1]
        dist = np.sum((points - points[last_idx]) ** 2, axis=1)
        dists = np.minimum(dists, dist)
        sampled_indices[i] = np.argmax(dists)

    return list(sampled_indices)


# ==================== 预处理流水线 ====================

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


def remove_outliers_sor(pcd, nb_neighbors=20, std_ratio=2.0):
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


def ifps_sample(points, num_samples=1024):
    """迭代最远点采样 (IFPS)"""
    print(f"[预处理] 执行IFPS迭代最远点采样，目标点数: {num_samples}")

    if len(points) < num_samples:
        raise ValueError(f"[错误] IFPS采样失败：可用点数({len(points)}点) < 目标采样点数({num_samples}点)")

    sampled_points = iterative_farthest_point_sampling(points, num_samples, num_iters=10)
    print(f"[预处理] IFPS采样完成，采样点数: {sampled_points.shape[0]}")
    return sampled_points


def compute_normal_params(points):
    """从点云计算归一化参数（center, scale），供后续统一使用，输出float32"""
    min_coord = np.min(points, axis=0).astype(np.float32)
    max_coord = np.max(points, axis=0).astype(np.float32)
    center = ((min_coord + max_coord) / 2.0).astype(np.float32)
    extent = (max_coord - min_coord).astype(np.float32)
    scale = np.float32(np.max(extent) / 2.0)
    if scale <= 0:
        scale = np.float32(1.0)
    return center.astype(np.float32), np.float32(scale)


def apply_normalize(points, center, scale):
    """使用预计算的center/scale对任意点云做归一化，输出float32"""
    points = np.asarray(points, dtype=np.float32)
    translated = (points - center).astype(np.float32)
    normalized = (translated / scale).astype(np.float32)
    return normalized


def normalize_points(points):
    """包围盒中心化 + 归一化到[-1,1]（便捷封装）"""
    center, scale = compute_normal_params(points)
    normalized = apply_normalize(points, center, scale)

    print(f"[预处理] 归一化完成，中心点: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"[预处理] 缩放因子: {scale:.4f}")
    print(f"[预处理] 归一后坐标范围: x=[{normalized[:,0].min():.4f}, {normalized[:,0].max():.4f}], "
          f"y=[{normalized[:,1].min():.4f}, {normalized[:,1].max():.4f}], z=[{normalized[:,2].min():.4f}, {normalized[:,2].max():.4f}]")

    return normalized, center, scale


def denormalize_points(normalized_points, center, scale):
    """逆归一化，输出float32"""
    normalized_points = np.asarray(normalized_points, dtype=np.float32)
    result = (normalized_points * scale + center).astype(np.float32)
    return result


# ==================== PCA 标准化（PF-Net 推理专用） ====================

def pca_standardize_points(points):
    """
    PCA 主轴对齐标准化（仅作用于送入模型的采样副本）。

    正向流程：
      ① 计算均值 mean → 中心化 points - mean
      ② 协方差矩阵 → 特征分解 → 旋转矩阵 rot
      ③ 投影：pts_pca = (pts - mean) @ rot
      ④ Y-up 对齐：最大延展轴映射为 Y，旋转矩阵同步调整
      ⑤ Y 轴正向修正：数据 + 旋转矩阵同步翻转（baked into rot）
      ⑥ 尺度约束：最大包围盒延展归一化到 [-1, 1]

    Args:
        points: (N, 3) 已归一化的采样副本点云
    Returns:
        standardized: (N, 3) PCA 标准化后的点云
        params: 逆变换参数字典
    """
    points = np.asarray(points, dtype=np.float32)

    # ① 中心化
    mean = np.mean(points, axis=0).astype(np.float32)
    centered = (points - mean).astype(np.float32)

    # ② PCA 分解
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]  # 3x3，列=特征向量（按特征值降序）

    # ③ 投影到主成分坐标系
    rotated = centered @ eigenvectors  # (N,3) @ (3,3) = (N,3)

    # ④ Y-up 对齐：将最大延展轴映射为 Y
    axis_extents = np.ptp(rotated, axis=0)
    y_idx = int(np.argmax(axis_extents))

    remaining = [i for i in range(3) if i != y_idx]
    if axis_extents[remaining[0]] >= axis_extents[remaining[1]]:
        x_idx, z_idx = remaining[0], remaining[1]
    else:
        x_idx, z_idx = remaining[1], remaining[0]

    perm = [x_idx, y_idx, z_idx]
    rotated = rotated[:, perm]          # 重排数据列
    rot = eigenvectors[:, perm]          # 重排旋转矩阵列 → 新的3x3 rot

    # ⑤ Y 轴正向修正：数据和旋转矩阵同步翻转，使 Y 均值 > 0
    #    翻转 baked into rot，逆变换无需单独处理
    y_mean = np.mean(rotated[:, 1])
    if y_mean < 0:
        rotated[:, 1] *= -1
        rot[:, 1] *= -1

    # ⑥ 尺度约束：最大延展 → 2.0（即 [-1, 1]）
    extents = np.ptp(rotated, axis=0)
    max_extent = float(np.max(extents))
    scale = max_extent / 2.0 if max_extent > 1e-8 else 1.0
    standardized = (rotated / scale).astype(np.float32)

    # ---- 详细日志 ----
    print(f"[PCA正向] 均值mean = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"[PCA正向] 旋转矩阵 rot (3x3):")
    for row in range(3):
        print(f"  [{rot[row,0]: .6f}  {rot[row,1]: .6f}  {rot[row,2]: .6f}]")
    print(f"[PCA正向] 尺度scale = {scale:.6f}")
    print(f"[PCA正向] 标准化后包围盒: x=[{standardized[:,0].min():.4f}, {standardized[:,0].max():.4f}], "
          f"y=[{standardized[:,1].min():.4f}, {standardized[:,1].max():.4f}], "
          f"z=[{standardized[:,2].min():.4f}, {standardized[:,2].max():.4f}]")

    return standardized, {
        'mean': mean,
        'rot': rot.astype(np.float32),
        'scale': np.float32(scale)
    }


def inverse_pca_standardize_points(pred_pca, params):
    """
    逆 PCA 标准化：将模型预测点还原到原始归一化坐标系。

    逆变换流程（严格反向）：
      ① 反尺度：pred = pred_pca * scale
      ② 反旋转：pred = pred @ rot.T  （rot 已 baked Y-up + Y-flip）
      ③ 反平移：pred = pred + mean

    Args:
        pred_pca: (M, 3) 模型在 PCA 空间的预测点
        params: pca_standardize_points 返回的参数字典
    Returns:
        pred_pts: (M, 3) 还原到原始坐标系的预测点
    """
    pred_pca = np.asarray(pred_pca, dtype=np.float32)

    # 逆变换前：打印包围盒
    print(f"[PCA逆向] 逆变换前 pred_pca 包围盒: x=[{pred_pca[:,0].min():.4f}, {pred_pca[:,0].max():.4f}], "
          f"y=[{pred_pca[:,1].min():.4f}, {pred_pca[:,1].max():.4f}], "
          f"z=[{pred_pca[:,2].min():.4f}, {pred_pca[:,2].max():.4f}]")

    # ① 反尺度
    pts = pred_pca * params['scale']

    # ② 反旋转：pred = pred @ rot.T （Y-flip 已 baked into rot，无需单独处理）
    pts = pts @ params['rot'].T

    # ③ 反平移
    pred_pts = (pts + params['mean']).astype(np.float32)

    # 逆变换后：打印包围盒
    print(f"[PCA逆向] 逆变换后 pred_pts 包围盒: x=[{pred_pts[:,0].min():.4f}, {pred_pts[:,0].max():.4f}], "
          f"y=[{pred_pts[:,1].min():.4f}, {pred_pts[:,1].max():.4f}], "
          f"z=[{pred_pts[:,2].min():.4f}, {pred_pts[:,2].max():.4f}]")

    return pred_pts


# ==================== 后处理 ====================

def post_process_denoising(points, nb_neighbors=20, std_ratio=2.0):
    """SOR统计滤波"""
    print(f"[后处理] 执行SOR滤波去除漂浮离群点...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    denoised_pcd = pcd.select_by_index(ind)
    denoised_points = np.asarray(denoised_pcd.points, dtype=np.float32)

    removed = len(points) - len(denoised_points)
    print(f"[后处理] SOR滤波完成，去除 {removed} 个离群点，剩余点数: {len(denoised_points)}")
    return denoised_points


def dbscan_keep_largest_component(points, eps=0.25, min_samples=3):
    """DBSCAN聚类，仅保留最大连通域"""
    print(f"[后处理] 执行DBSCAN聚类分析，eps={eps}, min_samples={min_samples}...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        print(f"[后处理] 警告: 未检测到有效聚类，返回原始点云")
        return points

    label_counts = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        label_counts[label] = count

    max_label = max(label_counts, key=label_counts.get)
    max_cloud = pcd.select_by_index(np.where(labels == max_label)[0])
    max_points = np.asarray(max_cloud.points, dtype=np.float32)

    print(f"[后处理] DBSCAN完成，检测到 {len(unique_labels)} 个连通组件")
    print(f"[后处理] 保留最大组件，点数: {len(max_points)}")

    return max_points


# ==================== 模型初始化与推理 ====================

def load_pfnet_generator(ckpt_path, device='cuda'):
    """加载PF-Net/PoinTr生成器权重"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    print(f"[推理] 加载PF-Net生成器权重: {ckpt_path}")

    # 创建Generator模型（严格匹配checkpoint结构）
    model = PoinTrGenerator()

    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 判断加载结果是否包含"state_dict"键
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 存在外层字典，取 ckpt["state_dict"]
        print(f"[推理] 检测到外层state_dict，提取后包含 {len(checkpoint['state_dict'])} 个键")
        state_dict = checkpoint['state_dict']
    else:
        # 权重文件本身就是state_dict
        if isinstance(checkpoint, dict):
            print(f"[推理] 权重文件直接为state_dict，共 {len(checkpoint)} 个键")
        state_dict = checkpoint

    # 去除module.前缀（DataParallel训练会添加此前缀）
    # 同时去除latentfeature.子模块前缀（checkpoint中Convlayers和conv1/bn1的容器名）
    cleaned_state_dict = {}
    module_prefix_count = 0
    latent_prefix_count = 0
    for key, value in state_dict.items():
        cleaned_key = key
        if cleaned_key.startswith('module.'):
            cleaned_key = cleaned_key[len('module.'):]
            module_prefix_count += 1
        if cleaned_key.startswith('latentfeature.'):
            cleaned_key = cleaned_key[len('latentfeature.'):]
            latent_prefix_count += 1
        cleaned_state_dict[cleaned_key] = value

    if module_prefix_count > 0:
        print(f"[推理] 去除module.前缀，共处理 {module_prefix_count} 个键")
    if latent_prefix_count > 0:
        print(f"[推理] 去除latentfeature.前缀，共处理 {latent_prefix_count} 个键")

    # 严格加载（不再使用strict=False）
    model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"[推理] 【PF-Net权重完整匹配加载成功】")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[推理] 模型参数量: {total_params:,}")
    print(f"[推理] 加载完成，设备: {device}")

    # float32 预热：防止首次推理因GPU精度缓存报错
    print(f"[推理] 执行float32预热...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 512, 1024, device=device, dtype=torch.float32)
        _ = model(dummy_input)
    print(f"[推理] float32预热完成")

    return model


def run_pfnet_inference(feature_extractor, generator, points, device='cuda'):
    """运行PF-Net推理"""
    print(f"[推理] 开始PF-Net推理...")

    # ① 强制 numpy → float32，匹配模型权重精度
    points = np.asarray(points, dtype=np.float32)
    # ② 指定 dtype=torch.float32 再 to(device)
    input_tensor = torch.from_numpy(points).unsqueeze(0).to(device=device, dtype=torch.float32)
    print(f"[推理] 【输入张量已转为float32，匹配模型权重精度】")

    with torch.no_grad():
        # 特征提取: (B, N, 3) → (B, 512, N)
        features = feature_extractor(input_tensor)
        # Generator推理: (B, 512, N) → (B, num_pred, 3)
        pred_points = generator(features)

    # ③ 预测输出转回float32再转numpy
    pred_np = pred_points.squeeze(0).detach().cpu().float().numpy().astype(np.float32)
    print(f"[推理] 推理完成，预测点数: {len(pred_np)}, 输出dtype: {pred_np.dtype}")

    return pred_np


# ==================== 主推理函数 ====================

def pfnet_completion(input_ply_path, output_ply_path, device="cuda"):
    """
    PF-Net点云补全推理主函数（工业落地版）

    流程：
      ① 读取完整原始稠密点云 original_pcd（全程留存，不丢弃）
      ② 复制副本 sample_pcd → IFPS下采样至1024点 → 仅作模型推理输入
      ③ PF-Net推理仅输出缺失缺口预测点 pred_points
      ④ 最终点云 = original_pcd（原始稠密） + pred_points（AI缺口点）

    Args:
        input_ply_path: 输入残缺点云PLY文件路径
        output_ply_path: 输出补全点云PLY文件路径
        device: 推理设备 ('cuda' 或 'cpu')

    Returns:
        completed_points: (N, 3) numpy数组，补全后的完整点云
    """
    print("=" * 60)
    print("PF-Net点云补全推理流水线启动")
    print("=" * 60)

    if not torch.cuda.is_available() and device == 'cuda':
        print("[警告] CUDA不可用，切换到CPU模式")
        device = 'cpu'

    start_time = time.time()

    try:
        # ==================== 阶段1：输入预处理 ====================
        print("\n[阶段1] 输入预处理")
        print("-" * 40)

        # 1.1 加载完整原始稠密点云（全程留存，不丢弃）
        original_pcd, original_points = load_point_cloud(input_ply_path)
        print(f"[预处理] 保留全部原始稠密点用于最终输出，仅采样副本送入模型推理")
        print(f"[预处理] 原始稠密点云: {len(original_points)} 点")

        # 1.2 清除原始点云中的NaN/Inf无效坐标（仅清理，不稀疏化）
        clean_original_points = remove_nan_inf(original_points)
        if len(clean_original_points) != len(original_points):
            print(f"[预处理] 原始稠密点云清除无效坐标后: {len(clean_original_points)} 点")

        # 1.3 基于原始稠密点云计算全局唯一归一化参数
        center, scale = compute_normal_params(clean_original_points)
        print(f"[预处理] 【归一化中心/缩放因子由原始稠密点云计算】")
        print(f"[预处理] 原始稠密点云中心点: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"[预处理] 原始稠密点云缩放因子: {scale:.4f}")

        # 1.4 检查点数是否足够用于IFPS采样
        if len(clean_original_points) < 1024:
            raise ValueError(f"[错误] 原始点云点数不足({len(clean_original_points)}点)，PF-Net需要至少1024点")

        # 1.5 创建采样副本：仅用于模型推理
        # 副本执行SOR滤波 + IFPS下采样至1024点
        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(clean_original_points)

        # 1.6 副本SOR滤波（仅清理模型输入质量）
        sample_pcd_filtered, sample_sor_points = remove_outliers_sor(sample_pcd)
        sample_clean_points = remove_nan_inf(sample_sor_points)

        if len(sample_clean_points) < 1024:
            raise ValueError(f"[错误] 采样副本SOR滤波后点数不足({len(sample_clean_points)}点)，无法进行IFPS采样")

        # 1.7 副本IFPS采样至1024点
        sampled_points = ifps_sample(sample_clean_points, num_samples=1024)

        # 1.8 使用原始稠密点云的center/scale归一化1024采样副本
        normalized_points = apply_normalize(sampled_points, center, scale).astype(np.float32)
        print(f"[预处理] 1024采样副本已使用原始稠密点云参数归一化")
        print(f"[预处理] 归一后坐标范围: x=[{normalized_points[:,0].min():.4f}, {normalized_points[:,0].max():.4f}], "
              f"y=[{normalized_points[:,1].min():.4f}, {normalized_points[:,1].max():.4f}], "
              f"z=[{normalized_points[:,2].min():.4f}, {normalized_points[:,2].max():.4f}]")

        # 1.9 PCA 主轴对齐标准化（仅作用于采样副本，原始稠密点不动）
        if PCA_ENABLED:
            print(f"[预处理] 【PCA标准化：主轴对齐+Y-up+尺度约束，仅作用于采样副本】")
            pca_standardized, pca_params = pca_standardize_points(normalized_points)
            pca_standardized = pca_standardized.astype(np.float32)
        else:
            print(f"[预处理] 【PCA已禁用：跳过主轴对齐，直接使用原始归一化坐标推理】")
            pca_standardized = normalized_points
            pca_params = None

        # ==================== 阶段2：模型推理 ====================
        print("\n[阶段2] 模型推理")
        print("-" * 40)

        # 2.1 加载Generator模型（权重完整匹配加载）
        ckpt_path = 'PoinTr/ckpts/point_netG90.pth'
        generator = load_pfnet_generator(ckpt_path, device)

        # 2.2 创建特征提取器（随机初始化，不在checkpoint中）
        feature_extractor = PoinTrFeatureExtractor().to(device=device, dtype=torch.float32)
        feature_extractor.eval()

        # 2.3 运行推理 → 仅输出缺失缺口预测点（PCA标准化空间）
        pred_pca = run_pfnet_inference(
            feature_extractor, generator, pca_standardized, device
        )

        # 2.4 逆PCA：还原预测缺口点到原始归一化坐标系
        if PCA_ENABLED and pca_params is not None:
            pred_normalized = inverse_pca_standardize_points(pred_pca, pca_params).astype(np.float32)
            print(f"[推理] 逆PCA标准化完成，预测点已还原到原始坐标系")
        else:
            pred_normalized = pred_pca.astype(np.float32)
            print(f"[推理] PCA已禁用，跳过逆变换，预测点直接用于逆归一化")

        # 2.5 逆归一化：还原到原始稠密点坐标系
        pred_missing_points = denormalize_points(pred_normalized, center, scale).astype(np.float32)
        print(f"[推理] 逆归一化完成（仅作用于预测缺口点）")
        print(f"[推理] 预测缺口点坐标范围: x=[{pred_missing_points[:,0].min():.4f}, {pred_missing_points[:,0].max():.4f}], "
              f"y=[{pred_missing_points[:,1].min():.4f}, {pred_missing_points[:,1].max():.4f}], "
              f"z=[{pred_missing_points[:,2].min():.4f}, {pred_missing_points[:,2].max():.4f}]")

        # ==================== 阶段3：融合与后处理 ====================
        print("\n[阶段3] 融合与后处理")
        print("-" * 40)

        # 3.1 融合：原始稠密点云 + AI预测缺口点
        print(f"[融合] 原始稠密点: {len(clean_original_points)} 点")
        print(f"[融合] AI预测缺口点: {len(pred_missing_points)} 点")
        merged_points = np.concatenate([clean_original_points, pred_missing_points], axis=0)
        print(f"[融合] 合并后总点数: {len(merged_points)}")

        # 3.2 SOR滤波：作用于【原始+缺口合并后的完整稠密点】
        denoised_points = post_process_denoising(merged_points)

        # 3.3 PF-Net模式：跳过DBSCAN连通组件筛选
        # （合并后=原始主体+预测缺口，会被误分割为两个组件，导致主体被删除）
        print("[后处理] 【PF-Net模式：跳过连通组件筛选，防止原始主体被删除】")
        final_points = denoised_points
        print(f"[后处理] 跳过DBSCAN连通组件筛选，保留全部 {len(final_points)} 个点")

        # ==================== 阶段4：保存结果 ====================
        print("\n[阶段4] 保存结果")
        print("-" * 40)

        # 4.1 保存PLY文件
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(final_points)
        o3d.io.write_point_cloud(output_ply_path, output_pcd)
        print(f"[保存] 点云保存完成: {output_ply_path}")

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("PF-Net点云补全推理流水线完成")
        print(f"原始稠密点数: {len(original_points)}")
        print(f"IFPS采样点数(模型输入): {len(sampled_points)}")
        print(f"AI预测缺口点数: {len(pred_missing_points)}")
        print(f"融合后总点数: {len(merged_points)}")
        print(f"SOR后处理最终点数: {len(final_points)}")
        print(f"总耗时: {elapsed_time:.2f}秒")
        print("=" * 60)

        return final_points

    except Exception as e:
        print(f"\n[错误] PF-Net推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==================== 命令行接口 ====================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("使用方法: python pfnet_inference.py <input.ply> <output.ply> [--device cuda|cpu]")
        print("示例: python pfnet_inference.py input.ply output.ply --device cuda")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    device = "cuda"
    if "--device" in sys.argv:
        device_idx = sys.argv.index("--device")
        if device_idx + 1 < len(sys.argv):
            device = sys.argv[device_idx + 1]

    if not os.path.exists(input_path):
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)

    try:
        result = pfnet_completion(input_path, output_path, device=device)
        print(f"\n✅ 推理完成，输出点数: {len(result)}")
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        sys.exit(1)
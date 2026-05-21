import numpy as np
import open3d as o3d
import os


def read_point_cloud(filepath):
    """读取点云文件，支持 .ply, .xyz, .txt, .pcd 格式"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.ply':
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
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
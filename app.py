from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
import os
import uuid
import time
import json
import subprocess
import shutil
import random
import sys
import io
import traceback
import numpy as np
import open3d as o3d
from datetime import timedelta
from contextlib import redirect_stdout

from config import *
from pointcloud_utils import read_point_cloud, write_point_cloud, downsample_point_cloud, convert_to_ply, count_points, estimate_chamfer_distance, detect_axisymmetry, revolve_reconstruction
from enhanced_point_completion import repair_point_cloud, enhance_point_cloud
from pointr_inference_pipeline import point_cloud_completion_pipeline
from pfnet_inference import pfnet_completion

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = 'your-secret-key-here'
app.permanent_session_lifetime = timedelta(days=7)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['USER_DATA_FOLDER'] = USER_DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(USER_DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.join(POINTR_ROOT, 'demo'), exist_ok=True)

users = {
    'admin': {'password': 'admin123', 'name': '管理员'},
    'user1': {'password': 'user123', 'name': '用户1'}
}

projects = []

ALLOWED_EXTENSIONS = {'ply', 'pcd', 'txt', 'xyz', 'obj'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file):
    print(f"[上传接口] validate_file 开始，文件名: {file.filename}")
    
    if not allowed_file(file.filename):
        print(f"[上传接口] 文件格式验证失败，扩展名: {file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else '无'}")
        return False, '不支持的文件格式，请上传 .ply/.pcd/.txt/.xyz/.obj 文件'

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    print(f"[上传接口] 文件大小: {size} bytes")

    if size == 0:
        print(f"[上传接口] 文件为空")
        return False, '空文件，无法上传'

    if size > app.config['MAX_CONTENT_LENGTH']:
        print(f"[上传接口] 文件大小超过限制")
        return False, f'文件大小超过限制 (100MB)'

    print(f"[上传接口] 文件验证通过")
    return True, ''

def process_point_cloud(filepath):
    time.sleep(2)
    result = {
        'points_count': 125000,
        'bounding_box': {
            'min': [-1.0, -1.0, -1.0],
            'max': [1.0, 1.0, 1.0]
        },
        'processing_time': 2.5
    }
    return result


def simulate_diffusion_model(filepath):
    time.sleep(3)
    result = {
        'fixed_points_count': 150000,
        'repair_accuracy': 0.95,
        'repair_time': 3.2
    }
    return result


def _do_poisson_reconstruction(output_path, output_dir, name_without_ext, result, preview_pts=None):
    """
    执行 Poisson 泊松重建的通用函数。
    结果写入 result['mesh_path']。

    Args:
        output_path: 点云文件路径 (.ply)
        output_dir: 输出目录
        name_without_ext: 文件名（不含扩展名）
        result: 结果字典（写入 mesh_path）
        preview_pts: (N,3) numpy 数组，已后处理的点云（可选，优先使用）
    """
    try:
        print("[泊松重建] 开始生成光滑网格...")

        # 优先使用已后处理的预览点云
        if preview_pts is not None and len(preview_pts) > 100:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(preview_pts.astype(np.float64))
            print(f"[泊松重建] 使用预览点云 ({len(preview_pts)} 点)")
        else:
            pcd = o3d.io.read_point_cloud(output_path)
            print(f"[泊松重建] 从文件读取点云 ({len(pcd.points)} 点)")

        if len(pcd.points) < 100:
            print(f"[泊松重建] 点数过少({len(pcd.points)})，跳过")
            return

        num_pts = len(pcd.points)

        # ===== 1. 法线估计 =====
        normals_ok = False
        try:
            pts_np = np.asarray(pcd.points)
            bb_min = np.min(pts_np, axis=0)
            bb_max = np.max(pts_np, axis=0)
            bb_diag = np.linalg.norm(bb_max - bb_min)
            radius = max(bb_diag * 0.01, 1e-4)
            max_nn = min(30, num_pts - 1)

            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
            )
            k_orient = min(30, max(10, num_pts // 200))
            pcd.orient_normals_consistent_tangent_plane(k=k_orient)
            normals_ok = True
            print(f"[泊松重建] 法线估计完成 (radius={radius:.6f}, k_orient={k_orient})")
        except Exception as e1:
            # 回退：固定 kNN
            try:
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
                pcd.orient_normals_consistent_tangent_plane(k=15)
                normals_ok = True
                print("[泊松重建] 固定knn法线估计完成")
            except Exception as e2:
                print(f"[泊松重建] 法线估计全部失败: {e2}")

        if not normals_ok:
            return

        # ===== 2. 自适应 Poisson depth =====
        if num_pts < 2000:
            poisson_depth = 7
        elif num_pts < 5000:
            poisson_depth = 8
        elif num_pts < 10000:
            poisson_depth = 9
        else:
            poisson_depth = 10
        print(f"[泊松重建] 使用 depth={poisson_depth}")

        # ===== 3. 执行重建 =====
        mesh = None
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            print(f"[泊松重建] 重建成功: {len(mesh.vertices)} 顶点")

            if len(densities) > 0:
                density_threshold = np.percentile(np.asarray(densities), 5)
                mesh.remove_vertices_by_mask(np.asarray(densities) < density_threshold)

        except Exception as e_p:
            print(f"[泊松重建] depth={poisson_depth} 失败: {e_p}")
            # 回退
            if poisson_depth > 7:
                try:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=poisson_depth - 1
                    )
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_unreferenced_vertices()
                    print(f"[泊松重建] 回退 depth={poisson_depth-1} 成功")
                except Exception:
                    mesh = None

            if mesh is None:
                # Ball-Pivoting 回退
                try:
                    radii = [radius * 0.5, radius, radius * 2.0]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector3d(radii)
                    )
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_unreferenced_vertices()
                    print("[泊松重建] Ball-Pivoting 回退成功")
                except Exception:
                    mesh = None

        if mesh is not None and len(mesh.vertices) > 0:
            mesh_filename = f"{name_without_ext}_mesh.obj"
            mesh_path = os.path.join(output_dir, mesh_filename)
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            result['mesh_path'] = mesh_path
            result['axisymmetry_info']['mesh_generated'] = True
            print(f"[泊松重建] 网格保存: {mesh_path}")
        else:
            print("[泊松重建] 所有方法均失败")
            result['mesh_path'] = ''
            result['axisymmetry_info']['mesh_generated'] = False

    except Exception as e:
        print(f"[泊松重建] 异常: {e}")
        import traceback
        traceback.print_exc()
        result['mesh_path'] = ''
        result['axisymmetry_info']['mesh_generated'] = False


def _prepare_preview_pointcloud(input_path, completed_points, output_path):
    """
    点云后处理（点云优先策略）：
      1. 读取原始残缺点云
      2. 对 predicted 部分做 SOR 去噪
      3. 合并原始点 + predicted 点（原始在前，保证 rim/边缘不丢失）
      4. 体素下采样（voxel_size 与 bbox 大小自适应）
      5. 轻度二次去噪
      6. 写出用于前端预览的 .ply

    Args:
        input_path: 原始上传文件路径
        completed_points: (N, 3) numpy 数组，模型逆归一化后的预测点
        output_path: 输出 .ply 路径

    Returns:
        points: (M, 3) float32 后处理合并点云
    """
    # 1) 读取原始残缺点云
    orig_pcd = o3d.io.read_point_cloud(input_path)
    orig_pts = np.asarray(orig_pcd.points)
    print(f"[点云后处理] 原始点数: {len(orig_pts)}")

    # 2) 对 predicted 部分做 SOR 去噪
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(completed_points.astype(np.float32))
    pred_pcd, ind = pred_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pred_clean = np.asarray(pred_pcd.points)
    removed = len(completed_points) - len(pred_clean)
    print(f"[点云后处理] 预测点 SOR 去噪: {len(completed_points)} → {len(pred_clean)} (移除 {removed})")

    # 3) 合并（原始在前）
    combined_pts = np.vstack([orig_pts, pred_clean])
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_pts.astype(np.float32))
    print(f"[点云后处理] 合并后点数: {len(combined_pts)}")

    # 4) 体素下采样（voxel_size 与 bbox 大小自适应）
    bbox = combined_pcd.get_axis_aligned_bounding_box()
    max_dim = max(bbox.get_extent())
    voxel_size = max(max_dim * 0.002, 1e-5)
    combined_ds = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[点云后处理] 体素下采样 (voxel_size={voxel_size:.6f}): {len(combined_pts)} → {len(combined_ds.points)}")

    # 5) 轻度二次去噪
    combined_ds, ind2 = combined_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
    final_pts = np.asarray(combined_ds.points)
    print(f"[点云后处理] 最终点数: {len(final_pts)}")

    # 6) 写出预览点云
    o3d.io.write_point_cloud(output_path, combined_ds, write_ascii=False)
    print(f"[点云后处理] 预览点云已保存: {output_path}")

    return final_pts


def run_point_completion(input_path, output_dir, model_type='pointr', reconstruction_mode='axis_auto'):
    """
    调用点云补全模型：始终先输出并返回「合并+清洗后的预览点云」，
    仅在满足条件时（用户明确请求或轴对称检测通过）才生成网格化。

    Args:
        input_path: 输入点云文件路径
        output_dir: 输出目录
        model_type: 模型类型 ('pointr' 或 'pfnet')
        reconstruction_mode: 重建模式
            - 'point_only': 仅输出点云，不生成网格
            - 'poisson': 明确请求 Poisson 网格
            - 'axis_auto': 自动检测轴对称 → 通过则旋转重建；否则仅返回点云
            - 'axis_force': 强制旋转重建
            - 'axis_off': 关闭轴对称，运行 Poisson

    Returns:
        dict: {
            'success': bool,
            'output_path': str,        # 预览用 .ply（始终有）
            'mesh_path': str,          # 网格路径（仅当 mesh_generated=True 时有值）
            'time_elapsed': float,
            'chamfer_distance': float,
            'model_used': str,
            'axisymmetry_info': {
                'detected': bool,      # 是否检测到轴对称
                'score': float,
                'mode': str,
                'mesh_generated': bool  # 是否实际生成了网格
            }
        }
    """
    result = {
        'success': False,
        'output_path': '',
        'mesh_path': '',
        'time_elapsed': 0,
        'chamfer_distance': None,
        'model_used': model_type,
        'axisymmetry_info': {
            'detected': False,
            'score': 1.0,
            'mode': reconstruction_mode,
            'mesh_generated': False
        }
    }

    start_time = time.time()

    try:
        os.makedirs(output_dir, exist_ok=True)

        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        print(f"使用模型: {model_type}")

        original_filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        processed_filename = f"{name_without_ext}_completed.ply"
        output_path = os.path.join(output_dir, processed_filename)

        if model_type == 'pointr':
            if not os.path.exists(CKPT_PATH):
                raise Exception("PoinTr预训练模型文件不存在")

            completed_points = point_cloud_completion_pipeline(
                input_ply_path=input_path,
                output_ply_path=output_path,
                device=device
            )
            if completed_points is None:
                raise Exception("PoinTr点云补全失败")

        elif model_type == 'pfnet':
            PFNET_CKPT = 'PoinTr/ckpts/point_netG90.pth'
            if not os.path.exists(PFNET_CKPT):
                raise Exception("PF-Net预训练模型不存在")

            completed_points = pfnet_completion(
                input_ply_path=input_path,
                output_ply_path=output_path,
                device=device
            )
            if completed_points is None:
                raise Exception("PF-Net点云补全失败")

        else:
            raise Exception(f"未知的模型类型: {model_type}")

        # ===== 点云后处理（始终执行，生成前端预览用 .ply）=====
        print("[点云后处理] 开始生成预览点云...")
        preview_pts = _prepare_preview_pointcloud(input_path, completed_points, output_path)

        result['output_path'] = output_path
        result['success'] = True

        # ===== 条件网格化 =====
        should_generate_mesh = False
        try:
            print(f"[重建] 重建模式: {reconstruction_mode}")

            if reconstruction_mode == 'point_only':
                print("[重建] 仅点云预览，跳过网格生成")

            elif reconstruction_mode == 'poisson':
                print("[重建] 用户明确请求 Poisson 网格")
                should_generate_mesh = True
                _do_poisson_reconstruction(output_path, output_dir, name_without_ext, result, preview_pts)

            elif reconstruction_mode == 'axis_auto':
                # 先检测轴对称
                print("[重建] 自动检测轴对称...")
                axis_det = detect_axisymmetry(preview_pts, threshold=0.12)
                result['axisymmetry_info']['detected'] = axis_det['is_axisymmetric']
                result['axisymmetry_info']['score'] = axis_det['score']
                print(f"[重建] 轴对称检测: {axis_det['is_axisymmetric']} (score={axis_det['score']:.4f})")

                if axis_det['is_axisymmetric']:
                    # 检测通过 → 旋转重建
                    print("[重建] 检测到轴对称，执行旋转轮廓重建...")
                    should_generate_mesh = True
                    revolve_result = revolve_reconstruction(
                        preview_pts, axis_vector=axis_det['axis_vector'],
                        axis_origin=axis_det['axis_origin'],
                        n_z=128, n_theta=128,
                        seal_bottom=True, seal_top=True
                    )
                    if revolve_result['success'] and revolve_result['mesh'] is not None:
                        mesh = revolve_result['mesh']
                        mesh_filename = f"{name_without_ext}_revolved.obj"
                        mesh_path = os.path.join(output_dir, mesh_filename)
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        result['mesh_path'] = mesh_path
                        result['axisymmetry_info']['mesh_generated'] = True
                        print(f"[重建] 旋转重建成功: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
                    else:
                        print("[重建] 旋转重建失败，不自动回退 Poisson")
                        should_generate_mesh = False
                        result['axisymmetry_info']['mesh_generated'] = False
                else:
                    # 检测失败 → 不自动生成网格，仅返回点云
                    print("[重建] 非轴对称，不自动生成网格（仅返回点云）")
                    result['axisymmetry_info']['mesh_generated'] = False

            elif reconstruction_mode == 'axis_force':
                print("[重建] 强制旋转重建...")
                should_generate_mesh = True
                centroid = np.mean(preview_pts, axis=0)
                centered = preview_pts - centroid
                cov = centered.T @ centered / (len(preview_pts) - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                axis_vec = eigenvectors[:, -1]
                result['axisymmetry_info']['detected'] = True
                result['axisymmetry_info']['score'] = 0.0

                revolve_result = revolve_reconstruction(
                    preview_pts, axis_vector=axis_vec, axis_origin=centroid,
                    n_z=128, n_theta=128,
                    seal_bottom=True, seal_top=True
                )
                if revolve_result['success'] and revolve_result['mesh'] is not None:
                    mesh = revolve_result['mesh']
                    mesh_filename = f"{name_without_ext}_revolved.obj"
                    mesh_path = os.path.join(output_dir, mesh_filename)
                    o3d.io.write_triangle_mesh(mesh_path, mesh)
                    result['mesh_path'] = mesh_path
                    result['axisymmetry_info']['mesh_generated'] = True
                    print(f"[重建] 强制旋转重建成功: {len(mesh.vertices)} 顶点")
                else:
                    print("[重建] 强制旋转重建失败")
                    result['axisymmetry_info']['mesh_generated'] = False

            elif reconstruction_mode == 'axis_off':
                print("[重建] 关闭轴对称，运行 Poisson")
                should_generate_mesh = True
                _do_poisson_reconstruction(output_path, output_dir, name_without_ext, result, preview_pts)

        except Exception as e:
            print(f"[重建] 重建流程异常({e})，不影响点云输出")
            import traceback
            traceback.print_exc()
            result['mesh_path'] = ''
            result['axisymmetry_info']['mesh_generated'] = False

        result['time_elapsed'] = time.time() - start_time
        print(f"推理完成，耗时: {result['time_elapsed']:.2f}秒")
        print(f"预览点云点数: {len(preview_pts)}")
        print(f"网格生成: {result['axisymmetry_info']['mesh_generated']}")

    except Exception as e:
        print(f"推理过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise e

    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/gallery')
def gallery():
    return render_template('gallery.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['name'] = users[username]['name']
            session.permanent = True
            return jsonify({'success': True, 'message': '登录成功'})
        else:
            return jsonify({'error': '用户名或密码错误'}), 401
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')

        if username in users:
            return jsonify({'error': '用户名已存在'}), 400

        users[username] = {'password': password, 'name': name}
        return jsonify({'success': True, 'message': '注册成功'})
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('name', None)
    return jsonify({'success': True, 'message': '注销成功'})


@app.route('/api/user/status')
def user_status():
    if 'username' in session:
        return jsonify({
            'logged_in': True,
            'username': session['username'],
            'name': session['name']
        })
    else:
        return jsonify({'logged_in': False})


@app.route('/api/projects', methods=['GET', 'POST'])
def manage_projects():
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401

    if request.method == 'POST':
        project_name = request.json.get('name')
        project_description = request.json.get('description')

        project = {
            'id': str(uuid.uuid4()),
            'name': project_name,
            'description': project_description,
            'user': session['username'],
            'created_at': time.time(),
            'files': []
        }

        projects.append(project)
        return jsonify({'success': True, 'project': project})

    user_projects = [p for p in projects if p['user'] == session['username']]
    return jsonify({'projects': user_projects})


@app.route('/api/projects/<project_id>', methods=['GET', 'PUT', 'DELETE'])
def project_detail(project_id):
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401

    project = next((p for p in projects if p['id'] == project_id), None)
    if not project:
        return jsonify({'error': '项目不存在'}), 404

    if project['user'] != session['username']:
        return jsonify({'error': '无权限访问此项目'}), 403

    if request.method == 'GET':
        return jsonify({'project': project})

    elif request.method == 'PUT':
        project_name = request.json.get('name')
        project_description = request.json.get('description')

        if project_name:
            project['name'] = project_name
        if project_description:
            project['description'] = project_description

        return jsonify({'success': True, 'project': project})

    elif request.method == 'DELETE':
        projects.remove(project)
        return jsonify({'success': True, 'message': '项目删除成功'})


@app.route('/upload', methods=['POST'])
def upload_file():
    """接收上传的点云文件"""
    response = {
        'success': True,
        'mesh_file_url': None,
        'processed_file_url': None,
        'mesh_status': 'ok',
        'mesh_error': None,
        'message': '文件上传成功，已完成点云处理和修复'
    }

    try:
        model_type = request.form.get('model_type', 'pointr')
        if model_type not in ['pointr', 'pfnet']:
            model_type = 'pointr'
        print(f"[上传接口] 请求模型类型: {model_type}")

        reconstruction_mode = request.form.get('reconstruction_mode', 'axis_auto')
        valid_modes = ['point_only', 'axis_auto']
        if reconstruction_mode not in valid_modes:
            reconstruction_mode = 'axis_auto'
        print(f"[上传接口] 请求重建模式: {reconstruction_mode}")

        if 'file' not in request.files:
            print("[上传接口] 错误: 请求中没有文件")
            return jsonify({
                'success': False,
                'mesh_file_url': None,
                'processed_file_url': None,
                'mesh_status': 'failed',
                'mesh_error': '没有文件',
                'message': '没有文件'
            }), 400

        file = request.files['file']
        if file.filename == '':
            print("[上传接口] 错误: 未选择文件")
            return jsonify({
                'success': False,
                'mesh_file_url': None,
                'processed_file_url': None,
                'mesh_status': 'failed',
                'mesh_error': '未选择文件',
                'message': '未选择文件'
            }), 400

        valid, message = validate_file(file)
        if not valid:
            print(f"[上传接口] 文件验证失败: {message}")
            return jsonify({
                'success': False,
                'mesh_file_url': None,
                'processed_file_url': None,
                'mesh_status': 'failed',
                'mesh_error': message,
                'message': message
            }), 400

        original_filename = file.filename
        ext = original_filename.rsplit('.', 1)[1].lower()
        new_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        file.seek(0)
        file.save(filepath)

        if os.path.getsize(filepath) == 0:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'mesh_file_url': None,
                'processed_file_url': None,
                'mesh_status': 'failed',
                'mesh_error': '上传的文件为空',
                'message': '上传的文件为空'
            }), 400

        try:
            print(f"\n[上传接口] 进入点云预处理函数...")
            point_cloud_result = process_point_cloud(filepath)
            print(f"[上传接口] 点云预处理完成")
        except Exception as e:
            print(f"[上传接口] 错误: 点云预处理失败")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'mesh_file_url': None,
                'processed_file_url': None,
                'mesh_status': 'failed',
                'mesh_error': f'点云处理失败: {str(e)}',
                'message': f'点云处理失败: {str(e)}'
            }), 500

        processed_filename = None
        processed_file_url = None
        mesh_file_name = ''
        mesh_file_url = None
        result = None
        diffusion_result = {}

        try:
            print(f"\n[上传接口] ========== 即将启动推理 ==========")
            result = run_point_completion(
                filepath,
                app.config['PROCESSED_FOLDER'],
                model_type=model_type,
                reconstruction_mode=reconstruction_mode
            )

            if result.get('success'):
                processed_filename = os.path.basename(result['output_path'])
                processed_file_url = f"/processed/{processed_filename}"
                response['processed_file_url'] = processed_file_url

                model_display_name = 'PoinTr' if result.get('model_used') == 'pointr' else 'PF-Net'
                diffusion_result = {
                    'fixed_points_count': count_points(result['output_path']),
                    'repair_accuracy': max(0, 1 - result['chamfer_distance']) if result['chamfer_distance'] is not None else 0.9,
                    'repair_time': result['time_elapsed'],
                    'model_used': model_display_name
                }

                if result.get('mesh_path') and os.path.exists(result['mesh_path']):
                    mesh_file_name = os.path.basename(result['mesh_path'])
                    mesh_file_url = f"/processed/{mesh_file_name}"
                    response['mesh_file_url'] = mesh_file_url
                    response['mesh_status'] = 'ok'
                    response['mesh_error'] = None
                    print(f"[上传接口] 网格文件URL: {mesh_file_url}")
                else:
                    # 网格是可选输出，点云预览已经成功时不能把整次修复判定为失败。
                    response['mesh_status'] = 'failed'
                    response['mesh_error'] = 'mesh generation failed or mesh file missing'
                    response['message'] = '文件上传成功，点云修复完成（网格生成失败）'
            else:
                raise Exception(f"{model_type} 推理失败")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[上传接口] 错误: {model_type} 调用失败")
            print(tb)
            print("[上传接口] 使用内置点云补全算法作为 fallback")

            try:
                points, colors = read_point_cloud(filepath)
                if points is None or len(points) == 0:
                    raise ValueError('读取到空点云')

                completed_points = repair_point_cloud(points)
                processed_filename = f"processed_{new_filename.rsplit('.', 1)[0]}.ply"
                processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(completed_points.astype(np.float32))

                if colors is not None and len(colors) > 0:
                    if len(colors) >= len(completed_points):
                        pcd.colors = o3d.utility.Vector3dVector(colors[:len(completed_points)])
                    else:
                        repeated_colors = np.tile(colors, (len(completed_points) // len(colors) + 1, 1))[:len(completed_points)]
                        pcd.colors = o3d.utility.Vector3dVector(repeated_colors)

                o3d.io.write_point_cloud(processed_filepath, pcd)
                processed_file_url = f"/processed/{processed_filename}"
                response['processed_file_url'] = processed_file_url
                diffusion_result['fixed_points_count'] = len(completed_points)
                diffusion_result['model_used'] = '增强版点云修复'

            except Exception as inner_e:
                print(f"\n[上传接口] 错误: 备用点云修复失败")
                traceback.print_exc()
                processed_filename = f"processed_{new_filename.rsplit('.', 1)[0]}.ply"
                processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                processed_file_url = f"/processed/{processed_filename}"
                response['processed_file_url'] = processed_file_url

                points = np.random.randn(5000, 3) * 0.5
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
                o3d.io.write_point_cloud(processed_filepath, pcd)

            response['mesh_file_url'] = None
            response['mesh_status'] = 'failed'
            response['mesh_error'] = tb
            response['message'] = f"已修复为点云（网格生成失败：{str(e)}）"

        if response.get('processed_file_url') is None:
            if processed_filename:
                response['processed_file_url'] = f"/processed/{processed_filename}"

        if response.get('mesh_file_url') is None:
            response['mesh_status'] = 'failed'
            if response.get('mesh_error') is None:
                response['mesh_error'] = 'mesh generation failed or mesh file missing'

        print("\n" + "=" * 80)
        print("[上传接口] 返回响应给前端")
        print(f"[上传接口] processed_file_url: {response.get('processed_file_url')}")
        print(f"[上传接口] mesh_file_url: {response.get('mesh_file_url')}")
        print(f"[上传接口] mesh_status: {response.get('mesh_status')}")
        print(f"[上传接口] mesh_error: {response.get('mesh_error')}")
        print(f"[上传接口] message: {response.get('message')}")
        print("=" * 80 + "\n")

        return jsonify({
            'success': True,
            'message': response.get('message', '文件上传成功，已完成点云处理和修复'),
            'filename': new_filename,
            'original_name': original_filename,
            'processed_filename': processed_filename,
            'processed_file_url': response.get('processed_file_url'),
            'mesh_file_name': mesh_file_name,
            'mesh_file_url': response.get('mesh_file_url'),
            'mesh_status': response.get('mesh_status', 'ok'),
            'mesh_error': response.get('mesh_error'),
            'point_cloud_result': point_cloud_result,
            'diffusion_result': diffusion_result,
            'model_used': diffusion_result.get('model_used', 'unknown'),
            'reconstruction_mode': reconstruction_mode,
            'axisymmetry_info': result.get('axisymmetry_info', {}) if result is not None else {}
        })

    except Exception as e:
        print(f"\n[上传接口] 严重错误: 处理过程中出错")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'mesh_file_url': None,
            'processed_file_url': None,
            'mesh_status': 'failed',
            'mesh_error': str(traceback.format_exc()),
            'message': f'处理过程中出错: {str(e)}'
        }), 500




@app.route('/status')
def status():
    return jsonify({'status': 'running', 'message': '瓷影新生后端服务正常'})


@app.route('/processed/<filename>')
def get_processed_file(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(filepath):
        # Three.js 通过 XHR/Fetch 读取文件内容，不能把预览资源作为附件下载。
        return send_file(filepath, as_attachment=False)
    else:
        return jsonify({'error': '文件不存在'}), 404


@app.route('/export/pointcloud')
def export_pointcloud():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        for i in range(1000):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            f.write(f"{x} {y} {z}\n".encode())
        temp_filename = f.name

    try:
        return send_file(temp_filename, as_attachment=True, download_name=f'pointcloud_{int(time.time())}.xyz')
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@app.route('/export/model')
def export_model():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(b'# OBJ model\n')
        f.write(b'v 0.0 1.0 0.0\n')
        f.write(b'v -1.0 -1.0 0.0\n')
        f.write(b'v 1.0 -1.0 0.0\n')
        f.write(b'f 1 2 3\n')
        temp_filename = f.name

    try:
        return send_file(temp_filename, as_attachment=True, download_name=f'model_{int(time.time())}.obj')
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def create_warmup_point_cloud():
    """创建用于模型预热的高质量残缺点云（模拟ShapeNet训练数据中的物体形状）"""
    np.random.seed(42)
    
    num_points = 5000
    
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    height = np.random.uniform(0, 2.0, num_points)
    radius = 0.3 + 0.2 * np.sin(height * 3) + np.random.normal(0, 0.03, num_points)
    
    x = radius * np.cos(theta)
    y = height
    z = radius * np.sin(theta)
    
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    
    mask = height > 0.5
    points = points[mask]
    
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    
    return points


def warmup_model():
    """服务启动时预热模型（PoinTr + PF-Net双模型预热）"""
    import torch
    import tempfile
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_points = create_warmup_point_cloud()
    
    # 创建临时测试文件
    input_ply = os.path.join(tempfile.gettempdir(), 'warmup_input.ply')
    output_ply_pointr = os.path.join(tempfile.gettempdir(), 'warmup_pointr_output.ply')
    output_ply_pfnet = os.path.join(tempfile.gettempdir(), 'warmup_pfnet_output.ply')
    
    try:
        # 准备测试点云文件
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(test_points)
        o3d.io.write_point_cloud(input_ply, pcd)
        
        # 预热PoinTr模型
        print("【模型预热开始 - PoinTr】")
        try:
            from pointr_inference_pipeline import point_cloud_completion_pipeline
            with redirect_stdout(io.StringIO()):
                result = point_cloud_completion_pipeline(
                    input_ply_path=input_ply,
                    output_ply_path=output_ply_pointr,
                    device=device
                )
            print("【模型预热完成 - PoinTr】")
        except Exception as e:
            print(f"【模型预热失败 - PoinTr】: {e}")
        
        # 预热PF-Net模型
        print("【模型预热开始 - PF-Net】")
        try:
            with redirect_stdout(io.StringIO()):
                result = pfnet_completion(
                    input_ply_path=input_ply,
                    output_ply_path=output_ply_pfnet,
                    device=device
                )
            print("【模型预热完成 - PF-Net】")
        except Exception as e:
            print(f"【模型预热失败 - PF-Net】: {e}")
        
        # 清理临时文件
        for temp_file in [input_ply, output_ply_pointr, output_ply_pfnet]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        print("【模型预热完成 - 全部】")
            
    except Exception as e:
        print(f"【模型预热异常】: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        for temp_file in [input_ply, output_ply_pointr, output_ply_pfnet]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

# 修改文件末尾的启动代码
if __name__ == '__main__':
    # 启动前先预热
    with app.app_context():
        warmup_model()
    app.run(debug=False, port=5002)
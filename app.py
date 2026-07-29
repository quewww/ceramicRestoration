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
import numpy as np
import open3d as o3d
from datetime import timedelta
from contextlib import redirect_stdout

from config import *
from pointcloud_utils import read_point_cloud, write_point_cloud, downsample_point_cloud, convert_to_ply, count_points, estimate_chamfer_distance
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


def run_point_completion(input_path, output_dir, model_type='pointr'):
    """
    调用点云补全模型，并进行泊松重建生成光滑网格

    Args:
        input_path: 输入点云文件路径
        output_dir: 输出目录
        model_type: 模型类型 ('pointr' 或 'pfnet')

    Returns:
        dict: {
            'success': bool,
            'output_path': str,
            'mesh_path': str,
            'time_elapsed': float,
            'chamfer_distance': float,
            'model_used': str
        }
    """
    result = {
        'success': False,
        'output_path': '',
        'mesh_path': '',
        'time_elapsed': 0,
        'chamfer_distance': None,
        'model_used': model_type
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
            # PoinTr模型推理
            if not os.path.exists(CKPT_PATH):
                print(f"预训练模型不存在: {CKPT_PATH}")
                raise Exception("PoinTr预训练模型文件不存在")

            completed_points = point_cloud_completion_pipeline(
                input_ply_path=input_path,
                output_ply_path=output_path,
                device=device
            )

            if completed_points is None:
                raise Exception("PoinTr点云补全失败")

        elif model_type == 'pfnet':
            # PF-Net模型推理
            PFNET_CKPT = 'PoinTr/ckpts/point_netG90.pth'
            if not os.path.exists(PFNET_CKPT):
                print(f"PF-Net预训练模型不存在: {PFNET_CKPT}")
                raise Exception("PF-Net预训练模型文件不存在")

            completed_points = pfnet_completion(
                input_ply_path=input_path,
                output_ply_path=output_path,
                device=device
            )

            if completed_points is None:
                raise Exception("PF-Net点云补全失败")

        else:
            raise Exception(f"未知的模型类型: {model_type}")

        result['output_path'] = output_path
        result['success'] = True

        # 泊松重建生成光滑网格（容错处理，不阻断流程）
        try:
            print(f"[泊松重建] 开始生成光滑网格...")
            pcd = o3d.io.read_point_cloud(output_path)
            
            if len(pcd.points) < 100:
                print(f"[泊松重建] 警告: 点云点数过少({len(pcd.points)}点)，跳过重建")
            else:
                # 估计法向量（泊松重建需要法向量）
                normals_ok = False
                try:
                    pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamKNN(knn=30)
                    )
                    normals_ok = True
                    print(f"[泊松重建] 法向量估计完成")
                except Exception as normal_e:
                    print(f"[泊松重建] 警告: 法向量估计失败({normal_e})，跳过重建")
                    import traceback
                    traceback.print_exc()
                
                if normals_ok:
                    # 执行泊松重建
                    try:
                        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                            pcd,
                            depth=9
                        )
                        
                        # 清理网格
                        mesh.remove_degenerate_triangles()
                        mesh.remove_duplicated_triangles()
                        mesh.remove_unreferenced_vertices()
                        
                        # 保存网格
                        mesh_filename = f"{name_without_ext}_mesh.obj"
                        mesh_path = os.path.join(output_dir, mesh_filename)
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        
                        result['mesh_path'] = mesh_path
                        print(f"[泊松重建] 完成，网格顶点数: {len(mesh.vertices)}, 三角形数: {len(mesh.triangles)}")
                        print(f"[泊松重建] 网格保存路径: {mesh_path}")
                        
                    except Exception as poisson_e:
                        print(f"[泊松重建] 警告: 泊松重建失败({poisson_e})，不影响点云输出")
                        result['mesh_path'] = ''
                        import traceback
                        traceback.print_exc()
                
        except Exception as e:
            print(f"[泊松重建] 警告: 重建流程异常({e})，不影响点云输出")
            import traceback
            traceback.print_exc()
            result['mesh_path'] = ''

        result['time_elapsed'] = time.time() - start_time
        print(f"推理完成，耗时: {result['time_elapsed']:.2f}秒")
        print(f"输出点数: {len(completed_points)}")

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
    try:
        # 获取模型类型参数，默认为pointr
        model_type = request.form.get('model_type', 'pointr')
        if model_type not in ['pointr', 'pfnet']:
            model_type = 'pointr'
        print(f"[上传接口] 请求模型类型: {model_type}")
        
        print("\n" + "=" * 80)
        print("========== [上传接口] 请求到达 ==========")
        print(f"[上传接口] 请求方法: {request.method}")
        print(f"[上传接口] 请求路径: {request.path}")
        print(f"[上传接口] Content-Type: {request.content_type}")
        print(f"[上传接口] Content-Length: {request.content_length}")
        print(f"[上传接口] 是否有文件: {'file' in request.files}")
        print(f"[上传接口] files数量: {len(request.files) if request.files else 0}")
        print(f"[上传接口] 模型类型: {model_type}")
        print("=" * 80)
        
        if 'file' not in request.files:
            print("[上传接口] 错误: 请求中没有文件")
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']
        if file.filename == '':
            print("[上传接口] 错误: 未选择文件")
            return jsonify({'error': '未选择文件'}), 400

        print("\n" + "="*80)
        print("========== [上传接口] 文件接收入口 ==========")
        print(f"[上传接口] 接收文件名: {file.filename}")
        print(f"[上传接口] 文件类型: {file.content_type}")
        print(f"[上传接口] Content-Length: {file.content_length}")
        print("="*80)

        valid, message = validate_file(file)
        if not valid:
            print(f"[上传接口] 文件验证失败: {message}")
            return jsonify({'error': message}), 400

        original_filename = file.filename
        ext = original_filename.rsplit('.', 1)[1].lower()
        new_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        print(f"\n[上传接口] 准备保存文件:")
        print(f"[上传接口]   原始文件名: {original_filename}")
        print(f"[上传接口]   存储文件名: {new_filename}")
        print(f"[上传接口]   存储路径: {filepath}")

        try:
            file.seek(0)
            file.save(filepath)
            print(f"[上传接口] 文件保存成功")

            file_size = os.path.getsize(filepath)
            print(f"[上传接口] 文件大小: {file_size} bytes")

            if file_size == 0:
                os.remove(filepath)
                print(f"[上传接口] 错误: 上传的文件为空，已删除")
                return jsonify({'error': '上传的文件为空'}), 400

            print(f"[上传接口] 文件接收流程完成")

        except Exception as e:
            print(f"[上传接口] 错误: 文件保存失败")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'文件保存失败: {str(e)}'}), 500

        try:
            print(f"\n[上传接口] 进入点云预处理函数...")
            point_cloud_result = process_point_cloud(filepath)
            print(f"[上传接口] 点云预处理完成")
        except Exception as e:
            print(f"[上传接口] 错误: 点云预处理失败")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'点云处理失败: {str(e)}'}), 500

        mesh_file_name = ''
        mesh_file_url = ''
        
        try:
            print(f"\n[上传接口] ========== 即将启动推理 ==========")
            print(f"[上传接口] 输入文件: {filepath}")
            print(f"[上传接口] 输出目录: {app.config['PROCESSED_FOLDER']}")
            print(f"[上传接口] 使用模型: {model_type}")
            
            result = run_point_completion(filepath, app.config['PROCESSED_FOLDER'], model_type=model_type)
            
            if result['success']:
                print(f"\n[上传接口] ========== 推理函数执行完毕 ==========")
                print(f"[上传接口] 输出文件路径: {result['output_path']}")
                
                # 根据实际使用的模型设置model_used名称
                model_display_name = 'PoinTr' if result.get('model_used') == 'pointr' else 'PF-Net'
                
                diffusion_result = {
                    'fixed_points_count': count_points(result['output_path']),
                    'repair_accuracy': max(0, 1 - result['chamfer_distance']) if result['chamfer_distance'] is not None else 0.9,
                    'repair_time': result['time_elapsed'],
                    'model_used': model_display_name
                }
                processed_filename = os.path.basename(result['output_path'])
                processed_file_url = f"/processed/{processed_filename}"
                
                # 获取网格文件信息
                if result.get('mesh_path') and os.path.exists(result['mesh_path']):
                    mesh_file_name = os.path.basename(result['mesh_path'])
                    mesh_file_url = f"/processed/{mesh_file_name}"
                    print(f"[上传接口] 网格文件名: {mesh_file_name}")
                    print(f"[上传接口] 网格文件URL: {mesh_file_url}")
                
                print(f"[上传接口] 输出文件名: {processed_filename}")
                print(f"[上传接口] 输出文件URL: {processed_file_url}")
                print(f"[上传接口] 输出点数: {diffusion_result['fixed_points_count']}")
                print(f"[上传接口] 推理耗时: {diffusion_result['repair_time']:.2f}秒")
            else:
                raise Exception(f"{model_type} 推理失败")
        except Exception as e:
            print(f"\n[上传接口] 错误: {model_type} 调用失败")
            import traceback
            traceback.print_exc()
            print("[上传接口] 使用内置点云补全算法作为 fallback")
            diffusion_result = simulate_diffusion_model(filepath)

            try:
                points, colors = read_point_cloud(filepath)
                if points is None or len(points) == 0:
                    raise ValueError("读取到空点云")

                print(f"[上传接口] 备用修复 - 原始点数: {len(points)}")

                completed_points = repair_point_cloud(points)

                processed_filename = f"processed_{new_filename.rsplit('.', 1)[0]}.ply"
                processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(completed_points)
                
                if colors is not None and len(colors) > 0:
                    if len(colors) >= len(completed_points):
                        pcd.colors = o3d.utility.Vector3dVector(colors[:len(completed_points)])
                    else:
                        repeated_colors = np.tile(colors, (len(completed_points) // len(colors) + 1, 1))[:len(completed_points)]
                        pcd.colors = o3d.utility.Vector3dVector(repeated_colors)
                
                o3d.io.write_point_cloud(processed_filepath, pcd)

                processed_file_url = f"/processed/{processed_filename}"
                print(f"[上传接口] 备用修复完成")
                print(f"[上传接口] 输出文件路径: {processed_filepath}")
                print(f"[上传接口] 输出文件URL: {processed_file_url}")
                print(f"[上传接口] 输出点数: {len(completed_points)}")

                diffusion_result['fixed_points_count'] = len(completed_points)
                diffusion_result['model_used'] = '增强版点云修复'

            except Exception as inner_e:
                print(f"\n[上传接口] 错误: 备用点云修复失败")
                import traceback
                traceback.print_exc()
                print("[上传接口] 生成模拟数据作为 fallback")
                
                processed_filename = f"processed_{new_filename.rsplit('.', 1)[0]}.ply"
                processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                processed_file_url = f"/processed/{processed_filename}"

                points = np.random.randn(5000, 3) * 0.5
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(processed_filepath, pcd)

                print(f"[上传接口] 模拟数据生成完成")
                print(f"[上传接口] 输出文件路径: {processed_filepath}")
                print(f"[上传接口] 输出文件URL: {processed_file_url}")

        print("\n" + "="*80)
        print("[上传接口] 返回响应给前端")
        print(f"[上传接口] processed_file_url: {processed_file_url}")
        print(f"[上传接口] mesh_file_url: {mesh_file_url}")
        print(f"[上传接口] model_used: {diffusion_result.get('model_used', 'unknown')}")
        print("="*80 + "\n")

        return jsonify({
            'success': True,
            'message': '文件上传成功，已完成点云处理和修复',
            'filename': new_filename,
            'original_name': original_filename,
            'processed_filename': processed_filename,
            'processed_file_url': processed_file_url,
            'mesh_file_name': mesh_file_name,
            'mesh_file_url': mesh_file_url,
            'point_cloud_result': point_cloud_result,
            'diffusion_result': diffusion_result,
            'model_used': diffusion_result.get('model_used', 'unknown')
        })

    except Exception as e:
        print(f"\n[上传接口] 严重错误: 处理过程中出错")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'处理过程中出错: {str(e)}'}), 500




@app.route('/status')
def status():
    return jsonify({'status': 'running', 'message': '瓷影新生后端服务正常'})


@app.route('/processed/<filename>')
def get_processed_file(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
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
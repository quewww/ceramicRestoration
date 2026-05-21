import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PoinTr 相关路径
POINTR_ROOT = os.path.join(BASE_DIR, 'PoinTr')
CKPT_PATH = os.path.join(POINTR_ROOT, 'ckpts/pointr_training_from_scratch_c55_best.pth')
CFG_PATH = os.path.join(POINTR_ROOT, 'cfgs/ShapeNet55_models/PoinTr.yaml')
INFERENCE_SCRIPT = os.path.join(POINTR_ROOT, 'tools/inference.py')

# 上传和输出目录
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
USER_DATA_FOLDER = os.path.join(BASE_DIR, 'user_data')

# 模型参数
MAX_POINTS = 2048
INFERENCE_TIMEOUT = 300
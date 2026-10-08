# -*- coding: utf-8 -*-

import os
import sys
from flask import Flask, request, jsonify, Blueprint
import traceback
import json
import numpy as np
import pandas as pd
import torch
import pickle as pkl
from datetime import datetime, timezone, timedelta
import TimesNet
from timefeatures import time_features
import logging
import warnings

# 创建蓝图
api_bp = Blueprint('api', __name__, url_prefix='/api/load_day')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 模型和预处理初始化
class ModelArgs:
    def __init__(self):
        self.task_name = "long_term_forecast"
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.e_layers = 2
        self.features = 'S'
        self.inverse = True
        self.use_amp = False
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.d_model = 16
        self.d_ff = 32
        self.top_k = 5
        self.num_kernels = 6
        self.embed = 'timeF'
        self.timeenc = 1
        self.freq = 'h'
        self.dropout = 0.1


# 加载模型和scaler
def load_model_and_scaler():
    try:
        args = ModelArgs()
        model = TimesNet.Model(args)

        # 加载模型以及归一器
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "load_forecast_model.pth")
        scaler_path = os.path.join(base_dir, "scaler.pkl")

        forecast_model = torch.load(model_path, map_location='cpu')
        model.load_state_dict(forecast_model)
        model.eval()

        with open(scaler_path, 'rb') as f:
            scaler = pkl.load(f)

        logger.info("模型和标准化器加载成功")
        return args, model, scaler
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


# 全局加载模型
try:
    MODEL_ARGS, MODEL, SCALER = load_model_and_scaler()
except Exception as e:
    logger.critical(f"服务启动失败: 无法加载模型 - {str(e)}")
    sys.exit(1)


def prepare_input_data(input_json):
    """准备模型输入数据（处理毫秒级时间戳和缺失值）"""
    try:
        data = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(data, dict) or "feats" not in data:
            raise ValueError("输入数据格式错误，需要包含feats字段")

        # 读取时间
        timestamps = [
            datetime.strptime(item["timestamp"],"%Y-%m-%d %H:%M:%S")
            .strftime("%Y-%m-%d %H:%M:%S")
            for item in data["feats"]
        ]

        # 先收集所有有效值计算均值
        valid_values = []
        for item in data["feats"]:
            val = item.get("prev_load")
            # 处理空字符串、None 和空白字符串
            if val is not None and str(val).strip() != '':
                try:
                    value = float(val)
                    valid_values.append(value)
                except (ValueError, TypeError):
                    # 记录但跳过无效值
                    logger.debug(f"无法转换的值: {val}，已跳过")
                    pass

        if not valid_values:
            raise ValueError("输入数据中没有有效的prev_load值")

        mean_value = sum(valid_values) / len(valid_values)

        # 处理缺失值（使用前向填充，如果没有前值则使用均值）
        values = []
        prev_valid_value = None

        for item in data["feats"]:
            val = item.get("prev_load")
            # 处理空字符串、None 和空白字符串
            if val is not None and str(val).strip() != '':
                try:
                    value = float(val)
                    prev_valid_value = value  # 更新最后一个有效值
                    values.append(value)
                except (ValueError, TypeError):
                    # 无法转换的值视为缺失值
                    if prev_valid_value is not None:
                        values.append(prev_valid_value)  # 前向填充
                    else:
                        values.append(mean_value)  # 使用均值填充
            else:
                # 空值或空字符串
                if prev_valid_value is not None:
                    values.append(prev_valid_value)  # 前向填充
                else:
                    values.append(mean_value)  # 使用均值填充

        # logger.info(f"处理后数据点数量: {len(values)}, 首时间戳: {timestamps[0]}, 均值: {mean_value:.2f}")

        if len(values) < MODEL_ARGS.seq_len:
            raise ValueError(f"需要至少{MODEL_ARGS.seq_len}个数据点，当前收到{len(values)}")

        return np.array(values).reshape((MODEL_ARGS.seq_len, MODEL_ARGS.enc_in)), timestamps
    except json.JSONDecodeError:
        raise ValueError("输入数据不是有效的JSON格式")
    except Exception as e:
        logger.error(f"输入处理失败: {str(e)}")
        raise


@api_bp.route('/predict', methods=['POST'])
def predict():
    """API endpoint for load forecast"""
    try:
        # 获取输入数据
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                "success": False,
                "code": "40001",
                "message": "No data provided",
                "data": None
            }), 400

        # 记录请求数据
        logger.info(f"Received prediction request with {len(input_data.get('feats', []))} data points")

        # 准备输入数据
        historical_data, timestamps = prepare_input_data(input_data)

        # 执行预测
        with torch.no_grad():
            seq_x, seq_x_mark, dec_inp, dec_inp_mark = prepare_single_sequence(
                MODEL_ARGS, SCALER, historical_data, timestamps
            )
            outputs = MODEL(seq_x, seq_x_mark, dec_inp, dec_inp_mark)

        # 后处理
        f_dim = -1 if MODEL_ARGS.features == 'MS' else 0
        outputs = outputs[:, -MODEL_ARGS.pred_len:, f_dim:]
        outputs_np = outputs.detach().cpu().numpy()

        if MODEL_ARGS.inverse:
            outputs_np = SCALER.inverse_transform(outputs_np.reshape(-1, 1)).reshape(outputs_np.shape)

        # 生成响应数据
        last_timestamp = pd.to_datetime(timestamps[-1])
        future_timestamps = _generate_past_and_future_timestamps(MODEL_ARGS, last_timestamp, MODEL_ARGS.label_len,
                                                                 MODEL_ARGS.pred_len)
        future_timestamps = future_timestamps[MODEL_ARGS.label_len + 1:]

        # 按照api_desc中的格式返回
        response = {
            "success": True,
            "code": "00000",
            "message": "",
            "data": [
                {
                    "predicted_load": str(val),
                    "timestamp": str(ts)
                }
                for ts, val in zip(future_timestamps, outputs_np.flatten())
            ]
        }
        return jsonify(response)

    except ValueError as ve:
        logger.warning(f"输入数据验证失败: {str(ve)}")
        return jsonify({
            "success": False,
            "code": "40002",
            "message": str(ve),
            "data": None
        }), 400
    except Exception as e:
        logger.error(f"预测失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "code": "50001",
            "message": "Internal server error",
            "data": None
        }), 500


def prepare_single_sequence(args, scaler, historical_data, timestamps):
    """准备单个序列的推理输入 """
    if historical_data.shape[0] < args.seq_len:
        raise ValueError(f"历史数据长度{historical_data.shape[0]}小于seq_len{args.seq_len}")

    seq_x = historical_data[-args.seq_len:]
    seq_x_scaled = scaler.transform(seq_x)

    seq_x_mark = _create_time_features(args, timestamps[-args.seq_len:])

    dec_inp_data = historical_data[-args.label_len:]
    dec_inp_scaled = scaler.transform(dec_inp_data)
    dec_inp = np.zeros((args.label_len + args.pred_len, historical_data.shape[1]))
    dec_inp[:args.label_len] = dec_inp_scaled

    last_timestamp = pd.to_datetime(timestamps[-1])
    future_timestamps = _generate_past_and_future_timestamps(args, last_timestamp, args.label_len, args.pred_len)
    dec_inp_mark = _create_time_features(args, future_timestamps)

    seq_x_tensor = torch.FloatTensor(seq_x_scaled).unsqueeze(0)
    seq_x_mark_tensor = torch.FloatTensor(seq_x_mark).unsqueeze(0)
    dec_inp_tensor = torch.FloatTensor(dec_inp).unsqueeze(0)
    dec_inp_mark_tensor = torch.FloatTensor(dec_inp_mark).unsqueeze(0)

    return seq_x_tensor, seq_x_mark_tensor, dec_inp_tensor, dec_inp_mark_tensor


def _create_time_features(args, timestamps):
    """创建时间特征 """
    df_stamp = pd.DataFrame({'timestamp': timestamps})
    df_stamp['timestamp'] = pd.to_datetime(df_stamp['timestamp'], format='%Y-%m-%d %H:%M:%S')
    if args.timeenc == 0:
        df_stamp['month'] = df_stamp['timestamp'].dt.month
        df_stamp['day'] = df_stamp['timestamp'].dt.day
        df_stamp['weekday'] = df_stamp['timestamp'].dt.weekday
        df_stamp['hour'] = df_stamp['timestamp'].dt.hour
        data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
    else:
        data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=args.freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp


def _generate_past_and_future_timestamps(args, last_timestamp, past_steps, future_steps):
    """生成过去和未来的时间戳序列"""
    timestamps = []

    # 生成过去时间戳（从最远到最近）
    for i in range(past_steps, 0, -1):
        if args.freq == 'h':
            new_ts = last_timestamp + timedelta(minutes=i * 15)
        elif args.freq == 't' or args.freq == 'min':
            new_ts = last_timestamp - timedelta(minutes=i)
        elif args.freq == 'd':
            new_ts = last_timestamp - timedelta(days=i)
        elif args.freq == 'w':
            new_ts = last_timestamp - timedelta(weeks=i)
        else:
            new_ts = last_timestamp - timedelta(hours=i)  # 默认按小时
        timestamps.append(new_ts)

    # 添加当前时间戳
    timestamps.append(last_timestamp)

    # 生成未来时间戳
    for i in range(1, future_steps + 1):
        if args.freq == 'h':
            new_ts = last_timestamp + timedelta(minutes=i * 15)
        elif args.freq == 't' or args.freq == 'min':
            new_ts = last_timestamp + timedelta(minutes=i)
        elif args.freq == 'd':
            new_ts = last_timestamp + timedelta(days=i)
        elif args.freq == 'w':
            new_ts = last_timestamp + timedelta(weeks=i)
        else:
            new_ts = last_timestamp + timedelta(hours=i)  # 默认按小时
        timestamps.append(new_ts)

    return [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in timestamps]

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口，返回详细的模型和服务状态
    """
    try:
        # 检查模型加载状态
        model_loaded = _MODEL_LOADED
        health_status = {
            "status": "healthy" if model_loaded else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "loaded": model_loaded,
                "info": _MODEL_INFO if model_loaded else {}
            },
            "service": {
                "name": "负荷预测天预测服务",
                "version": "1.0.0",
                "prediction_type": "day_forecast"
            }
        }
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# 创建 Flask 应用
app = Flask(__name__)

# 全局变量，用于缓存模型和特征名
_MODEL_CACHE = None
_FEATURE_NAMES_CACHE = None
_MODEL_LOADED = False
_MODEL_INFO = {}

# 注册蓝图
app.register_blueprint(api_bp)

# 可选：保持根路径的接口（如果需要）
@app.route('/')
def index():
    return jsonify({
        "service": "负荷预测日前预测服务",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/load_day/health",
            "predict": "/api/load_day/predict",
        }
    })

if __name__ == '__main__':
    # 配置 Flask 服务
    app.config['JSON_AS_ASCII'] = False  # 支持中文
    app.config['JSON_SORT_KEYS'] = False  # 保持JSON键顺序
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大请求大小

    # 隐藏 Flask 开发服务器警告
    warnings.filterwarnings("ignore", message=".*Development.*")

    # 启动服务
    print("=" * 70)
    print("负荷预测单点预测服务")
    print("=" * 70)
    print(f"服务地址: http://localhost:5000")
    print(f"健康检查: GET  http://localhost:5000/api/load_day/health")
    print(f"预测接口: POST http://localhost:5000/api/load_day/predict")
    print("=" * 70)

    # 开发模式运行
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=5000,
        debug=False,  # 生产环境建议设为 False
        threaded=True  # 支持多线程
    )
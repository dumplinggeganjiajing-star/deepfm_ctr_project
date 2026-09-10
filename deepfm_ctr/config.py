# -*- coding: utf-8 -*-
"""配置管理"""
import os
# 数据配置
DATA_CONFIG = {
    # 云端优先通过环境变量或 train.py --data_dir 指定数据目录。
    'data_dir': os.getenv('DEEPFM_DATA_DIR', './data'),
    # 拼接文件路径
    'start_date': '20260420',
    'split_date': '20260520',
    'end_date': '20260620',
    'val_ratio': 0.1,
    'seed': 42,
}

# 模型配置
MODEL_CONFIG = {
    'embed_dim': 8,
    'dnn_hidden_units': (128, 128),
    'dropout_rate': 0.3,
    'learning_rate': 1e-3,
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 2048,
    'epochs': 10,
    'early_stopping_patience': 3,
}

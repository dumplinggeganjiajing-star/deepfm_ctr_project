#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测脚本（推理部署用）"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf
from deepfm_ctr.DataProcess import DataProcessor, GROUP_COL


def predict(model_path, preprocessor_path, data_file, output_file):
    """批量预测"""
    # 加载模型
    model = tf.keras.models.load_model(model_path, compile=False)
    processor = DataProcessor.load_artifact(preprocessor_path)
    
    # 加载数据
    df = pd.read_parquet(data_file)
    
    # 必须复用训练期的派生规则、词汇表、OOV 规则和数值统计量。
    transformed = processor.transform(df)
    x = processor.make_model_inputs(transformed)
    predictions = model.predict(x, verbose=0).reshape(-1)
    
    # 输出结果
    df['prediction'] = predictions
    output_parent = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_parent, exist_ok=True)
    if GROUP_COL in df.columns:
        output_columns = [GROUP_COL, 'prediction']
    else:
        print(f"⚠️ 预测数据缺少 {GROUP_COL}，结果仅输出 prediction")
        output_columns = ['prediction']
    df[output_columns].to_csv(output_file, index=False)
    print(f"预测完成，结果保存至: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--preprocessor', required=True, help='预处理 artifact 路径')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    args = parser.parse_args()
    
    predict(args.model, args.preprocessor, args.data, args.output)

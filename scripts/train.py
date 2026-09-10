#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练脚本（生产部署用）"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfm_ctr.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from deepfm_ctr.DataProcess import DataProcessor, GROUP_COL
from deepfm_ctr.model import DeepFMBuilder
from deepfm_ctr.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_CONFIG['data_dir'])
    parser.add_argument('--start_date', default=DATA_CONFIG['start_date'])
    parser.add_argument('--split_date', default=DATA_CONFIG['split_date'])
    parser.add_argument('--end_date', default=DATA_CONFIG['end_date'])
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'])
    parser.add_argument(
        '--patience',
        type=int,
        default=TRAINING_CONFIG['early_stopping_patience'],
    )
    parser.add_argument(
        '--sample_rate',
        type=float,
        default=None,
        help='每个数据文件的随机采样比例，范围 (0, 1]；默认使用全量数据',
    )
    parser.add_argument('--output_dir', default='./artifacts')
    args = parser.parse_args()

    if args.sample_rate is not None and not 0 < args.sample_rate <= 1:
        parser.error('--sample_rate 必须位于 (0, 1] 范围内')
    for name in ('batch_size', 'epochs', 'patience'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} 必须是正整数')
    
    print("=== DeepFM CTR训练开始 ===")
    
    # 1. 数据准备
    print("1. 加载数据...")
    processor = DataProcessor(
        data_dir=args.data_dir,
        start_date=args.start_date,
        split_date=args.split_date,
        end_date=args.end_date,
        val_ratio=DATA_CONFIG['val_ratio'],
        seed=DATA_CONFIG['seed'],
    )
    train, val, test = processor.load_and_preprocess(
        sample_rate=args.sample_rate
    )
    if args.sample_rate is not None:
        print(f"采样比例: {args.sample_rate:.2%}")
    print(f"训练集: {len(train):,}, 验证集: {len(val):,}, 测试集: {len(test):,}")
    
    # 2. 构建模型
    print("2. 构建模型...")
    # 获取类别特征词汇表
    vocabularies = processor.vocabularies
    
    builder = DeepFMBuilder(
        categorical_features=processor.cat_features,
        numeric_features=processor.num_features,
        embed_dim=MODEL_CONFIG['embed_dim'],
        dnn_hidden_units=MODEL_CONFIG['dnn_hidden_units'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        learning_rate=MODEL_CONFIG['learning_rate'],
        seed=DATA_CONFIG['seed'],
    )
    model = builder.build_model(vocabularies)
    model.summary()
    
    # 3. 训练
    print("3. 开始训练...")
    trainer = ModelTrainer(
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    x_train = processor.make_model_inputs(train)
    y_train = processor.make_labels(train)
    x_val = processor.make_model_inputs(val)
    y_val = processor.make_labels(val)
    history = trainer.train(x_train, y_train, x_val, y_val)
    
    # 4. 评估
    print("4. 评估模型...")
    test_groups = processor.make_groups(test)
    if test_groups is None:
        print(f"⚠️ 测试集缺少 {GROUP_COL}，跳过 GAUC、NDCG 和冷启动 GAUC")
    cold_start_masks = processor.make_cold_start_masks(train, test)
    metrics = trainer.evaluate(
        processor.make_model_inputs(test),
        processor.make_labels(test),
        test_groups,
        segment_masks=cold_start_masks,
    )
    print(
        "测试集: "
        f"AUC={metrics['auc']:.4f}, "
        f"GAUC={metrics['gauc']:.4f}, "
        f"LogLoss={metrics['logloss']:.4f}, "
        f"NDCG@10={metrics['ndcg_at_10']:.4f}"
    )
    print(
        "CTR: "
        f"预测均值={metrics['predicted_ctr_mean']:.6f}, "
        f"真实值={metrics['actual_ctr']:.6f}"
    )
    print(
        "冷启动: "
        f"新用户GAUC={metrics['new_user_gauc']:.4f} "
        f"(样本数={int(cold_start_masks['new_user'].sum()):,}), "
        f"老用户GAUC={metrics['old_user_gauc']:.4f} "
        f"(样本数={int(cold_start_masks['old_user'].sum()):,}), "
        f"新文章GAUC={metrics['new_article_gauc']:.4f} "
        f"(样本数={int(cold_start_masks['new_article'].sum()):,})"
    )
    print(f"Embedding参数量: {metrics['embedding_params']:,}")
    
    # 5. 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"deepfm_{args.split_date}.keras")
    model.save(model_path)
    artifact_path = os.path.join(
        args.output_dir, f"preprocessor_{args.split_date}.json"
    )
    processor.save_artifact(artifact_path)
    print(f"模型已保存: {model_path}")
    print(f"预处理 artifact 已保存: {artifact_path}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""只负责模型训练与评估。"""
from typing import Dict, Mapping, Optional

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding


class ModelTrainer:
    def __init__(
        self,
        model,
        batch_size,
        epochs,
        patience,
    ):
        for name, value in {
            'batch_size': batch_size,
            'epochs': epochs,
            'patience': patience,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def train(self, x_train, y_train, x_val, y_val, callbacks=None):
        """训练模型；输入必须已由 DataProcessor 完成转换。"""
        if callbacks is None:
            callbacks = [EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=self.patience,
                restore_best_weights=True,
            )]
        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=list(callbacks),
            verbose=1,
        )

    def evaluate(
        self,
        x_test,
        y_test,
        groups,
        segment_masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """计算全局、组内排序、概率质量和冷启动分群指标。"""
        y_test = np.asarray(y_test).reshape(-1)
        predictions = np.asarray(
            self.model.predict(x_test, verbose=0)
        ).reshape(-1)
        if len(y_test) != len(predictions):
            raise ValueError("y_test 与 predictions 的长度必须一致")
        auc = roc_auc_score(y_test, predictions)
        grouped = (
            self._prepare_grouped_arrays(y_test, predictions, groups)
            if groups is not None
            else None
        )
        gauc = self._calculate_grouped_gauc(*grouped) if grouped else float('nan')
        ndcg_at_10 = (
            self._calculate_grouped_ndcg(*grouped, k=10)
            if grouped
            else float('nan')
        )
        metrics = {
            'auc': auc,
            'gauc': gauc,
            'logloss': log_loss(
                y_test,
                np.clip(predictions, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            ),
            'ndcg_at_10': ndcg_at_10,
            'predicted_ctr_mean': float(np.mean(predictions)),
            'actual_ctr': float(np.mean(y_test)),
            'new_user_gauc': float('nan'),
            'old_user_gauc': float('nan'),
            'new_article_gauc': float('nan'),
            'embedding_params': self._count_embedding_params(),
        }
        if segment_masks:
            for segment in ('new_user', 'old_user', 'new_article'):
                mask = segment_masks.get(segment)
                if mask is None:
                    continue
                mask = np.asarray(mask, dtype=bool).reshape(-1)
                if len(mask) != len(y_test):
                    raise ValueError(f"{segment} mask 与 y_test 的长度必须一致")
                if groups is not None and mask.any():
                    group_array = np.asarray(groups).reshape(-1)
                    segment_grouped = self._prepare_grouped_arrays(
                        y_test[mask], predictions[mask], group_array[mask]
                    )
                    metrics[f'{segment}_gauc'] = self._calculate_grouped_gauc(
                        *segment_grouped
                    )
        return metrics

    def _count_embedding_params(self) -> int:
        """统计 Linear 和 FM/DNN 路径中所有类别 Embedding 表参数。"""
        return int(sum(
            layer.count_params()
            for layer in self.model.layers
            if isinstance(layer, Embedding)
        ))

    @staticmethod
    def _prepare_grouped_arrays(labels, preds, groups):
        """同步按组排序，并一次性生成所有连续分组的边界。"""
        labels = np.asarray(labels).reshape(-1)
        preds = np.asarray(preds).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        if not (len(labels) == len(preds) == len(groups)):
            raise ValueError("labels、preds 和 groups 的长度必须一致")
        if len(groups) == 0:
            return labels, preds, np.array([0], dtype=np.int64)

        order = np.argsort(groups, kind='stable')
        sorted_groups = groups[order]
        boundaries = np.concatenate((
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1,
            np.array([len(sorted_groups)], dtype=np.int64),
        ))
        return labels[order], preds[order], boundaries

    @staticmethod
    def _calculate_grouped_ndcg(labels, preds, boundaries, k=10):
        """利用连续分组边界计算宏平均 NDCG@k。"""
        scores = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            group_labels = labels[start:stop]
            group_preds = preds[start:stop]
            group_size = stop - start
            if group_size < 2 or np.sum(group_labels) <= 0:
                continue
            limit = min(k, group_size)
            discounts = 1.0 / np.log2(np.arange(2, limit + 2))
            ranked = group_labels[
                np.argsort(-group_preds, kind='stable')[:limit]
            ]
            ideal = np.sort(group_labels)[::-1][:limit]
            dcg = np.sum((np.power(2.0, ranked) - 1.0) * discounts)
            idcg = np.sum((np.power(2.0, ideal) - 1.0) * discounts)
            scores.append(dcg / idcg)
        return float(np.mean(scores)) if scores else float('nan')

    @classmethod
    def _calculate_ndcg(cls, labels, preds, groups, k=10):
        """兼容独立调用：按组计算宏平均 NDCG@k。"""
        grouped = cls._prepare_grouped_arrays(labels, preds, groups)
        return cls._calculate_grouped_ndcg(*grouped, k=k)

    @staticmethod
    def _calculate_grouped_gauc(labels, preds, boundaries):
        """利用连续分组边界计算按有效组样本数加权的 GAUC。"""
        total_auc, total_cnt = 0.0, 0
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            group_labels = labels[start:stop]
            group_size = stop - start
            if group_size < 2 or len(np.unique(group_labels)) < 2:
                continue
            total_auc += (
                roc_auc_score(group_labels, preds[start:stop]) * group_size
            )
            total_cnt += group_size
        return total_auc / total_cnt if total_cnt > 0 else float('nan')

    def _calculate_gauc(self, labels, preds, groups):
        """兼容独立调用：使用连续分组边界计算加权 GAUC。"""
        grouped = self._prepare_grouped_arrays(labels, preds, groups)
        return self._calculate_grouped_gauc(*grouped)
    

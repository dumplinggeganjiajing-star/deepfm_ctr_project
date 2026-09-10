# deepfm_ctr/__init__.py
"""
DeepFM CTR 排序模型包
"""

from .config import DATA_CONFIG
from .DataProcess import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL,
    DataProcessor, FeatureSpec, DerivedFeatureSpec, VocabularySpec,
)
from .model import DeepFMBuilder, FMLayer, LinearFieldSum, NumericFeatureEmbedding
from .trainer import ModelTrainer

__version__ = "1.0.0"
__all__ = [
    'DATA_CONFIG',
    'CATEGORICAL_FEATURES', 
    'NUMERIC_FEATURES',
    'LABEL',
    'DataProcessor',
    'FeatureSpec',
    'DerivedFeatureSpec',
    'VocabularySpec',
    'DeepFMBuilder',
    'FMLayer',
    'LinearFieldSum',
    'NumericFeatureEmbedding',
    'ModelTrainer',
]

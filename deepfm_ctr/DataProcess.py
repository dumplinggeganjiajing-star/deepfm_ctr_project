# -*- coding: utf-8 -*-
"""数据加载与预处理"""
import glob
import json
import os
import pandas as pd
import numpy as np
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict, Any, Mapping


CATEGORICAL_FEATURES = [
    "is_logined", "article_type", "has_new_article", "sourceId",
    "gender", "day_of_week", "theme_type", "current_hour",
    "is_double_column", "is_in_click_seq", "is_in_play_seq",
    "is_in_follow_seq", "indexno_bucket",
]

NUMERIC_FEATURES = [
    "hotArticle", "rrfScore", "exposure_hour_diff", "effective_exposure_3d",
    "article_click_user_rate_14d", "age", "article_like_cnt_all",
    "article_length", "theme_like_avg_quantity_7d", "effective_exposure_7d",
    "article_click_7d", "article_click_user_rate_3d",
    "article_click_user_rate_1d", "article_expo_user_1d",
    "article_click_rate_14d", "theme_click_rate_7d",
    "theme_avg_content_length", "theme_click_user_rate_7d",
    "theme_follow_user_cnt_1d", "user_theme_page_duration_1d",
    "theme_video_rate_90d", "theme_avg_click_cnt_1d",
    "article_page_duration_14d", "article_exposure_1d",
    "article_click_rate_3d", "theme_follow_user_cnt_7d",
    "article_author_fans_cnt", "theme_reply_quantity_1d",
    "article_click_3d", "article_expo_cnt_14d", "theme_avg_click_cnt_7d",
    "article_click_rate_7d", "user_theme_click_cnt_14d",
    "theme_expo_cnt_1d", "community_exposure_3d",
    "distinct_article_exposure_3d", "article_page_duration_3d",
    "theme_click_cnt_1d", "article_click_1d", "theme_like_avg_quantity_1d",
    "days_since_last_click", "user_theme_exposure_rate_14d",
    "article_click_user_rate_7d", "theme_click_user_7d",
    "article_content_duration_14d", "article_avg_page_duration_7d",
    "article_expo_user_14d", "theme_content_duration_14d",
    "article_exposure_3d", "user_theme_game_duration_7d",
    "user_theme_click_duration_1d", "article_expo_cnt_7d",
    "user_theme_game_duration_3d", "article_reply_user_quantity_14d",
]

LABEL = "ctr_label"
GROUP_COL = "userid"
EVAL_GROUP_COL = "__eval_userid"
DATE_COL = "part_date"
USER_ID_COL = "userid"
ARTICLE_ID_COL = "article_id"
THEME_ID_COL = "theme_id"


OOV_ID = 0
MISSING_ID = 1
FIRST_CATEGORY_ID = 2
ARTIFACT_VERSION = 6


@dataclass(frozen=True)
class FeatureSpec:
    """单个特征的稳定定义，供预处理和模型构建共同使用。"""

    name: str
    kind: str
    dtype: str
    valid_range: Optional[Tuple[float, float]] = None
    oov_id: Optional[int] = None
    missing_id: Optional[int] = None


@dataclass(frozen=True)
class DerivedFeatureSpec:
    """声明一个原始字段到模型字段的派生规则。"""

    name: str
    source: str
    rule: str
    default: Any


@dataclass(frozen=True)
class VocabularySpec:
    """一个类别特征的完整 embedding 词汇契约。"""

    name: str
    size: int
    oov_id: int
    missing_id: int
    first_category_id: int


_CATEGORICAL_VALID_RANGES = {
    'gender': (0, 2), 'is_logined': (0, 1),
    'has_new_article': (0, 1), 'is_double_column': (0, 1),
    'is_in_click_seq': (0, 1), 'is_in_play_seq': (0, 1),
    'is_in_follow_seq': (0, 1), 'current_hour': (0, 23),
    'day_of_week': (0, 6), 'article_type': (0, 10),
    'theme_type': (0, 10), 'sourceId': (0, 100),
    'indexno_bucket': (0, 25),
}


DERIVED_FEATURE_SPECS: Mapping[str, DerivedFeatureSpec] = {
    'is_double_column': DerivedFeatureSpec(
        name='is_double_column',
        source='pagename',
        rule='double_page_to_binary',
        default=0,
    ),
    'has_new_article': DerivedFeatureSpec(
        name='has_new_article',
        source='newArticle',
        rule='numeric_presence_to_binary',
        default=0,
    ),
    'indexno_bucket': DerivedFeatureSpec(
        name='indexno_bucket',
        source='indexno',
        rule='position_bucket',
        default=None,
    ),
}


FEATURE_SPECS: Mapping[str, FeatureSpec] = {
    **{
        name: FeatureSpec(
            name=name,
            kind=(
                'entity_categorical'
                if name == USER_ID_COL
                else 'position_categorical'
                if name == 'indexno_bucket'
                else 'categorical'
            ),
            dtype='int32',
            valid_range=_CATEGORICAL_VALID_RANGES.get(name),
            oov_id=OOV_ID,
            missing_id=MISSING_ID,
        )
        for name in CATEGORICAL_FEATURES
    },
    **{
        name: FeatureSpec(name=name, kind='numeric', dtype='float32')
        for name in NUMERIC_FEATURES
    },
}


# ============================================================
# 辅助函数：批量检查和修复特征
# ============================================================
def check_and_fix_features(df, categorical_features, numeric_features):
    """
    批量检查和修复特征中的非法值
    
    Args:
        df: DataFrame
        categorical_features: 类别特征列表
        numeric_features: 数值特征列表
    
    Returns:
        修复后的 DataFrame
    """
    print("="*60)
    print("批量特征检查与修复")
    print("="*60)
    
    # 1. 检查数值特征中的异常值
    print("\n1. 检查数值特征:")
    for col in numeric_features:
        if col not in df.columns:
            continue
        
        invalid_mask = (
            (df[col] == -2147483648) | 
            (df[col] == -999999999) | 
            (df[col] == 999999999) |
            pd.isna(df[col])
        )
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"   ⚠️ {col}: 有 {invalid_count} 个非法值 ({invalid_count/len(df)*100:.2f}%)")
            df.loc[invalid_mask, col] = 0
            print(f"   ✅ 修复 {col}: 替换了 {invalid_count} 个非法值为 0")
    
    # 2. 检查类别特征
    print("\n2. 检查类别特征:")
    for col in categorical_features:
        if col not in df.columns:
            print(f"   ⚠️ {col}: 列不存在，创建默认值 0")
            df[col] = 0
            continue
        
        invalid_mask = (
            (df[col] == -2147483648) |
            (df[col] == -999999999) |
            (df[col] == 999999999) |
            pd.isna(df[col])
        )
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"   ⚠️ {col}: 发现 {invalid_count} 个非法值")
            df.loc[invalid_mask, col] = 0
            print(f"   ✅ 修复 {col}: 替换了 {invalid_count} 个非法值为 0")
    
    # 3. 特殊处理：检查分类特征的取值范围
    print("\n3. 检查分类特征取值范围:")
    for col in categorical_features:
        if col not in df.columns:
            continue

        spec = FEATURE_SPECS.get(col)
        if spec is not None and spec.valid_range is not None:
            min_val, max_val = spec.valid_range
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                print(f"   ⚠️ {col}: 有 {out_of_range.sum()} 个值超出范围 [{min_val}, {max_val}]")
                df.loc[out_of_range, col] = min_val
                print(f"   ✅ 修复 {col}: 将超出范围的值设为 {min_val}")
    
    # 4. 优化数据类型
    print("\n4. 优化数据类型:")
    for col in categorical_features:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int32)
            except:
                df[col] = df[col].astype(str).map(lambda x: hash(x) % 1000).astype(np.int32)
    
    print("\n✅ 批量检查与修复完成！")
    return df


def diagnose_features(df, categorical_features, numeric_features):
    """
    诊断特征状态（用于在训练前检查）
    """
    print("\n" + "="*60)
    print("特征诊断报告")
    print("="*60)
    
    print(f"\n总样本数: {len(df):,}")
    
    print(f"\n类别特征 ({len(categorical_features)} 个):")
    for col in categorical_features:
        if col in df.columns:
            unique_vals = df[col].unique()
            min_val = df[col].min() if len(unique_vals) > 0 else 'N/A'
            max_val = df[col].max() if len(unique_vals) > 0 else 'N/A'
            null_count = df[col].isna().sum()
            invalid_count = (df[col] == -2147483648).sum()
            print(f"   {col:20s}: unique={len(unique_vals):4d}, range=[{min_val}, {max_val}], null={null_count:5d}, invalid={invalid_count:5d}")
        else:
            print(f"   {col:20s}: ❌ 列不存在")
    
    print(f"\n数值特征 ({len(numeric_features)} 个):")
    for col in numeric_features:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            null_count = df[col].isna().sum()
            invalid_count = (df[col] == -2147483648).sum()
            print(f"   {col:20s}: range=[{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}, null={null_count:5d}, invalid={invalid_count:5d}")
        else:
            print(f"   {col:20s}: ❌ 列不存在")


# ============================================================
# DataProcessor 类
# ============================================================
class DataProcessor:
    def __init__(self, data_dir: str, start_date: str, split_date: str, end_date: str,
                 val_ratio: float = 0.1, seed: int = 42):
        self.data_dir = data_dir
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = end_date
        self.val_ratio = val_ratio
        self.seed = seed
        
        # 预处理参数
        self._cat_maps = {}
        self.num_stats = {}
        self._vocabularies = {}
        self._is_fitted = False
        self.feature_specs = FEATURE_SPECS
        self.cat_features = tuple(CATEGORICAL_FEATURES)
        self.num_features = tuple(NUMERIC_FEATURES)

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        """返回 fit() 生成的 Embedding.input_dim，不允许外部推算。"""
        if not self._is_fitted:
            raise RuntimeError("DataProcessor 尚未 fit，无法读取 vocab_sizes")
        return {name: spec.size for name, spec in self._vocabularies.items()}

    @property
    def vocabularies(self) -> Dict[str, VocabularySpec]:
        """返回 OOV/MISSING ID 与大小配套的不可变词汇契约。"""
        if not self._is_fitted:
            raise RuntimeError("DataProcessor 尚未 fit，无法读取 vocabularies")
        return dict(self._vocabularies)

    @property
    def cat_maps(self) -> Dict[str, Dict[str, int]]:
        """返回词汇映射副本，防止调用方绕过 fit() 修改拟合状态。"""
        if not self._is_fitted:
            raise RuntimeError("DataProcessor 尚未 fit，无法读取 cat_maps")
        return {name: dict(mapping) for name, mapping in self._cat_maps.items()}

    @staticmethod
    def _current_schema() -> Dict[str, Any]:
        schema = {
            'categorical_features': list(CATEGORICAL_FEATURES),
            'numeric_features': list(NUMERIC_FEATURES),
            'label': LABEL,
            'group_col': GROUP_COL,
            'date_col': DATE_COL,
            'feature_specs': {
                name: asdict(spec) for name, spec in FEATURE_SPECS.items()
            },
            'derived_feature_specs': {
                name: asdict(spec) for name, spec in DERIVED_FEATURE_SPECS.items()
            },
        }
        # JSON 会把 tuple 转成 list；在保存前统一成 JSON 数据模型，保证
        # 内存中的当前 schema 与重新加载后的 schema 可直接严格比较。
        return json.loads(json.dumps(schema, ensure_ascii=False))

    def save_artifact(self, filepath: str) -> None:
        """保存训练期完整预处理状态，供验证和推理复用。"""
        if not self._is_fitted:
            raise RuntimeError("DataProcessor 尚未 fit，不能保存 artifact")

        payload = {
            'artifact_version': ARTIFACT_VERSION,
            'schema': self._current_schema(),
            'processor_config': {
                'data_dir': self.data_dir,
                'start_date': self.start_date,
                'split_date': self.split_date,
                'end_date': self.end_date,
                'val_ratio': self.val_ratio,
                'seed': self.seed,
            },
            'cat_maps': self._cat_maps,
            'vocabularies': {
                name: asdict(spec) for name, spec in self._vocabularies.items()
            },
            'num_stats': {
                name: list(stats) for name, stats in self.num_stats.items()
            },
        }

        parent = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(parent, exist_ok=True)
        temp_path = f"{filepath}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_path, filepath)

    @classmethod
    def load_artifact(cls, filepath: str) -> "DataProcessor":
        """加载并严格校验训练期预处理状态。"""
        with open(filepath, 'r', encoding='utf-8') as file:
            payload = json.load(file)

        version = payload.get('artifact_version')
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"不支持的预处理 artifact 版本: {version}; "
                f"当前版本: {ARTIFACT_VERSION}"
            )
        if payload.get('schema') != cls._current_schema():
            raise ValueError("预处理 artifact 与当前 DataProcess 特征定义不一致")

        config = payload['processor_config']
        processor = cls(**config)
        processor._cat_maps = {
            name: {str(value): int(index) for value, index in mapping.items()}
            for name, mapping in payload['cat_maps'].items()
        }
        processor._vocabularies = {
            name: VocabularySpec(**spec)
            for name, spec in payload['vocabularies'].items()
        }
        processor.num_stats = {
            name: (float(stats[0]), float(stats[1]))
            for name, stats in payload['num_stats'].items()
        }
        processor._validate_fitted_state()
        processor._is_fitted = True
        return processor

    def _validate_fitted_state(self) -> None:
        if set(self._cat_maps) != set(self._vocabularies):
            raise ValueError("artifact 中 cat_maps 与 vocabularies 特征集合不一致")
        for name, vocab in self._vocabularies.items():
            mapping = self._cat_maps[name]
            expected_size = FIRST_CATEGORY_ID + len(mapping)
            if vocab.name != name or vocab.size != expected_size:
                raise ValueError(f"artifact 中 '{name}' 的词汇表大小或名称不一致")
            if (
                vocab.oov_id != OOV_ID
                or vocab.missing_id != MISSING_ID
                or vocab.first_category_id != FIRST_CATEGORY_ID
            ):
                raise ValueError(f"artifact 中 '{name}' 的特殊类别 ID 不一致")
            if sorted(mapping.values()) != list(range(FIRST_CATEGORY_ID, vocab.size)):
                raise ValueError(f"artifact 中 '{name}' 的真实类别 ID 不连续")

    def make_model_inputs(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """把已 transform 的数据转换为命名模型输入。"""
        inputs = {}
        for name in self.cat_features:
            if name not in df.columns:
                raise ValueError(f"缺少类别特征: {name}")
            inputs[name] = df[name].to_numpy(dtype=np.int32).reshape(-1, 1)
        for name in self.num_features:
            if name not in df.columns:
                raise ValueError(f"缺少数值特征: {name}")
            inputs[name] = df[name].to_numpy(dtype=np.float32).reshape(-1, 1)
        return inputs

    @staticmethod
    def make_labels(df: pd.DataFrame) -> np.ndarray:
        """从已 transform 的训练/评估数据中提取标签。"""
        if LABEL not in df.columns:
            raise ValueError(f"缺少标签列: {LABEL}")
        return df[LABEL].to_numpy(dtype=np.float32).reshape(-1)

    @staticmethod
    def make_groups(df: pd.DataFrame) -> Optional[np.ndarray]:
        """提取 GAUC 分组 ID；数据无分组列时返回 None。"""
        if EVAL_GROUP_COL in df.columns:
            return df[EVAL_GROUP_COL].to_numpy()
        if GROUP_COL not in df.columns:
            return None
        return df[GROUP_COL].to_numpy()

    @staticmethod
    def make_cold_start_masks(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """标记评估集中训练期未见的用户和文章。

        缺失 ID 不视为一个可学习的已知实体；因此缺失 ID 对应的样本不会被
        计入冷启动分群。该方法只依赖原始标识列，不改变模型特征。
        """
        masks = {}
        for result_name, column in (
            ('new_user', USER_ID_COL),
            ('new_article', ARTICLE_ID_COL),
        ):
            train_column = (
                EVAL_GROUP_COL
                if column == USER_ID_COL and EVAL_GROUP_COL in train_df.columns
                else column
            )
            eval_column = (
                EVAL_GROUP_COL
                if column == USER_ID_COL and EVAL_GROUP_COL in eval_df.columns
                else column
            )
            if (
                train_column not in train_df.columns
                or eval_column not in eval_df.columns
            ):
                masks[result_name] = np.zeros(len(eval_df), dtype=bool)
                if result_name == 'new_user':
                    masks['old_user'] = np.zeros(len(eval_df), dtype=bool)
                continue
            known = set(train_df[train_column].dropna().astype(str))
            values = eval_df[eval_column]
            masks[result_name] = (
                values.notna() & ~values.astype(str).isin(known)
            ).to_numpy(dtype=bool)
            if result_name == 'new_user':
                masks['old_user'] = (
                    values.notna() & values.astype(str).isin(known)
                ).to_numpy(dtype=bool)
        return masks

    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 DERIVED_FEATURE_SPECS 集中执行全部派生规则。"""
        df = df.copy()
        for spec in DERIVED_FEATURE_SPECS.values():
            if spec.name in df.columns:
                continue

            if spec.source not in df.columns:
                df[spec.name] = spec.default
                continue

            if spec.rule == 'double_page_to_binary':
                source = df[spec.source]
                if source.dtype == 'object' or pd.api.types.is_string_dtype(source):
                    df[spec.name] = source.map(
                        lambda value: 1
                        if str(value).strip().lower() in {'double', '1', 'true'}
                        else 0
                    ).astype(np.int32)
                else:
                    df[spec.name] = pd.to_numeric(
                        source, errors='coerce'
                    ).fillna(spec.default).eq(1).astype(np.int32)
            elif spec.rule == 'numeric_presence_to_binary':
                numeric = pd.to_numeric(df[spec.source], errors='coerce')
                valid = numeric.notna() & np.isfinite(numeric) & ~numeric.isin(
                    {-2147483648, -999999999, 999999999}
                )
                df[spec.name] = valid.astype(np.int32)
            elif spec.rule == 'position_bucket':
                numeric = pd.to_numeric(df[spec.source], errors='coerce')
                rounded = numeric.round()
                valid = (
                    numeric.notna()
                    & np.isfinite(numeric)
                    & np.isclose(numeric, rounded, rtol=0.0, atol=1e-8)
                    & (numeric >= 0)
                )
                bucket = pd.Series(np.nan, index=df.index, dtype=np.float64)
                bucket.loc[valid & (rounded <= 20)] = rounded
                bucket.loc[valid & (rounded > 20) & (rounded <= 50)] = 21
                bucket.loc[valid & (rounded > 50) & (rounded <= 100)] = 22
                bucket.loc[valid & (rounded > 100) & (rounded <= 200)] = 23
                bucket.loc[valid & (rounded > 200) & (rounded <= 500)] = 24
                bucket.loc[valid & (rounded > 500)] = 25
                df[spec.name] = bucket
            else:
                raise ValueError(
                    f"未知派生规则: feature={spec.name}, rule={spec.rule}"
                )

        return df
    
    def load_data(self, sample_rate: Optional[float] = None) -> pd.DataFrame:
        """加载原始数据"""
        parquet_files = sorted(glob.glob(os.path.join(self.data_dir, "comm_sample_all_*.parquet")))
        if parquet_files:
            frames = [pd.read_parquet(f) for f in parquet_files]
        else:
            csv_files = sorted(glob.glob(os.path.join(self.data_dir, "video_sample_*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"未找到数据文件在 {self.data_dir}")
            frames = [pd.read_csv(f, sep="\t") for f in csv_files]
        
        if sample_rate is not None:
            frames = [d.sample(frac=sample_rate, random_state=self.seed) for d in frames]
        
        df = pd.concat(frames, axis=0, ignore_index=True)
        
        # 日期处理
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y%m%d")
        
        df = self._derive_features(df)
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        split_strategy: str = 'date',
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按日期边界切分，或按目标样本比例选择完整自然日边界。"""
        if split_strategy not in {'date', 'ratio'}:
            raise ValueError("split_strategy 必须是 'date' 或 'ratio'")
        if DATE_COL not in df.columns:
            raise ValueError(f"缺少日期列: {DATE_COL}")

        start = pd.Timestamp(self.start_date)
        split = pd.Timestamp(self.split_date)
        end = pd.Timestamp(self.end_date)

        if split_strategy == 'ratio':
            test_ratio = 1.0 - train_ratio - validation_ratio
            if not (
                0 < train_ratio < 1
                and 0 < validation_ratio < 1
                and test_ratio > 0
            ):
                raise ValueError("训练、验证、测试比例必须均大于0且总和为1")
            if start > end:
                raise ValueError("start_date 不能晚于 end_date")

            window = df[
                (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
            ].sort_values(DATE_COL, kind='stable').reset_index(drop=True)
            day = window[DATE_COL].dt.normalize()
            daily_counts = day.value_counts(sort=False).sort_index()
            if len(daily_counts) < 3:
                raise ValueError("完整日期比例切分至少需要3个不同日期")

            # 只能在自然日之间选择边界。遍历两个边界，令训练集累计比例
            # 最接近 train_ratio，训练+验证累计比例最接近二者之和。
            # 日期数通常远小于样本数，因此该搜索不会成为性能瓶颈。
            cumulative = daily_counts.cumsum().to_numpy()
            total = len(window)
            validation_end_ratio = train_ratio + validation_ratio
            best = None
            for train_days in range(1, len(daily_counts) - 1):
                train_actual = cumulative[train_days - 1] / total
                for validation_end_days in range(
                    train_days + 1, len(daily_counts)
                ):
                    validation_end_actual = (
                        cumulative[validation_end_days - 1] / total
                    )
                    error = (
                        abs(train_actual - train_ratio)
                        + abs(validation_end_actual - validation_end_ratio)
                    )
                    candidate = (error, train_days, validation_end_days)
                    if best is None or candidate < best:
                        best = candidate

            _, train_days, validation_end_days = best
            train_last_day = daily_counts.index[train_days - 1]
            validation_last_day = daily_counts.index[validation_end_days - 1]

            train = window.loc[day <= train_last_day].copy()
            val = window.loc[
                (day > train_last_day) & (day <= validation_last_day)
            ].copy()
            test = window.loc[day > validation_last_day].copy()
            if train.empty or val.empty or test.empty:
                raise ValueError("比例切分后训练、验证或测试集为空")
            return train, val, test

        if not start < split < end:
            raise ValueError("日期必须满足 start_date < split_date < end_date")
        
        train = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= split)].copy()
        test = df[(df[DATE_COL] > split) & (df[DATE_COL] <= end)].copy()
        
        train = train.sort_values(DATE_COL).reset_index(drop=True)
        
        n_val = max(1, int(len(train) * self.val_ratio))
        val = train.iloc[-n_val:].copy()
        train = train.iloc[:-n_val].copy()
        
        return train, val, test
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        核心清洗函数：清洗所有特征中的非法值和超出范围的值
        """
        df = df.copy()
        
        # ============================================================
        # 1. 清理类别特征
        # ============================================================
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue

            # 缺失/非法值不能改写为真实业务类别 0。
            sentinel_mask = (
                (df[col] == -2147483648) | 
                (df[col] == -999999999) | 
                (df[col] == 999999999)
            )
            df.loc[sentinel_mask, col] = np.nan

            # 实体 ID 只做缺失/哨兵值处理，不施加数值范围，也不进行
            # 整数化转换，避免字符串 ID 或大整数 ID 的业务语义被破坏。
            if self.feature_specs[col].kind == 'entity_categorical':
                continue

            numeric = pd.to_numeric(df[col], errors='coerce')

            # 类别特征的原始值必须是整数。pandas 1.5 的 nullable Int64
            # 会拒绝 1.5 之类的非等价转换；这里显式将其归为缺失类别，
            # 避免静默截断后改变业务语义。接近整数的浮点误差则安全归整。
            rounded = numeric.round()
            non_integer = numeric.notna() & ~np.isclose(
                numeric, rounded, rtol=0.0, atol=1e-8
            )
            if non_integer.any():
                print(
                    f"⚠️ {col}: {int(non_integer.sum())} 个非整数类别值已按缺失处理"
                )
                numeric = numeric.mask(non_integer)
            numeric = numeric.round()

            valid_range = self.feature_specs[col].valid_range
            if valid_range is not None:
                min_val, max_val = valid_range
                out_of_range = (numeric < min_val) | (numeric > max_val)
                if out_of_range.any():
                    numeric.loc[out_of_range] = np.nan

            df[col] = numeric.astype('Int64')
        
        # ============================================================
        # 2. 清理数值特征
        # ============================================================
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            
            invalid_mask = (
                (df[col] == -2147483648) | 
                (df[col] == -999999999) | 
                (df[col] == 999999999) |
                pd.isna(df[col])
            )
            if invalid_mask.any():
                df.loc[invalid_mask, col] = 0
        
        return df
    
    def fit(self, train_df: pd.DataFrame) -> "DataProcessor":
        """仅使用训练集拟合词汇表、vocab_size 和数值统计量。"""
        # ============================================================
        # 1. 清洗训练数据
        # ============================================================
        train_df_clean = self._clean_data(self._derive_features(train_df))
        cat_maps = {}
        vocabularies = {}
        num_stats = {}
        
        # ============================================================
        # 2. 构建类别编码映射（真实类别从 FIRST_CATEGORY_ID 开始）
        # ============================================================
        print("\n" + "="*60)
        print("构建类别编码映射")
        print("="*60)
        
        for c in CATEGORICAL_FEATURES:
            if c in train_df_clean.columns:
                # 获取所有唯一值
                vals = train_df_clean[c].dropna().astype(str).unique()
                # 排序
                vals = sorted(vals)
                
                # 0=OOV，1=MISSING，真实类别（包括原始值 0）从 2 开始。
                cat_maps[c] = {
                    v: i for i, v in enumerate(vals, start=FIRST_CATEGORY_ID)
                }
                vocabularies[c] = VocabularySpec(
                    name=c,
                    size=FIRST_CATEGORY_ID + len(cat_maps[c]),
                    oov_id=OOV_ID,
                    missing_id=MISSING_ID,
                    first_category_id=FIRST_CATEGORY_ID,
                )
                
                # 打印诊断信息
                unique_vals = sorted(train_df_clean[c].dropna().unique())
                print(f"✅ {c:20s}: vocab_size = {vocabularies[c].size:3d}, unique = {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
            else:
                print(f"⚠️ {c:20s}: 不存在")
        
        # ============================================================
        # 3. 计算数值特征的统计量
        # ============================================================
        print("\n" + "="*60)
        print("计算数值特征统计量")
        print("="*60)
        
        for c in NUMERIC_FEATURES:
            if c in train_df_clean.columns:
                s = pd.to_numeric(train_df_clean[c], errors='coerce')
                lo, hi = s.quantile(0.001), s.quantile(0.999)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = s.min(), s.max()
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
                elif hi <= lo:
                    # 常量特征仍需保留非零缩放区间，避免 transform 除零。
                    hi = lo + 1.0
                num_stats[c] = (float(lo), float(hi))
                print(f"✅ {c:20s}: stats = [{lo:.4f}, {hi:.4f}]")

        # 所有参数成功计算后再一次性发布，避免拟合失败留下半成品状态。
        self._cat_maps = cat_maps
        self._vocabularies = vocabularies
        self.num_stats = num_stats
        self._validate_fitted_state()
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用预处理（清洗 + 编码 + 归一化）
        """
        if not self._is_fitted:
            raise RuntimeError("DataProcessor 尚未 fit，不能执行 transform")

        df = self._derive_features(df)
        
        # ============================================================
        # Step 1: 清洗数据
        # ============================================================
        df = self._clean_data(df)

        # 保留原始用户实体键，专供 GAUC/NDCG 和新老用户分群使用；该列
        # 不在模型特征列表中，也不会由 make_model_inputs() 传入模型。
        if GROUP_COL in df.columns:
            df[EVAL_GROUP_COL] = df[GROUP_COL].copy()
        
        # ============================================================
        # Step 2: 类别特征编码
        # ============================================================
        for c in CATEGORICAL_FEATURES:
            if c not in df.columns:
                continue
            
            if c in self._cat_maps:
                missing_mask = df[c].isna()
                encoded = df[c].astype(str).map(self._cat_maps[c]).fillna(OOV_ID)
                encoded.loc[missing_mask] = MISSING_ID
                df[c] = encoded.astype(np.int32)
                vocab = self._vocabularies[c]
                if ((df[c] < 0) | (df[c] >= vocab.size)).any():
                    raise ValueError(
                        f"类别特征 '{c}' 的编码超出词汇表范围 [0, {vocab.size})"
                    )
            else:
                raise RuntimeError(f"类别特征 '{c}' 尚未拟合词汇表")
        
        # ============================================================
        # Step 3: 数值特征归一化
        # ============================================================
        for c in NUMERIC_FEATURES:
            if c not in df.columns:
                continue
            
            if c in self.num_stats:
                lo, hi = self.num_stats[c]
                v = pd.to_numeric(df[c], errors='coerce').fillna(lo).clip(lo, hi)
                df[c] = ((v - lo) / (hi - lo)).astype(np.float32)
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
        
        # ============================================================
        # Step 4: 标签处理
        # ============================================================
        if LABEL in df.columns:
            df[LABEL] = pd.to_numeric(df[LABEL], errors='coerce').fillna(0).astype(np.float32)
        
        return df
    
    def load_and_preprocess(
        self,
        sample_rate: Optional[float] = None,
        split_strategy: str = 'date',
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """一站式加载和预处理"""
        df = self.load_data(sample_rate)
        train, val, test = self.split_data(
            df,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        
        # 拟合并转换训练集
        self.fit(train)
        train = self.transform(train)
        val = self.transform(val)
        test = self.transform(test)
        
        return train, val, test

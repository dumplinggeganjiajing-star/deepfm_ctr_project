# -*- coding: utf-8 -*-
"""DeepFM模型定义"""
import tensorflow as tf
from tensorflow.keras import Input, Model as KerasModel
from tensorflow.keras.layers import (
    Dense, Embedding, Flatten, Concatenate, Dropout, Activation, Add
)
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="FMLayer")
class FMLayer(tf.keras.layers.Layer):
    """FM二阶交叉层"""
    def call(self, inputs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)
        return 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)


@register_keras_serializable(package="FMLayer")
class LinearFieldSum(tf.keras.layers.Layer):
    """在 Keras Functional 图内对各类别的一阶权重求和。"""

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1, keepdims=False)


@register_keras_serializable(package="FMLayer")
class NumericFeatureEmbedding(tf.keras.layers.Layer):
    """用连续特征值缩放一个可训练字段向量。"""

    def __init__(self, output_dim, embeddings_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer
        )

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='embedding',
            shape=(1, self.output_dim),
            initializer=self.embeddings_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # (batch, 1) -> (batch, 1, output_dim)
        return tf.expand_dims(inputs, axis=-1) * self.embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'embeddings_initializer': tf.keras.initializers.serialize(
                self.embeddings_initializer
            ),
        })
        return config


class DeepFMBuilder:
    def __init__(self, categorical_features, numeric_features, 
                 embed_dim=8, dnn_hidden_units=(128, 128), 
                 dropout_rate=0.3, learning_rate=1e-3, seed=42):
        self.cat_features = categorical_features
        self.num_features = numeric_features
        self.embed_dim = embed_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.seed = seed
    
    def _build_embedding_dict(self, vocabularies, linear=False):
        """
        构建embedding层字典
        
        Args:
            vocabularies: DataProcess.fit() 生成的词汇契约字典
            linear: 是否为 Linear 部分（True: dim=1, False: dim=embed_dim）
        """
        emb_dict = {}
        for name, vocab in vocabularies.items():
            dim = 1 if linear else self.embed_dim
            prefix = "linear_" if linear else "embed_" # 符合Keras层命名规范
            emb_dict[name] = Embedding(
                input_dim=vocab.size,
                output_dim=dim,
                embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.001, seed=self.seed),
                name=f'{prefix}_{name}'
            )
        return emb_dict
    
    def build_model(self, vocabularies):
        """构建完整的DeepFM模型"""
        expected = set(self.cat_features)
        actual = set(vocabularies)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"词汇表特征不匹配: missing={missing}, extra={extra}")

        for name, vocab in vocabularies.items():
            if vocab.name != name:
                raise ValueError(f"词汇表名称不匹配: key={name}, spec={vocab.name}")
            if not (0 <= vocab.oov_id < vocab.size):
                raise ValueError(f"{name} 的 OOV ID 超出词汇表范围")
            if not (0 <= vocab.missing_id < vocab.size):
                raise ValueError(f"{name} 的 MISSING ID 超出词汇表范围")
            if vocab.oov_id == vocab.missing_id:
                raise ValueError(f"{name} 的 OOV 与 MISSING ID 不能相同")

        inputs = {}
        for name in self.cat_features:
            inputs[name] = Input(shape=(1,), name=name, dtype='int32')
        for name in self.num_features:
            inputs[name] = Input(shape=(1,), name=name, dtype='float32')
        
        # ✅ Linear 部分用独立的 embedding (dim=1)
        linear_emb_dict = self._build_embedding_dict(vocabularies, linear=True)
        
        # ✅ FM/DNN 部分用独立的 embedding (dim=embed_dim)
        fm_emb_dict = self._build_embedding_dict(vocabularies, linear=False)

        linear_num_dict = {
            name: NumericFeatureEmbedding(
                output_dim=1,
                embeddings_initializer=TruncatedNormal(
                    mean=0.0, stddev=0.001, seed=self.seed
                ),
                name=f'linear_num_{name}',
            )
            for name in self.num_features
        }
        fm_num_dict = {
            name: NumericFeatureEmbedding(
                output_dim=self.embed_dim,
                embeddings_initializer=TruncatedNormal(
                    mean=0.0, stddev=0.001, seed=self.seed
                ),
                name=f'embed_num_{name}',
            )
            for name in self.num_features
        }
        
        # 1. Linear 部分
        linear_embeds = []
        for name in self.cat_features:
            x = linear_emb_dict[name](inputs[name])  # (batch, 1, 1)
            linear_embeds.append(x)
        for name in self.num_features:
            x = linear_num_dict[name](inputs[name])  # (batch, 1, 1)
            linear_embeds.append(x)
        
        linear_concat = Concatenate(axis=1)(linear_embeds)  # (batch, n, 1)
        linear_logit = LinearFieldSum(name='linear_field_sum')(
            linear_concat
        )  # (batch, 1)
        
        # 2. FM 部分
        fm_embeds = []
        for name in self.cat_features:
            x = fm_emb_dict[name](inputs[name])  # (batch, 1, embed_dim)
            fm_embeds.append(x)
        for name in self.num_features:
            x = fm_num_dict[name](inputs[name])  # (batch, 1, embed_dim)
            fm_embeds.append(x)
        
        fm_concat = Concatenate(axis=1)(fm_embeds)  # (batch, n, embed_dim)
        fm_logit = FMLayer()(fm_concat)  # (batch, 1)
        
        # 3. DNN 部分
        embedded = Concatenate(axis=1)(fm_embeds)  # (batch, n, embed_dim)
        dnn_input = Flatten()(embedded)  # (batch, n * embed_dim)
        
        x = dnn_input
        for i, units in enumerate(self.dnn_hidden_units):
            x = Dense(units, activation='relu', name=f'dnn_{i}')(x)
            if i < len(self.dnn_hidden_units) - 1:
                x = Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        dnn_logit = Dense(1, use_bias=False, activation=None,
                          kernel_initializer=tf.keras.initializers.glorot_normal(self.seed),
                          name='dnn_output')(x)  # (batch, 1)
        
        # 合并三路
        final_logit = Add()([linear_logit, fm_logit, dnn_logit])  # (batch, 1)
        output = Activation('sigmoid', name='dfm_out')(final_logit)
        
        model = KerasModel(inputs=inputs, outputs=output, name='DeepFM')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

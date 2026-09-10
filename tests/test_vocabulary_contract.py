import pytest
import tensorflow as tf

from deepfm_ctr.DataProcess import VocabularySpec
from deepfm_ctr.model import DeepFMBuilder


def _builder():
    return DeepFMBuilder(['category'], ['value'])


def test_model_rejects_missing_vocabulary_feature():
    with pytest.raises(ValueError, match='missing'):
        _builder().build_model({})


def test_model_rejects_colliding_oov_and_missing_ids():
    vocabulary = VocabularySpec(
        name='category', size=3, oov_id=0, missing_id=0, first_category_id=2
    )

    with pytest.raises(ValueError, match='不能相同'):
        _builder().build_model({'category': vocabulary})


def test_model_embedding_uses_vocabulary_contract_size():
    vocabulary = VocabularySpec(
        name='category', size=5, oov_id=0, missing_id=1, first_category_id=2
    )

    model = _builder().build_model({'category': vocabulary})

    assert model.get_layer('embed__category').input_dim == 5
    assert model.get_layer('linear__category').input_dim == 5


def test_numeric_feature_enters_linear_fm_and_dnn_field_representation():
    vocabulary = VocabularySpec(
        name='category', size=5, oov_id=0, missing_id=1, first_category_id=2
    )

    model = _builder().build_model({'category': vocabulary})

    linear_numeric = model.get_layer('linear_num_value')
    fm_numeric = model.get_layer('embed_num_value')
    assert linear_numeric.output.shape[-1] == 1
    assert fm_numeric.output.shape[-1] == _builder().embed_dim
    flatten = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Flatten)
    )
    assert flatten.input.shape[1] == 2

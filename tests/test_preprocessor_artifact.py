import json

import pandas as pd
import pytest

from deepfm_ctr.DataProcess import (
    DataProcessor,
    EVAL_GROUP_COL,
    MISSING_ID,
    OOV_ID,
)


def _processor():
    return DataProcessor('unused', '20260101', '20260102', '20260103')


def test_artifact_round_trip_preserves_training_transform(tmp_path):
    processor = _processor().fit(
        pd.DataFrame({'indexno': [0, 1], 'rrfScore': [10.0, 20.0]})
    )
    path = tmp_path / 'preprocessor.json'
    processor.save_artifact(str(path))

    loaded = DataProcessor.load_artifact(str(path))
    raw = pd.DataFrame({'indexno': [2, None], 'rrfScore': [15.0, 30.0]})

    expected = processor.transform(raw)
    actual = loaded.transform(raw)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual['indexno_bucket'].tolist() == [OOV_ID, MISSING_ID]
    assert loaded.vocabularies == processor.vocabularies
    assert loaded.vocab_sizes == processor.vocab_sizes


def test_artifact_rejects_schema_drift(tmp_path):
    processor = _processor().fit(pd.DataFrame({'indexno': [0]}))
    path = tmp_path / 'preprocessor.json'
    processor.save_artifact(str(path))
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['schema']['categorical_features'].append('unexpected_feature')
    path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(ValueError, match='特征定义不一致'):
        DataProcessor.load_artifact(str(path))


def test_unfitted_processor_cannot_be_saved(tmp_path):
    with pytest.raises(RuntimeError, match='尚未 fit'):
        _processor().save_artifact(str(tmp_path / 'preprocessor.json'))


def test_constant_numeric_feature_does_not_generate_nan():
    processor = _processor().fit(pd.DataFrame({'rrfScore': [5.0, 5.0]}))

    transformed = processor.transform(pd.DataFrame({'rrfScore': [5.0, 7.0]}))

    assert transformed['rrfScore'].notna().all()
    assert transformed['rrfScore'].tolist() == [0.0, 1.0]


def test_missing_group_column_disables_gauc_instead_of_failing():
    assert _processor().make_groups(pd.DataFrame({'other': [1]})) is None


def test_version0_keeps_userid_only_as_an_evaluation_group():
    processor = _processor().fit(pd.DataFrame({
        'userid': ['user-a', 'user-b', None],
    }))

    transformed = processor.transform(pd.DataFrame({
        'userid': ['user-a', 'unseen-user', None],
    }))

    assert 'userid' not in processor.cat_features
    assert 'theme_id' not in processor.cat_features
    assert 'article_id' not in processor.cat_features
    assert 'userid' not in processor.vocabularies
    assert transformed['userid'].tolist()[:2] == ['user-a', 'unseen-user']
    assert transformed[EVAL_GROUP_COL].tolist()[:2] == [
        'user-a', 'unseen-user'
    ]
    assert processor.make_groups(transformed).tolist()[:2] == [
        'user-a', 'unseen-user'
    ]

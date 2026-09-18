import pandas as pd
import pytest

from deepfm_ctr.DataProcess import FIRST_CATEGORY_ID, MISSING_ID, OOV_ID, DataProcessor


def _processor():
    return DataProcessor('.', '20260101', '20260102', '20260103')


def test_real_zero_missing_and_oov_have_distinct_ids():
    processor = _processor()
    processor.fit(pd.DataFrame({'indexno': [0, 1, None]}))
    transformed = processor.transform(pd.DataFrame({'indexno': [0, None, 2]}))

    assert transformed['indexno_bucket'].tolist() == [FIRST_CATEGORY_ID, MISSING_ID, OOV_ID]
    assert processor.vocab_sizes['indexno_bucket'] == FIRST_CATEGORY_ID + 2


def test_out_of_range_position_is_missing_not_first_position():
    processor = _processor()
    processor.fit(pd.DataFrame({'indexno': [0, 1]}))
    transformed = processor.transform(pd.DataFrame({'indexno': [99, 0]}))

    assert transformed['indexno_bucket'].tolist() == [OOV_ID, FIRST_CATEGORY_ID]


def test_derived_feature_rule_is_owned_by_data_processor():
    processor = _processor()
    raw = pd.DataFrame({'pagename': ['double', 'single', 'TRUE', None]})

    derived = processor._derive_features(raw)

    assert derived['is_double_column'].tolist() == [1, 0, 1, 0]


def test_existing_derived_feature_takes_precedence_over_source():
    processor = _processor()
    raw = pd.DataFrame({'pagename': ['double'], 'is_double_column': [0]})

    derived = processor._derive_features(raw)

    assert derived['is_double_column'].tolist() == [0]


def test_vocab_sizes_and_transform_require_fit():
    processor = _processor()

    with pytest.raises(RuntimeError, match='尚未 fit'):
        _ = processor.vocab_sizes
    with pytest.raises(RuntimeError, match='尚未 fit'):
        processor.transform(pd.DataFrame({'indexno': [0]}))


def test_fit_owns_vocab_size_and_returns_a_defensive_copy():
    processor = _processor().fit(pd.DataFrame({'indexno': [0, 1]}))

    sizes = processor.vocab_sizes
    sizes['indexno_bucket'] = 999

    assert processor.vocab_sizes['indexno_bucket'] == FIRST_CATEGORY_ID + 2

    vocab = processor.vocabularies['indexno_bucket']
    assert vocab.size == FIRST_CATEGORY_ID + 2
    assert vocab.oov_id == OOV_ID
    assert vocab.missing_id == MISSING_ID
    assert vocab.first_category_id == FIRST_CATEGORY_ID


def test_all_missing_category_still_has_special_token_vocabulary():
    processor = _processor().fit(
        pd.DataFrame({'indexno': pd.Series([None], dtype='object')})
    )

    transformed = processor.transform(pd.DataFrame({'indexno': [None]}))

    assert processor.vocab_sizes['indexno_bucket'] == FIRST_CATEGORY_ID
    assert transformed['indexno_bucket'].tolist() == [MISSING_ID]


def test_fractional_categorical_value_is_treated_as_missing():
    processor = _processor().fit(pd.DataFrame({'indexno': [0.0, 1.0]}))

    transformed = processor.transform(pd.DataFrame({'indexno': [1.5, 1.0]}))

    assert transformed['indexno_bucket'].tolist() == [MISSING_ID, FIRST_CATEGORY_ID + 1]


def test_index_position_uses_head_preserving_long_tail_buckets():
    processor = _processor()
    derived = processor._derive_features(pd.DataFrame({
        'indexno': [0, 20, 21, 50, 51, 100, 101, 200, 201, 500, 501, 4272, None]
    }))

    assert derived['indexno_bucket'].tolist()[:12] == [
        0, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25
    ]
    assert pd.isna(derived['indexno_bucket'].iloc[-1])


def test_new_article_missingness_is_explicit_binary_feature():
    processor = _processor()
    derived = processor._derive_features(pd.DataFrame({
        'newArticle': [1.0, 0.9999, None, 'bad', -999999999]
    }))

    assert derived['has_new_article'].tolist() == [1, 1, 0, 0, 0]


def test_hot_article_is_numeric_not_categorical():
    processor = _processor()

    assert 'hotArticle' in processor.num_features
    assert 'hotArticle' not in processor.cat_features


def test_user_content_history_state_encodes_all_binary_combinations():
    processor = _processor()
    derived = processor._derive_features(pd.DataFrame({
        'is_in_click_seq':  [0, 1, 0, 1, 0, 1, 0, 1],
        'is_in_play_seq':   [0, 0, 1, 1, 0, 0, 1, 1],
        'is_in_follow_seq': [0, 0, 0, 0, 1, 1, 1, 1],
    }))

    assert derived['user_content_history_state'].tolist() == list(range(8))


def test_user_content_history_state_preserves_invalid_or_missing_as_missing():
    processor = _processor()
    derived = processor._derive_features(pd.DataFrame({
        'is_in_click_seq': [1, None, 1, 2],
        'is_in_play_seq': [1, 1, None, 0],
        'is_in_follow_seq': [0, 0, 0, 0],
    }))

    assert derived['user_content_history_state'].iloc[0] == 3
    assert derived['user_content_history_state'].iloc[1:].isna().all()


def test_user_content_history_state_is_fitted_as_one_categorical_feature():
    processor = _processor().fit(pd.DataFrame({
        'is_in_click_seq': [0, 1, 1],
        'is_in_play_seq': [0, 0, 1],
        'is_in_follow_seq': [0, 0, 0],
    }))
    transformed = processor.transform(pd.DataFrame({
        'is_in_click_seq': [0, 0, None],
        'is_in_play_seq': [0, 1, 0],
        'is_in_follow_seq': [0, 0, 0],
    }))

    assert processor.vocab_sizes['user_content_history_state'] == 5
    assert transformed['user_content_history_state'].tolist() == [
        FIRST_CATEGORY_ID, OOV_ID, MISSING_ID,
    ]

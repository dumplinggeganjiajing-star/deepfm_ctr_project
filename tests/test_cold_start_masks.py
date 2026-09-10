import numpy as np
import pandas as pd

from deepfm_ctr.DataProcess import DataProcessor


def test_cold_start_masks_compare_eval_ids_with_training_ids():
    train = pd.DataFrame({
        'userid': ['u1', 'u2'],
        'article_id': ['a1', 'a2'],
    })
    test = pd.DataFrame({
        'userid': ['u1', 'u3', None],
        'article_id': ['a3', 'a2', None],
    })

    masks = DataProcessor.make_cold_start_masks(train, test)

    np.testing.assert_array_equal(masks['new_user'], [False, True, False])
    np.testing.assert_array_equal(masks['old_user'], [True, False, False])
    np.testing.assert_array_equal(masks['new_article'], [True, False, False])


def test_cold_start_masks_use_preserved_raw_user_ids_after_encoding():
    processor = DataProcessor('unused', '20260101', '20260102', '20260103')
    train = processor.fit(pd.DataFrame({
        'userid': ['u1', 'u2'],
        'article_id': ['a1', 'a2'],
    })).transform(pd.DataFrame({
        'userid': ['u1', 'u2'],
        'article_id': ['a1', 'a2'],
    }))
    test = processor.transform(pd.DataFrame({
        'userid': ['u1', 'u3'],
        'article_id': ['a3', 'a2'],
    }))

    masks = processor.make_cold_start_masks(train, test)

    np.testing.assert_array_equal(masks['new_user'], [False, True])
    np.testing.assert_array_equal(masks['old_user'], [True, False])
    assert processor.make_groups(test).tolist() == ['u1', 'u3']

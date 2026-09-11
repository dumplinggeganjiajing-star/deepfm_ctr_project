import pandas as pd
import pytest

from deepfm_ctr.DataProcess import DataProcessor


def _processor():
    return DataProcessor('unused', '20260828', '20260830', '20260831')


def _dated_frame():
    return pd.DataFrame({
        'part_date': pd.to_datetime([
            '2026-08-28', '2026-08-29', '2026-08-30', '2026-08-31'
        ]),
        'row': [28, 29, 30, 31],
    })


def test_tune_mode_uses_split_date_only_for_validation():
    train, val, test = _processor().split_data(_dated_frame(), mode='tune')

    assert train['row'].tolist() == [28, 29]
    assert val['row'].tolist() == [30]
    assert test['row'].tolist() == [31]


def test_final_mode_adds_split_date_to_training_without_validation():
    train, val, test = _processor().split_data(_dated_frame(), mode='final')

    assert train['row'].tolist() == [28, 29, 30]
    assert val is None
    assert test['row'].tolist() == [31]


def test_split_rejects_unknown_mode():
    with pytest.raises(ValueError, match='mode'):
        _processor().split_data(_dated_frame(), mode='unknown')


def test_tune_mode_rejects_empty_validation_day():
    frame = _dated_frame().query("row != 30")

    with pytest.raises(ValueError, match='验证日'):
        _processor().split_data(frame, mode='tune')


def test_ratio_strategy_splits_sorted_window_80_10_10():
    dates = pd.date_range('2026-08-28', periods=10, freq='h')
    frame = pd.DataFrame({
        'part_date': dates[::-1],
        'row': list(range(9, -1, -1)),
    })

    train, val, test = _processor().split_data(
        frame,
        mode='tune',
        split_strategy='ratio',
        train_ratio=0.8,
        validation_ratio=0.1,
    )

    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1
    assert train["part_date"].max() <= val["part_date"].min()
    assert val["part_date"].max() <= test["part_date"].min()


def test_ratio_strategy_is_not_available_in_final_mode():
    with pytest.raises(ValueError, match='tune'):
        _processor().split_data(
            _dated_frame(), mode='final', split_strategy='ratio'
        )

import pandas as pd
import pytest

from deepfm_ctr.DataProcess import DataProcessor


def _processor():
    return DataProcessor('unused', '20260101', '20260105', '20260110')


def _frame():
    dates = pd.to_datetime([
        '2026-01-01',
        '2026-01-02',
        '2026-01-03',
        '2026-01-04',
        '2026-01-05',
        '2026-01-06',
        '2026-01-07',
        '2026-01-08',
        '2026-01-09',
        '2026-01-10',
    ])
    return pd.DataFrame({
        'part_date': dates[::-1],
        'row': list(range(9, -1, -1)),
    })


def test_ratio_split_is_chronological_80_10_10():
    train, val, test = _processor().split_data(
        _frame(),
        split_strategy='ratio',
        train_ratio=0.8,
        validation_ratio=0.1,
    )

    assert (len(train), len(val), len(test)) == (8, 1, 1)
    assert train['part_date'].max() < val['part_date'].min()
    assert val['part_date'].max() < test['part_date'].min()


def test_ratio_split_keeps_high_volume_boundary_days_intact():
    frame = pd.DataFrame({
        'part_date': pd.to_datetime(
            ['2026-01-01'] * 7
            + ['2026-01-02'] * 2
            + ['2026-01-03'] * 3
            + ['2026-01-04'] * 2
        ),
        'row': range(14),
    })

    train, val, test = _processor().split_data(
        frame,
        split_strategy='ratio',
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    split_dates = [
        set(train['part_date'].dt.normalize()),
        set(val['part_date'].dt.normalize()),
        set(test['part_date'].dt.normalize()),
    ]
    assert split_dates[0].isdisjoint(split_dates[1])
    assert split_dates[1].isdisjoint(split_dates[2])
    assert split_dates[0].isdisjoint(split_dates[2])
    assert train['part_date'].max() < val['part_date'].min()
    assert val['part_date'].max() < test['part_date'].min()


def test_ratio_split_requires_three_distinct_dates():
    frame = pd.DataFrame({
        'part_date': pd.to_datetime(
            ['2026-01-01', '2026-01-01', '2026-01-02']
        )
    })

    with pytest.raises(ValueError, match='3个不同日期'):
        _processor().split_data(frame, split_strategy='ratio')


def test_ratio_split_rejects_invalid_ratios():
    with pytest.raises(ValueError, match='比例'):
        _processor().split_data(
            _frame(),
            split_strategy='ratio',
            train_ratio=0.9,
            validation_ratio=0.2,
        )


def test_original_date_strategy_remains_available():
    frame = pd.DataFrame({
        'part_date': pd.to_datetime([
            '2026-01-01', '2026-01-02', '2026-01-05', '2026-01-06'
        ])
    })

    train, val, test = _processor().split_data(frame, split_strategy='date')

    assert len(train) + len(val) == 3
    assert len(test) == 1

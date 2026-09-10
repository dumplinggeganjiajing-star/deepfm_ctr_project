import numpy as np
import pytest

from deepfm_ctr.trainer import ModelTrainer


class FakeModel:
    def __init__(self, predictions=None):
        self.predictions = predictions
        self.fit_call = None

    def fit(self, *args, **kwargs):
        self.fit_call = (args, kwargs)
        return 'history'

    def predict(self, inputs, verbose=0):
        return self.predictions

    @property
    def layers(self):
        return []


def test_train_only_forwards_prepared_inputs_to_model():
    model = FakeModel()
    trainer = ModelTrainer(model, batch_size=16, epochs=3, patience=2)
    x_train, y_train = {'category': np.array([[2]])}, np.array([1.0])
    x_val, y_val = {'category': np.array([[0]])}, np.array([0.0])

    history = trainer.train(x_train, y_train, x_val, y_val, callbacks=[])

    assert history == 'history'
    args, kwargs = model.fit_call
    assert args[0] is x_train
    assert args[1] is y_train
    assert kwargs['validation_data'][0] is x_val
    assert kwargs['validation_data'][1] is y_val
    assert kwargs['batch_size'] == 16
    assert kwargs['epochs'] == 3


def test_evaluate_calculates_auc_and_weighted_gauc():
    model = FakeModel(np.array([[0.1], [0.9], [0.2], [0.8]]))
    trainer = ModelTrainer(model, batch_size=16, epochs=3, patience=2)

    metrics = trainer.evaluate(
        {'unused': np.zeros((4, 1))},
        np.array([0, 1, 0, 1]),
        np.array(['a', 'a', 'b', 'b']),
    )

    assert metrics['auc'] == 1.0
    assert metrics['gauc'] == 1.0
    assert metrics['logloss'] > 0
    assert metrics['ndcg_at_10'] == 1.0
    assert metrics['predicted_ctr_mean'] == pytest.approx(0.5)
    assert metrics['actual_ctr'] == pytest.approx(0.5)
    assert np.isnan(metrics['new_user_gauc'])
    assert np.isnan(metrics['old_user_gauc'])
    assert np.isnan(metrics['new_article_gauc'])
    assert metrics['embedding_params'] == 0


def test_gauc_rejects_mismatched_lengths():
    trainer = ModelTrainer(FakeModel(), batch_size=16, epochs=3, patience=2)

    with pytest.raises(ValueError, match='长度必须一致'):
        trainer._calculate_gauc([0, 1], [0.2], ['a', 'a'])


def test_evaluate_skips_gauc_without_groups():
    model = FakeModel(np.array([[0.1], [0.9]]))
    trainer = ModelTrainer(model, batch_size=16, epochs=3, patience=2)

    metrics = trainer.evaluate({}, np.array([0, 1]), groups=None)

    assert metrics['auc'] == 1.0
    assert np.isnan(metrics['gauc'])
    assert np.isnan(metrics['ndcg_at_10'])


def test_evaluate_calculates_cold_start_gauc():
    model = FakeModel(np.array([[0.1], [0.9], [0.2], [0.8]]))
    trainer = ModelTrainer(model, batch_size=16, epochs=3, patience=2)

    metrics = trainer.evaluate(
        {},
        np.array([0, 1, 0, 1]),
        np.array(['a', 'a', 'b', 'b']),
        segment_masks={
            'new_user': np.array([True, True, False, False]),
            'old_user': np.array([False, False, True, True]),
            'new_article': np.ones(4, dtype=bool),
        },
    )

    assert metrics['new_user_gauc'] == 1.0
    assert metrics['old_user_gauc'] == 1.0
    assert metrics['new_article_gauc'] == 1.0


def test_ndcg_at_10_uses_group_macro_average():
    score = ModelTrainer._calculate_ndcg(
        labels=[1, 0, 0, 1],
        preds=[0.9, 0.1, 0.1, 0.9],
        groups=['a', 'a', 'b', 'b'],
        k=10,
    )

    assert score == 1.0


def test_group_metrics_keep_labels_and_predictions_aligned_when_unsorted():
    trainer = ModelTrainer(FakeModel(), batch_size=16, epochs=3, patience=2)
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0.1, 0.2, 0.9, 0.8])
    groups = np.array(['a', 'b', 'a', 'b'])

    assert trainer._calculate_gauc(labels, predictions, groups) == 1.0
    assert trainer._calculate_ndcg(labels, predictions, groups, k=10) == 1.0


@pytest.mark.parametrize('name', ['batch_size', 'epochs', 'patience'])
def test_training_configuration_must_be_positive(name):
    values = {'batch_size': 16, 'epochs': 3, 'patience': 2}
    values[name] = 0

    with pytest.raises(ValueError, match=f'{name} 必须是正整数'):
        ModelTrainer(FakeModel(), **values)

import pandas as pd

from deepfm_ctr.DataProcess import (
    CATEGORICAL_FEATURES,
    GROUP_COL,
    LABEL,
    NUMERIC_FEATURES,
    DataProcessor,
)
from deepfm_ctr.model import DeepFMBuilder
from scripts.predict import predict


def _raw_frame(rows=4):
    data = {name: [0] * rows for name in CATEGORICAL_FEATURES}
    data.update({name: [float(i) for i in range(rows)] for name in NUMERIC_FEATURES})
    data[GROUP_COL] = [f'group-{i}' for i in range(rows)]
    data[LABEL] = [i % 2 for i in range(rows)]
    return pd.DataFrame(data)


def test_saved_model_and_artifact_work_with_prediction_entrypoint(tmp_path):
    raw = _raw_frame()
    processor = DataProcessor(
        'unused', '20260101', '20260102', '20260103'
    ).fit(raw)
    model = DeepFMBuilder(
        processor.cat_features,
        processor.num_features,
        embed_dim=2,
        dnn_hidden_units=(4,),
    ).build_model(processor.vocabularies)
    transformed = processor.transform(raw)
    model.train_on_batch(
        processor.make_model_inputs(transformed),
        processor.make_labels(transformed),
    )

    model_path = tmp_path / 'model.keras'
    artifact_path = tmp_path / 'preprocessor.json'
    input_path = tmp_path / 'input.parquet'
    output_path = tmp_path / 'nested' / 'predictions.csv'
    model.save(model_path)
    processor.save_artifact(str(artifact_path))
    raw.to_parquet(input_path)

    predict(str(model_path), str(artifact_path), str(input_path), str(output_path))

    result = pd.read_csv(output_path)
    assert result.columns.tolist() == [GROUP_COL, 'prediction']
    assert len(result) == len(raw)
    assert result['prediction'].between(0.0, 1.0).all()

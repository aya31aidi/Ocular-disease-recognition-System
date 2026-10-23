"""Exercises the real Keras code path with a tiny model shaped like the notebook's:
a nested convolutional backbone inside a Sequential, followed by a dense head."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ocular.ml.predictor import KerasPredictor, create_predictor  # noqa: E402

from .conftest import make_fundus  # noqa: E402

FORMATS = {"JPEG", "PNG"}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Conv2D(4, 3, strides=4, activation="relu", name="conv_a")(inputs)
    x = tf.keras.layers.Conv2D(8, 3, strides=4, activation="relu", name="conv_b")(x)
    backbone = tf.keras.Model(inputs, x, name="backbone")
    backbone.trainable = False

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation="softmax"),
    ])
    model.build((None, 224, 224, 3))
    path = tmp_path_factory.mktemp("model") / "tiny.h5"
    model.save(path)
    return path


def test_predicts_with_gradcam(model_path):
    predictor = KerasPredictor(model_path, FORMATS)
    result = predictor.analyze(make_fundus())

    assert not result.demo
    assert len(result.probabilities) == 8
    assert abs(sum(p["probability"] for p in result.probabilities) - 1) < 1e-3
    assert result.heatmap_jpeg and result.heatmap_jpeg[:2] == b"\xff\xd8"


def test_gradcam_pass_matches_plain_forward(model_path):
    predictor = KerasPredictor(model_path, FORMATS)
    with_cam = predictor.analyze(make_fundus(), with_heatmap=True)
    without = predictor.analyze(make_fundus(), with_heatmap=False)
    a = np.array([p["probability"] for p in with_cam.probabilities])
    b = np.array([p["probability"] for p in without.probabilities])
    assert np.allclose(a, b, atol=1e-3)
    assert without.heatmap_jpeg is None


def test_factory_prefers_real_model(model_path):
    config = {"ALLOWED_IMAGE_FORMATS": FORMATS, "DEMO_MODE": "auto", "MODEL_PATH": model_path}
    assert create_predictor(config).status["mode"] == "model"


def test_factory_falls_back_to_demo(tmp_path):
    config = {"ALLOWED_IMAGE_FORMATS": FORMATS, "DEMO_MODE": "auto", "MODEL_PATH": tmp_path / "missing.h5"}
    assert create_predictor(config).status["mode"] == "demo"

    config["DEMO_MODE"] = "off"
    with pytest.raises(FileNotFoundError):
        create_predictor(config)

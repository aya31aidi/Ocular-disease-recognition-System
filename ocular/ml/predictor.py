import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..conditions import CONDITIONS
from . import preprocessing as pp

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    key: str
    confidence: float
    probabilities: list[dict]
    image_jpeg: bytes
    heatmap_jpeg: bytes | None
    warnings: list[str] = field(default_factory=list)
    demo: bool = False
    inference_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "prediction": {
                "key": self.key,
                "label": next(p["name"] for p in self.probabilities if p["key"] == self.key),
                "confidence": round(self.confidence, 4),
            },
            "probabilities": self.probabilities,
            "warnings": self.warnings,
            "demo": self.demo,
            "inference_ms": self.inference_ms,
        }


class BasePredictor:
    demo = False

    def __init__(self, allowed_formats: set[str]):
        self.allowed_formats = allowed_formats
        self.status = {"mode": "model", "reason": None}

    def analyze(self, data: bytes, with_heatmap: bool = True) -> Prediction:
        original = pp.load_rgb(data, self.allowed_formats)
        fundus = pp.crop_to_fundus(original)
        warnings = pp.quality_warnings(original, fundus)
        display = pp.to_display(fundus)

        started = time.perf_counter()
        probs, cam = self._infer(pp.to_model_input(fundus), fundus, with_heatmap)
        elapsed = int((time.perf_counter() - started) * 1000)

        ranked = sorted(
            (
                {"key": c.key, "name": c.name, "short_name": c.short_name, "probability": round(float(p), 4)}
                for c, p in zip(CONDITIONS, probs)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )
        heatmap = pp.encode_jpeg(pp.overlay_heatmap(display, cam)) if cam is not None else None

        return Prediction(
            key=ranked[0]["key"],
            confidence=ranked[0]["probability"],
            probabilities=ranked,
            image_jpeg=pp.encode_jpeg(display),
            heatmap_jpeg=heatmap,
            warnings=warnings,
            demo=self.demo,
            inference_ms=elapsed,
        )

    def _infer(self, x: np.ndarray, fundus: np.ndarray, with_heatmap: bool):
        raise NotImplementedError


class KerasPredictor(BasePredictor):
    """Wraps the VGG19 transfer-learning model trained in the notebook."""

    def __init__(self, model_path: Path, allowed_formats: set[str]):
        super().__init__(allowed_formats)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        self.tf = tf
        self.model, keras = self._load(model_path)
        n_out = self.model.output_shape[-1]
        if n_out != len(CONDITIONS):
            raise ValueError(f"Model has {n_out} outputs, expected {len(CONDITIONS)}.")

        self._lock = threading.Lock()
        try:
            self._feature_model, self._head = self._build_gradcam(keras)
        except Exception:  # Grad-CAM is a bonus; predictions still work without it.
            log.exception("Grad-CAM setup failed, heatmaps disabled")
            self._feature_model, self._head = None, []

        # Compiled graphs run about 3x faster than eager calls; warm them up so
        # the first request does not pay the tracing cost.
        spec = [tf.TensorSpec((1, pp.MODEL_INPUT_SIZE, pp.MODEL_INPUT_SIZE, 3), tf.float32)]
        self._predict_fn = tf.function(lambda x: self.model(x, training=False), input_signature=spec)
        self._gradcam_fn = tf.function(self._gradcam_step, input_signature=spec) if self._feature_model else None
        dummy = tf.zeros(spec[0].shape)
        self._predict_fn(dummy)
        if self._gradcam_fn is not None:
            self._gradcam_fn(dummy)
        self.status["model_path"] = str(model_path)

    def _load(self, path: Path):
        try:
            return self.tf.keras.models.load_model(path, compile=False), self.tf.keras
        except Exception:
            # Keras 3 occasionally rejects legacy Keras 2 .h5 files; tf_keras reads them natively.
            try:
                import tf_keras
            except ImportError:
                raise
            log.warning("Falling back to tf_keras to load %s", path)
            return tf_keras.models.load_model(path, compile=False), tf_keras

    def _build_gradcam(self, keras):
        conv_type = keras.layers.Conv2D
        backbone = next(
            (layer for layer in self.model.layers
             if hasattr(layer, "layers") and any(isinstance(sub, conv_type) for sub in layer.layers)),
            None,
        )
        if backbone is not None:
            # Sequential([VGG19, Flatten, Dense, ...]): split at the nested backbone.
            last_conv = [layer for layer in backbone.layers if isinstance(layer, conv_type)][-1]
            feature_model = keras.Model(backbone.inputs, [last_conv.output, backbone.output])
            head = self.model.layers[self.model.layers.index(backbone) + 1:]
            return feature_model, head

        last_conv = [layer for layer in self.model.layers if isinstance(layer, conv_type)][-1]
        return keras.Model(self.model.inputs, [last_conv.output, self.model.output]), []

    def _gradcam_step(self, inputs):
        tf = self.tf
        with tf.GradientTape() as tape:
            # The VGG weights are frozen, so nothing is recorded unless the input is watched.
            tape.watch(inputs)
            conv_out, y = self._feature_model(inputs, training=False)
            for layer in self._head:
                y = layer(y, training=False)
            score = tf.gather(y[0], tf.argmax(y[0]))
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
        return y[0], cam

    def _infer(self, x, fundus, with_heatmap):
        inputs = self.tf.convert_to_tensor(x)
        with self._lock:
            if not with_heatmap or self._gradcam_fn is None:
                return self._predict_fn(inputs).numpy()[0], None
            probs, cam = self._gradcam_fn(inputs)

        cam = cam.numpy()
        peak = cam.max()
        return probs.numpy(), (cam / peak if peak > 0 else None)


class DemoPredictor(BasePredictor):
    """Stand-in used when the trained weights are not available.

    Scores are synthetic but deterministic per image, so the interface can be
    explored end to end. Every result produced here is flagged as demo output.
    """

    demo = True

    def __init__(self, allowed_formats: set[str], reason: str):
        super().__init__(allowed_formats)
        self.status = {"mode": "demo", "reason": reason}

    def _infer(self, x, fundus, with_heatmap):
        seed = int.from_bytes(hashlib.sha256(x.tobytes()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        alpha = np.full(len(CONDITIONS), 0.6)
        alpha[rng.integers(len(CONDITIONS))] = 6.0
        probs = rng.dirichlet(alpha)

        cam = None
        if with_heatmap:
            # A blurred edge map highlights vessels and the optic disc, which is
            # roughly where a real Grad-CAM tends to focus on fundus photos.
            gray = cv2.cvtColor(cv2.resize(fundus, (224, 224)), cv2.COLOR_RGB2GRAY).astype(np.float32)
            edges = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            cam = cv2.GaussianBlur(edges, (0, 0), 12)
            cam = cam / cam.max() if cam.max() > 0 else None
        return probs, cam


def create_predictor(config) -> BasePredictor:
    formats = config["ALLOWED_IMAGE_FORMATS"]
    mode = config["DEMO_MODE"]
    path = Path(config["MODEL_PATH"])

    if mode == "on":
        return DemoPredictor(formats, "Demo mode is switched on in the configuration.")

    if not path.exists():
        if mode == "off":
            raise FileNotFoundError(f"Model weights not found at {path}")
        log.warning("Model weights not found at %s, running in demo mode", path)
        return DemoPredictor(formats, f"Model weights were not found at {path.name}.")

    with open(path, "rb") as fh:
        is_lfs_pointer = fh.read(40).startswith(b"version https://git-lfs")
    if is_lfs_pointer:
        message = f"{path.name} is a Git LFS pointer, not the weights; run `git lfs pull`."
        if mode == "off":
            raise RuntimeError(message)
        log.warning(message)
        return DemoPredictor(formats, message)

    try:
        predictor = KerasPredictor(path, formats)
        log.info("Loaded model from %s", path)
        return predictor
    except Exception as exc:
        if mode == "off":
            raise
        log.exception("Could not load the model, running in demo mode")
        reason = "TensorFlow is not installed." if isinstance(exc, ImportError) else f"The model failed to load: {exc}"
        return DemoPredictor(formats, reason)

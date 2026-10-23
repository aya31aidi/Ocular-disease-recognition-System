"""Small JSON API so the model can be used from scripts or other services.

    curl -F "file=@fundus.jpg" http://localhost:5000/api/v1/predict
"""

import base64

from flask import Blueprint, current_app, jsonify, request

from ..conditions import CONDITIONS
from ..ml import ImageError

bp = Blueprint("api", __name__, url_prefix="/api/v1")


@bp.get("/health")
def health():
    status = current_app.extensions["predictor"].status
    return jsonify(
        status="ok",
        mode=status["mode"],
        demo_reason=status.get("reason"),
        classes=[{"index": i, "key": c.key, "name": c.name} for i, c in enumerate(CONDITIONS)],
    )


@bp.post("/predict")
def predict():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="Send the image as multipart form data under the field 'file'."), 400

    with_heatmap = request.args.get("heatmap", "").lower() in {"1", "true", "yes"}
    try:
        prediction = current_app.extensions["predictor"].analyze(file.read(), with_heatmap=with_heatmap)
    except ImageError as exc:
        return jsonify(error=str(exc)), 422

    payload = prediction.to_dict()
    if with_heatmap and prediction.heatmap_jpeg:
        payload["heatmap_jpeg_base64"] = base64.b64encode(prediction.heatmap_jpeg).decode()
    return jsonify(payload)

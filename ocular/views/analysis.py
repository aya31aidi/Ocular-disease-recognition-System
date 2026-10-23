import base64
import json
import uuid
from collections import Counter
from pathlib import Path

from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   send_file, url_for)
from flask_login import current_user, login_required

from ..conditions import CONDITIONS_BY_KEY
from ..extensions import db
from ..ml import ImageError
from ..models import Analysis

bp = Blueprint("analysis", __name__)

EYE_SIDES = {"left": "Left eye", "right": "Right eye"}


def _data_uri(jpeg: bytes | None) -> str | None:
    return f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}" if jpeg else None


def _save_analysis(prediction, filename, eye_side, note) -> Analysis:
    analysis_id = uuid.uuid4().hex
    folder = Path(current_app.config["UPLOAD_DIR"]) / str(current_user.id)
    folder.mkdir(parents=True, exist_ok=True)

    image_rel = f"{current_user.id}/{analysis_id}.jpg"
    (folder / f"{analysis_id}.jpg").write_bytes(prediction.image_jpeg)
    heatmap_rel = None
    if prediction.heatmap_jpeg:
        heatmap_rel = f"{current_user.id}/{analysis_id}_cam.jpg"
        (folder / f"{analysis_id}_cam.jpg").write_bytes(prediction.heatmap_jpeg)

    analysis = Analysis(
        id=analysis_id,
        user_id=current_user.id,
        original_filename=filename[:255] if filename else None,
        eye_side=eye_side,
        note=note,
        predicted_key=prediction.key,
        confidence=prediction.confidence,
        probabilities_json=json.dumps(prediction.probabilities),
        warnings_json=json.dumps(prediction.warnings),
        demo=prediction.demo,
        inference_ms=prediction.inference_ms,
        image_path=image_rel,
        heatmap_path=heatmap_rel,
    )
    db.session.add(analysis)
    db.session.commit()
    return analysis


@bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html", eye_sides=EYE_SIDES)

    file = request.files.get("file")
    if file is None or not file.filename:
        flash("Choose a fundus image to analyse.", "error")
        return redirect(url_for("analysis.analyze"))

    eye_side = request.form.get("eye_side") if request.form.get("eye_side") in EYE_SIDES else None
    note = (request.form.get("note") or "").strip()[:280] or None

    try:
        prediction = current_app.extensions["predictor"].analyze(file.read())
    except ImageError as exc:
        flash(str(exc), "error")
        return redirect(url_for("analysis.analyze"))

    if current_user.is_authenticated:
        analysis = _save_analysis(prediction, file.filename, eye_side, note)
        return redirect(url_for("analysis.detail", analysis_id=analysis.id, fresh=1))

    result = {
        "id": None,
        "key": prediction.key,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "warnings": prediction.warnings,
        "demo": prediction.demo,
        "inference_ms": prediction.inference_ms,
        "image_src": _data_uri(prediction.image_jpeg),
        "heatmap_src": _data_uri(prediction.heatmap_jpeg),
        "filename": file.filename,
        "eye_side": EYE_SIDES.get(eye_side),
        "note": note,
        "created_at": None,
    }
    return render_template("result.html", result=result, condition=CONDITIONS_BY_KEY[prediction.key])


def _owned_or_404(analysis_id) -> Analysis:
    analysis = db.session.get(Analysis, analysis_id)
    if analysis is None or analysis.user_id != current_user.id:
        abort(404)
    return analysis


@bp.route("/analyses/<analysis_id>")
@login_required
def detail(analysis_id):
    a = _owned_or_404(analysis_id)
    result = {
        "id": a.id,
        "key": a.predicted_key,
        "confidence": a.confidence,
        "probabilities": a.probabilities,
        "warnings": a.warnings,
        "demo": a.demo,
        "inference_ms": a.inference_ms,
        "image_src": url_for("analysis.image", analysis_id=a.id, kind="original"),
        "heatmap_src": url_for("analysis.image", analysis_id=a.id, kind="heatmap") if a.heatmap_path else None,
        "filename": a.original_filename,
        "eye_side": EYE_SIDES.get(a.eye_side),
        "note": a.note,
        "created_at": a.created_at,
    }
    return render_template("result.html", result=result, condition=CONDITIONS_BY_KEY[a.predicted_key],
                           fresh=request.args.get("fresh") == "1")


@bp.route("/analyses/<analysis_id>/<any(original, heatmap):kind>.jpg")
@login_required
def image(analysis_id, kind):
    a = _owned_or_404(analysis_id)
    rel = a.image_path if kind == "original" else a.heatmap_path
    if not rel:
        abort(404)
    path = Path(current_app.config["UPLOAD_DIR"]) / rel
    if not path.exists():
        abort(404)
    response = send_file(path, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "private, max-age=86400"
    return response


@bp.route("/analyses/<analysis_id>/delete", methods=["POST"])
@login_required
def delete(analysis_id):
    a = _owned_or_404(analysis_id)
    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    for rel in (a.image_path, a.heatmap_path):
        if rel:
            (upload_dir / rel).unlink(missing_ok=True)
    db.session.delete(a)
    db.session.commit()
    flash("Analysis deleted.", "info")
    return redirect(url_for("analysis.dashboard"))


@bp.route("/dashboard")
@login_required
def dashboard():
    selected = request.args.get("condition")
    if selected not in CONDITIONS_BY_KEY:
        selected = None

    all_rows = current_user.analyses.with_entities(Analysis.predicted_key).all()
    counts = Counter(key for (key,) in all_rows)
    total = sum(counts.values())

    query = current_user.analyses.order_by(Analysis.created_at.desc())
    if selected:
        query = query.filter_by(predicted_key=selected)
    page = query.paginate(page=request.args.get("page", 1, type=int), per_page=12, error_out=False)

    latest = current_user.analyses.order_by(Analysis.created_at.desc()).first()
    stats = {
        "total": total,
        "flagged": total - counts.get("normal", 0),
        "most_common": counts.most_common(1)[0] if counts else None,
        "latest": latest.created_at if latest else None,
    }
    return render_template("dashboard.html", page=page, counts=counts, stats=stats, selected=selected)

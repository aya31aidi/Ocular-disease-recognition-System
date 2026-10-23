import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from .conditions import CONDITIONS, CONDITIONS_BY_KEY, URGENCY_LABELS
from .config import Config
from .extensions import csrf, db, login_manager
from .ml import create_predictor


def create_app(config_object=Config, **overrides) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.config.update(overrides)
    Config.init_instance(app.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)

    from .views.analysis import bp as analysis_bp
    from .views.api import bp as api_bp
    from .views.auth import bp as auth_bp
    from .views.pages import bp as pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(api_bp)
    csrf.exempt(api_bp)

    with app.app_context():
        from . import models  # noqa: F401  (registers tables)
        db.create_all()

    app.extensions["predictor"] = create_predictor(app.config)

    @app.context_processor
    def inject_globals():
        return {
            "predictor_status": app.extensions["predictor"].status,
            "conditions": CONDITIONS,
            "conditions_by_key": CONDITIONS_BY_KEY,
            "urgency_labels": URGENCY_LABELS,
            "current_year": datetime.now(timezone.utc).year,
        }

    @app.template_filter("pct")
    def pct(value, digits=1):
        return f"{value * 100:.{digits}f}%"

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask) -> None:
    def wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(404)
    def not_found(_):
        if wants_json():
            return jsonify(error="Not found"), 404
        return render_template("errors/error.html", code=404,
                               title="Page not found",
                               message="This page does not exist or has moved."), 404

    @app.errorhandler(413)
    def too_large(_):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        message = f"The file is larger than the {limit} MB limit."
        if wants_json():
            return jsonify(error=message), 413
        return render_template("errors/error.html", code=413,
                               title="File too large", message=message), 413

    @app.errorhandler(500)
    def server_error(_):
        if wants_json():
            return jsonify(error="Internal server error"), 500
        return render_template("errors/error.html", code=500,
                               title="Something went wrong",
                               message="The server hit an unexpected error. Please try again."), 500

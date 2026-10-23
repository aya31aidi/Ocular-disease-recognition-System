import json
from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def _utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow)

    analyses = db.relationship(
        "Analysis", backref="user", lazy="dynamic", cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else ""


class Analysis(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, index=True)

    original_filename = db.Column(db.String(255))
    eye_side = db.Column(db.String(10))
    note = db.Column(db.String(280))

    predicted_key = db.Column(db.String(40), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities_json = db.Column(db.Text, nullable=False)
    warnings_json = db.Column(db.Text, default="[]")
    demo = db.Column(db.Boolean, default=False)
    inference_ms = db.Column(db.Integer)

    # Images live in the database so history survives on hosts without a
    # persistent disk (Vercel, Render free tier). Deferred so list views stay light.
    image_jpeg = db.deferred(db.Column(db.LargeBinary))
    heatmap_jpeg = db.deferred(db.Column(db.LargeBinary))
    # Filesystem paths written by earlier versions; kept so old records still load.
    image_path = db.Column(db.String(255), nullable=False, default="")
    heatmap_path = db.Column(db.String(255))

    @property
    def has_heatmap(self) -> bool:
        return bool(self.heatmap_path) or self.heatmap_jpeg is not None

    @property
    def probabilities(self) -> list[dict]:
        return json.loads(self.probabilities_json)

    @property
    def warnings(self) -> list[str]:
        return json.loads(self.warnings_json or "[]")


class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    topic = db.Column(db.String(40))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow)

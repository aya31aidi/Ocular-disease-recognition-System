"""Behaviour that matters when the app runs somewhere other than a laptop."""

import sqlite3

import numpy as np
import pytest

from ocular import create_app
from ocular.config import TestConfig, _database_url
from ocular.ml.preprocessing import resize_linear
from ocular.models import Analysis


def test_postgres_urls_use_psycopg_driver(monkeypatch, tmp_path):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host/db?sslmode=require")
    assert _database_url(tmp_path) == "postgresql+psycopg://u:p@host/db?sslmode=require"

    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@host/db")
    assert _database_url(tmp_path) == "postgresql+psycopg://u:p@host/db"

    monkeypatch.delenv("POSTGRES_URL")
    assert _database_url(tmp_path).startswith("sqlite:///")


def test_old_sqlite_schema_is_upgraded(tmp_path):
    # Table layout written by the first version, which kept images on disk.
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE analysis (
            id VARCHAR(32) PRIMARY KEY, user_id INTEGER NOT NULL, created_at DATETIME,
            original_filename VARCHAR(255), eye_side VARCHAR(10), note VARCHAR(280),
            predicted_key VARCHAR(40) NOT NULL, confidence FLOAT NOT NULL,
            probabilities_json TEXT NOT NULL, warnings_json TEXT, demo BOOLEAN,
            inference_ms INTEGER, image_path VARCHAR(255) NOT NULL, heatmap_path VARCHAR(255))""")

    app = create_app(TestConfig, INSTANCE_DIR=tmp_path, UPLOAD_DIR=tmp_path / "uploads",
                     SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}", SECRET_KEY="test")
    with app.app_context():
        assert Analysis.query.count() == 0  # would raise if the new columns were missing
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(analysis)")}
    assert {"image_jpeg", "heatmap_jpeg"} <= columns


def test_resize_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 256, (600, 900, 3), dtype=np.uint8)
    ours = resize_linear(img, 224, 224).astype(int)
    theirs = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype(int)
    # OpenCV uses fixed-point arithmetic, so values may differ by one level.
    assert np.abs(ours - theirs).max() <= 1

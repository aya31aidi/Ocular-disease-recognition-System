import io

import numpy as np
import pytest
from PIL import Image

from ocular import create_app
from ocular.config import TestConfig
from ocular.extensions import db as _db


def make_fundus(width=600, height=480, fmt="JPEG", mode="RGB") -> bytes:
    """Synthetic fundus-like photo: orange disc with a bright optic disc on black."""
    yy, xx = np.mgrid[:height, :width]
    cy, cx, r = height / 2, width / 2, min(width, height) * 0.45
    dist = np.hypot(yy - cy, xx - cx)
    img = np.zeros((height, width, 3), np.uint8)
    disc = dist < r
    shade = (1 - dist / r * 0.5)[disc]
    img[disc] = np.stack([210 * shade, 95 * shade, 40 * shade], axis=-1).astype(np.uint8)
    optic = np.hypot(yy - cy, xx - (cx + r * 0.4)) < r * 0.15
    img[optic] = (250, 220, 150)
    buf = io.BytesIO()
    Image.fromarray(img).convert(mode).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def app(tmp_path):
    app = create_app(
        TestConfig,
        INSTANCE_DIR=tmp_path,
        UPLOAD_DIR=tmp_path / "uploads",
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'test.db'}",
        SECRET_KEY="test",
    )
    yield app
    with app.app_context():
        _db.session.remove()
        _db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def fundus_jpeg():
    return make_fundus()


@pytest.fixture
def signed_in(client):
    client.post("/signup", data={
        "name": "Test User", "email": "test@example.com",
        "password": "correct-horse", "confirm": "correct-horse",
    })
    return client

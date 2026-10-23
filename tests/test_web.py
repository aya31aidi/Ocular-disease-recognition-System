import io

import pytest

from ocular.models import Analysis, ContactMessage


@pytest.mark.parametrize("path", ["/", "/analyze", "/conditions", "/model", "/contact", "/signin", "/signup"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"RetinaLens" in response.data or b"Retina" in response.data


def test_legacy_urls_redirect(client):
    assert client.get("/services").status_code == 301
    assert client.get("/about").status_code == 301


def test_unknown_page_is_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert b"Page not found" in response.data


def test_anonymous_analysis_renders_without_saving(app, client, fundus_jpeg):
    response = client.post("/analyze", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"All classes" in response.data
    assert b"data:image/jpeg;base64" in response.data
    with app.app_context():
        assert Analysis.query.count() == 0
    assert not any((app.config["UPLOAD_DIR"]).rglob("*.jpg"))


def test_analysis_without_file_redirects(client):
    response = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert response.status_code == 302


def test_analysis_with_bad_file_shows_error(client):
    response = client.post("/analyze", data={"file": (io.BytesIO(b"garbage"), "x.jpg")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert b"could not be read" in response.data


def test_signed_in_analysis_is_saved_and_private(app, signed_in, fundus_jpeg):
    response = signed_in.post("/analyze", data={
        "file": (io.BytesIO(fundus_jpeg), "eye.jpg"), "eye_side": "left", "note": "baseline",
    }, content_type="multipart/form-data")
    assert response.status_code == 302
    detail_url = response.headers["Location"]

    with app.app_context():
        analysis = Analysis.query.one()
        assert analysis.eye_side == "left"
        assert analysis.note == "baseline"
        assert len(analysis.probabilities) == 8
        analysis_id = analysis.id

    assert signed_in.get(detail_url).status_code == 200
    image = signed_in.get(f"/analyses/{analysis_id}/original.jpg")
    assert image.status_code == 200 and image.mimetype == "image/jpeg"
    assert b"baseline" in signed_in.get("/dashboard").data

    # Another account cannot see it.
    signed_in.post("/signout")
    signed_in.post("/signup", data={"name": "Other", "email": "other@example.com",
                                    "password": "another-pass", "confirm": "another-pass"})
    assert signed_in.get(detail_url).status_code == 404
    assert signed_in.get(f"/analyses/{analysis_id}/original.jpg").status_code == 404


def test_delete_removes_record(app, signed_in, fundus_jpeg):
    signed_in.post("/analyze", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                   content_type="multipart/form-data")
    with app.app_context():
        analysis = Analysis.query.one()
        analysis_id = analysis.id
        assert analysis.image_jpeg[:2] == b"\xff\xd8"  # stored in the database, not on disk
    assert not any(app.config["UPLOAD_DIR"].rglob("*.jpg"))

    signed_in.post(f"/analyses/{analysis_id}/delete")
    with app.app_context():
        assert Analysis.query.count() == 0
    assert signed_in.get(f"/analyses/{analysis_id}/original.jpg").status_code == 404


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/signin" in response.headers["Location"]


def test_signup_validation(client):
    response = client.post("/signup", data={"name": "A", "email": "bad", "password": "short", "confirm": "x"})
    assert b"valid email" in response.data
    assert b"at least 8" in response.data


def test_signin_flow(client):
    client.post("/signup", data={"name": "Ada L", "email": "ada@example.com",
                                 "password": "analytical", "confirm": "analytical"})
    client.post("/signout")

    bad = client.post("/signin", data={"email": "ada@example.com", "password": "wrong"})
    assert b"does not match" in bad.data

    good = client.post("/signin", data={"email": "ADA@example.com", "password": "analytical"})
    assert good.status_code == 302
    assert client.get("/dashboard").status_code == 200


def test_signin_ignores_external_next(client):
    client.post("/signup", data={"name": "Ada L", "email": "ada@example.com",
                                 "password": "analytical", "confirm": "analytical"})
    client.post("/signout")
    response = client.post("/signin?next=https://evil.example/", data={
        "email": "ada@example.com", "password": "analytical"})
    assert "evil.example" not in response.headers["Location"]


def test_contact_form_saves_message(app, client):
    response = client.post("/contact", data={
        "name": "Sam", "email": "sam@example.com", "topic": "bug", "message": "The slider sticks on Safari.",
    })
    assert response.status_code == 302
    with app.app_context():
        assert ContactMessage.query.one().topic == "bug"


def test_contact_form_validation(client):
    response = client.post("/contact", data={"name": "", "email": "x", "message": "hi"})
    assert b"Please enter a name" in response.data
    assert b"at least 10" in response.data

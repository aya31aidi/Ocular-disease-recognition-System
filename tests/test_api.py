import io


def test_health(client):
    data = client.get("/api/v1/health").get_json()
    assert data["status"] == "ok"
    assert data["mode"] == "demo"
    assert [c["index"] for c in data["classes"]] == list(range(8))


def test_predict(client, fundus_jpeg):
    response = client.post("/api/v1/predict", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    data = response.get_json()
    probs = [p["probability"] for p in data["probabilities"]]
    assert len(probs) == 8
    assert abs(sum(probs) - 1) < 1e-3
    assert probs == sorted(probs, reverse=True)
    assert data["prediction"]["key"] == data["probabilities"][0]["key"]
    assert data["demo"] is True
    assert "heatmap_jpeg_base64" not in data


def test_predict_is_deterministic(client, fundus_jpeg):
    def run():
        return client.post("/api/v1/predict", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                           content_type="multipart/form-data").get_json()["probabilities"]
    assert run() == run()


def test_predict_with_heatmap(client, fundus_jpeg):
    data = client.post("/api/v1/predict?heatmap=1", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                       content_type="multipart/form-data").get_json()
    assert data["heatmap_jpeg_base64"]


def test_predict_errors(client):
    assert client.post("/api/v1/predict").status_code == 400
    bad = client.post("/api/v1/predict", data={"file": (io.BytesIO(b"nope"), "x.png")},
                      content_type="multipart/form-data")
    assert bad.status_code == 422
    assert "error" in bad.get_json()


def test_api_404_is_json(client):
    response = client.get("/api/v1/unknown")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_oversized_upload_is_json_413(app, client, fundus_jpeg):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    response = client.post("/api/v1/predict", data={"file": (io.BytesIO(fundus_jpeg), "eye.jpg")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
    assert "limit" in response.get_json()["error"]

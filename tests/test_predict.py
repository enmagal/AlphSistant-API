from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_predict_image():
    filepath='tests/files/sa1.wav'

    response = client.post(
        "/predict",files={"file":("filename", open(filepath,"rb"),"audio/wav")}
    )

    assert response.status_code == 200

def test_predict_text():
    filepath = "tests/files/fichier.txt"

    response = client.post(
        "/predict", files={"file": ("filename", open(filepath, "rb"), "text/plain")}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File provided is not an audio."}
from io import BytesIO
from typing import List
import librosa

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from .model import load_model, predict, prepare_audio
from PIL import Image
from pydantic import BaseModel


app = FastAPI()

model = load_model()


# Define the response JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []


@app.post("/predict", response_model=Prediction)
async def prediction(file: UploadFile = File(...)):

    # Ensure that the file is an image
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File provided is not an audio.")

    content = await file.read()
    audio, sample_rate = librosa.load(BytesIO(content), sr=None)

    # preprocess the image and prepare it for classification
    X = prepare_audio(audio, sample_rate, target=100)

    response = predict(X, model)

    # return the response as a JSON
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predictions": response,
    }

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
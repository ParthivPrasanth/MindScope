from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from predict import predict

app = FastAPI(title="MindScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    survey_dict: dict
    free_text: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(body: PredictRequest):
    try:
        result = predict(body.survey_dict, body.free_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

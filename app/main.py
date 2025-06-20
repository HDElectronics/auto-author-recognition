from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from app.transformer.transformer_inference import TransformerInference

app = FastAPI()

# Allow Streamlit frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

###############################
# LSTM endpoint
@app.post("/predict/lstm/")
def predict_lstm(request: PredictionRequest):
    import torch
    from app.lstm.lstm_inference import LSTMClassifier, predict, electra

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = electra.config.hidden_size
    model = LSTMClassifier(hidden_size, 16).to(device)
    # Use relative path for weights
    weights_path = os.path.join(os.path.dirname(__file__), "lstm", "weights", "checkpoint_epoch3794_valacc0.72.pt")

    if not os.path.exists(weights_path):
        raise HTTPException(status_code=500, detail=f"LSTM weights file not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Run prediction and return the integer label
    import numpy as np
    emb = predict.__globals__['get_embedding'](request.text)
    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
    return {
        "predicted_author": pred,
    }

###############################
# Transformer endpoint
@app.post("/predict/transformer/")
def predict_transformer(request: PredictionRequest):
    transformer = TransformerInference()
    author = transformer.predict_author(request.text)
    return {
        "predicted_author": author,
    }

@app.get("/")
def root():
    return {"message": "Auto Author Recognition API"}

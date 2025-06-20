# Auto Author Recognition

A web application for automatic author recognition from text using deep learning models (LSTM and Transformer). The app provides a FastAPI backend for inference and a Streamlit frontend for user interaction.

---

## Features

- **LSTM and Transformer Models**: Predicts the author of a given text using either an LSTM-based or Transformer-based classifier.
- **REST API**: FastAPI endpoints for model inference.
- **Web UI**: Streamlit frontend for uploading text, selecting models, and viewing predictions.
- **Docker Support**: Easily run the entire stack in a container.

---

## Architecture

![Architecture Diagram](architecture.dot)

- **FastAPI Backend** (`app/main.py`): Serves `/predict/lstm/` and `/predict/transformer/` endpoints.
- **LSTM Module** (`app/lstm/lstm_inference.py`): Loads LSTM model and weights for inference.
- **Transformer Module** (`app/transformer/transformer_inference.py`): Loads Transformer model and weights for inference.
- **Streamlit Frontend** (`app/streamlit_app.py`): User interface for uploading text and viewing predictions.

---

## Getting Started

### Prerequisites
- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [Docker](https://www.docker.com/)

### Installation (Local)

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd auto-author-recognition
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download/Place Model Weights:**
   - LSTM weights: `app/lstm/weights/checkpoint_epoch3794_valacc0.72.pt`
   - Transformer weights: Download the `weights` folder from the provided Google Drive link and place it inside `app/transformer/` so that you have `app/transformer/weights/` containing the model files.

4. **Run FastAPI backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. **Run Streamlit frontend (in a new terminal):**
   ```bash
   streamlit run app/streamlit_app.py
   ```

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t auto-author-recognition .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 auto-author-recognition
   ```

---

## Usage

- **Web UI:**
  - Open [http://localhost:8501](http://localhost:8501) in your browser.
  - Upload or paste text, select a model, and click "Predict Author".

- **API Endpoints:**
  - `POST /predict/lstm/` with JSON `{ "text": "..." }`
  - `POST /predict/transformer/` with JSON `{ "text": "..." }`

  Example using `curl`:
  ```bash
  curl -X POST "http://localhost:8000/predict/lstm/" -H "Content-Type: application/json" -d '{"text": "your text here"}'
  ```

---

## File Structure

```
app/
  main.py                # FastAPI app
  streamlit_app.py       # Streamlit UI
  lstm/
    lstm_inference.py    # LSTM model logic
    weights/             # LSTM model weights
  transformer/
    transformer_inference.py # Transformer model logic
    weights/                 # Transformer model files
requirements.txt
Dockerfile
architecture.dot         # Architecture diagram (Graphviz)
```

---

## Notes
- Make sure the model weights are present in the correct folders before running.
- The app uses the `aubmindlab/araelectra-base-discriminator` model for embeddings.
- For production, update CORS and security settings as needed.

---

## License
MIT License

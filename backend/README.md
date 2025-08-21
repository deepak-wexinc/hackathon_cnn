# Fraud Detection Backend

This is the FastAPI backend for the fraud detection system using visual anomaly detection.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create directories for genuine documents:
```bash
mkdir genuine_documents
```

4. Add genuine documents to the `genuine_documents` folder (optional - the system will create synthetic training data if none are provided).

## Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check and model status
- `POST /analyze` - Analyze uploaded documents for fraud detection
- `POST /retrain` - Retrain the model with current genuine documents

## How it Works

1. The system uses a convolutional autoencoder neural network
2. It's trained on genuine documents to learn normal patterns
3. When analyzing new documents, it calculates reconstruction error
4. Documents with high reconstruction error are flagged as potential fraud
5. The system provides confidence scores and detailed analysis results

## Supported File Formats

- PDF files
- Image files: JPG, JPEG, PNG, BMP, TIFF

## Notes

- The system requires Poppler for PDF processing. Install it based on your OS:
  - Windows: Download from https://poppler.freedesktop.org/
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`
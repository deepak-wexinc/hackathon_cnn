from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from typing import List
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config
IMAGE_SIZE = (256, 256)
GENUINE_DOCS_PATH = os.path.join(os.getcwd(), 'genuine_documents')

class FraudDetector:
    def __init__(self):
        self.model = None
        self.threshold = None
        self.is_trained = False

    def load_documents_from_files(self, file_paths: List[str], image_size=(256, 256)):
        documents = []
        filenames = []

        for filepath in file_paths:
            try:
                if filepath.lower().endswith('.pdf'):
                    pages = convert_from_path(filepath)
                    for idx, page in enumerate(pages):
                        img_np = np.array(page.convert('RGB'))
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        documents.append(img_cv)
                        filenames.append(f"{os.path.basename(filepath)}_page_{idx+1}")
                else:
                    img = cv2.imread(filepath)
                    if img is not None:
                        documents.append(img)
                        filenames.append(os.path.basename(filepath))
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        processed_docs = []
        for img in documents:
            resized = cv2.resize(img, image_size)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray.astype('float32') / 255.0
            processed_docs.append(np.expand_dims(normalized, axis=-1))

        return np.array(processed_docs), filenames

    def build_autoencoder(self, input_shape):
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        return keras.Model(encoder_inputs, decoded, name="autoencoder")

    def _create_synthetic_data(self, num_samples=50):
        synthetic_data = []
        for _ in range(num_samples):
            img = np.random.normal(0.8, 0.1, IMAGE_SIZE)
            for _ in range(np.random.randint(5, 15)):
                x = np.random.randint(0, IMAGE_SIZE[0] - 20)
                y = np.random.randint(0, IMAGE_SIZE[1] - 5)
                img[y:y+5, x:x+20] = np.random.normal(0.2, 0.05, (5, 20))
            img = np.clip(img, 0, 1)
            synthetic_data.append(np.expand_dims(img, axis=-1))
        return np.array(synthetic_data)

    def train_model(self):
        if not os.path.exists(GENUINE_DOCS_PATH):
            os.makedirs(GENUINE_DOCS_PATH)

        genuine_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            genuine_files.extend(glob.glob(os.path.join(GENUINE_DOCS_PATH, f'*.{ext}')))
        genuine_files.extend(glob.glob(os.path.join(GENUINE_DOCS_PATH, '*.pdf')))

        if not genuine_files:
            logger.warning("No genuine documents found. Creating synthetic training data.")
            x_train = self._create_synthetic_data()
        else:
            x_train, _ = self.load_documents_from_files(genuine_files, IMAGE_SIZE)

        if x_train.shape[0] == 0:
            raise ValueError("No training data available")

        self.model = self.build_autoencoder(x_train.shape[1:])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        x_train_noisy = x_train + np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        logger.info(f"Training model on {x_train.shape[0]} genuine documents (with noise)...")
        self.model.fit(x_train_noisy, x_train,
                       epochs=10,
                       batch_size=min(32, x_train.shape[0]),
                       shuffle=True,
                       validation_split=0.1 if x_train.shape[0] > 10 else 0,
                       verbose=0)

        genuine_recon = self.model.predict(x_train, verbose=0)
        genuine_errors = np.mean(np.square(x_train - genuine_recon), axis=(1, 2, 3))
        self.threshold = np.mean(genuine_errors) + 3 * np.std(genuine_errors)
        self.is_trained = True
        logger.info(f"Model trained. Threshold: {self.threshold:.4f}")

    def analyze_documents(self, file_paths: List[str], filenames: List[str]):
        if not self.is_trained:
            self.train_model()

        x_test, updated_filenames = self.load_documents_from_files(file_paths, IMAGE_SIZE)
        if x_test.shape[0] == 0:
            raise ValueError("No valid documents to analyze")

        predictions = self.model.predict(x_test, verbose=0)
        reconstruction_errors = np.mean(np.square(x_test - predictions), axis=(1, 2, 3))

        results = []
        for i, (error, filename) in enumerate(zip(reconstruction_errors, updated_filenames)):
            is_fraud = error > self.threshold
            confidence = min(0.99, 0.5 + abs(error - self.threshold) / (2 * self.threshold))
            results.append({
                "document_id": i + 1,
                "filename": filename,
                "mse_error": float(error),
                "is_fraud": bool(is_fraud),
                "confidence": float(confidence)
            })

        return {
            "threshold": float(self.threshold),
            "results": results,
            "total_documents": len(results),
            "fraud_detected": sum(1 for r in results if r["is_fraud"])
        }

fraud_detector = FraudDetector()

@app.on_event("startup")
async def startup_event():
    try:
        fraud_detector.train_model()
        logger.info("Fraud detection system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize fraud detection system: {e}")

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_trained": fraud_detector.is_trained,
        "threshold": fraud_detector.threshold if fraud_detector.threshold else None
    }

@app.post("/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        filenames = []

        for file in files:
            if not file.filename:
                continue

            file_path = os.path.join(temp_dir, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
                filenames.append(file.filename)
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")

        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files to analyze")

        try:
            results = fraud_detector.analyze_documents(file_paths, filenames)
            return JSONResponse(content=results)
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/retrain")
async def retrain_model():
    try:
        fraud_detector.train_model()
        return {"message": "Model retrained successfully", "threshold": fraud_detector.threshold}
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

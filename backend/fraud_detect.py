# fraud_detector_denoising.py
# A solution to detect forgeries, including watermarks, using a denoising autoencoder.
# This version outputs text-based results and includes the original filename in the summary.

import numpy as np
import cv2
import os
import glob
from PIL import Image
from pdf2image import convert_from_path
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- 1. DATA LOADING AND PREPROCESSING ---
# This step prepares the data for the model. It is crucial because the model will learn its definition
# of "normal" from the documents in the 'genuine_documents' folder. All documents are
# converted to a standardized format (grayscale, resized) so the model can process them consistently.

def load_documents_from_folder(folder_path, image_size=(256, 256)):
    """
    Loads documents (images and PDFs) from a specified folder,
    converts them to grayscale, and resizes them.
    
    Returns a tuple: (processed_documents_array, list_of_filenames).
    """
    print(f"Loading documents from: {folder_path}...")
    documents = []
    filenames = []
    
    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    
    # Load all images
    for ext in image_extensions:
        for filepath in glob.glob(os.path.join(folder_path, f'*.{ext}')):
            try:
                img = cv2.imread(filepath)
                if img is not None:
                    documents.append(img)
                    filenames.append(os.path.basename(filepath))
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

    # Load all PDFs and convert each page to an image
    for filepath in glob.glob(os.path.join(folder_path, '*.pdf')):
        try:
            pages = convert_from_path(filepath)
            for i, page in enumerate(pages):
                # Convert PIL Image to OpenCV format
                img_np = np.array(page.convert('RGB'))
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                documents.append(img_cv)
                # Create a filename convention for multi-page PDFs
                pdf_filename = os.path.basename(filepath)
                filenames.append(f"{pdf_filename}_page_{i+1}")
        except Exception as e:
            print(f"  Error converting PDF {filepath}: {e}")
            print("  Please ensure 'Poppler' is installed and its bin directory is in your system's PATH.")
            continue
    
    # Preprocess images
    processed_docs = []
    for img in documents:
        # Resize, convert to grayscale, and normalize
        resized = cv2.resize(img, image_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype('float32') / 255.0
        # Add a channel dimension for the model
        processed_docs.append(np.expand_dims(normalized, axis=-1))
        
    print(f"Found and processed {len(processed_docs)} documents.")
    return np.array(processed_docs), filenames

# Define folder paths
GENUINE_DOCS_PATH = os.path.join(os.getcwd(), 'genuine_documents')
SUSPECT_DOCS_PATH = os.path.join(os.getcwd(), 'suspect_documents')
IMAGE_SIZE = (256, 256)

# Load the datasets
x_train_genuine, _ = load_documents_from_folder(GENUINE_DOCS_PATH, IMAGE_SIZE)
if x_train_genuine.shape[0] == 0:
    print("Error: No genuine documents found. Please add documents to the 'genuine_documents' folder.")
    exit()

x_test_suspects, suspect_filenames = load_documents_from_folder(SUSPECT_DOCS_PATH, IMAGE_SIZE)
if x_test_suspects.shape[0] == 0:
    print("Warning: No suspect documents found. The model will still train, but no anomalies will be detected.")
    
# --- 2. MODEL ARCHITECTURE (The Denoising Autoencoder) ---
# This defines the "brain" of our fraud detection system. The model has two parts:
# The Encoder, which compresses the image into a small representation, and
# the Decoder, which tries to reconstruct the original image from that representation.
# This forces the model to learn only the most important features of a "normal" document.

def build_autoencoder(input_shape):
    """Builds a convolutional autoencoder model."""
    
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space (the bottleneck)
    encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(encoder_inputs, decoded, name="autoencoder")
    return autoencoder

# Build and compile the model
print("\nBuilding and compiling the denoising autoencoder model...")
model = build_autoencoder(x_train_genuine.shape[1:])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 3. TRAINING THE MODEL ---
# This is where the model learns what a "normal" document looks like.
# It is trained on NOISY versions of the genuine documents to force it to learn to "clean" the image.
# The target output is the ORIGINAL, clean document, not the noisy one.
print("\nTraining the model on genuine documents with added noise...")
epochs = 20  # Increased epochs for better learning
batch_size = 32

# Create noisy versions of the training data. This teaches the model to denoise.
x_train_noisy = x_train_genuine + np.random.normal(loc=0.0, scale=0.1, size=x_train_genuine.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)

# Train the model with noisy inputs and clean targets
model.fit(x_train_noisy, x_train_genuine, 
          epochs=epochs, 
          batch_size=batch_size,
          shuffle=True,
          validation_split=0.1)

# --- 4. ANOMALY DETECTION AND TEXT-BASED SUMMARY ---
# This is the final step where the actual fraud detection happens.
# A new document is passed through the model. If its reconstruction error is
# significantly higher than the genuine documents' errors, it's flagged as an anomaly.
if x_test_suspects.shape[0] > 0:
    print("\nDetecting anomalies in suspect documents...")
    predictions = model.predict(x_test_suspects)
    
    # Calculate the reconstruction error for each suspect document
    reconstruction_errors = np.mean(np.square(x_test_suspects - predictions), axis=(1, 2, 3))
    
    # Get the reconstruction errors for the genuine documents to set a baseline threshold.
    genuine_reconstructions = model.predict(x_train_genuine)
    genuine_errors = np.mean(np.square(x_train_genuine - genuine_reconstructions), axis=(1, 2, 3))

    # The threshold is the numerical boundary for fraud detection. We define it as the mean
    # of genuine errors plus 3 times their standard deviation (a common statistical approach).
    threshold = np.mean(genuine_errors) + 3 * np.std(genuine_errors)
    
    print(f"\nCalculated Anomaly Detection Threshold: {threshold:.4f}")

    # Print a summary
    print("\n--- Summary of Results ---")
    for i, error in enumerate(reconstruction_errors):
        status = "Forgery Detected! 🚨" if error > threshold else "Appears Genuine ✅"
        filename = suspect_filenames[i]
        print(f"File: {filename} -> MSE = {error:.4f} -> {status}")

else:
    print("\nNo suspect documents to analyze. Exiting.")

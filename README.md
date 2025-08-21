# Fraud Detection System using Visual Anomaly Detection

A comprehensive fraud detection system that uses deep learning and computer vision to detect forged documents through visual anomaly detection.

## Features

- **Modern React Frontend**: Beautiful, responsive UI built with React, TypeScript, and Tailwind CSS
- **FastAPI Backend**: High-performance Python backend with RESTful API
- **Deep Learning**: Convolutional autoencoder neural network for anomaly detection
- **Multi-format Support**: Handles PDF, JPG, PNG, BMP, and TIFF files
- **Real-time Analysis**: Instant fraud detection with confidence scores
- **Drag & Drop Upload**: Intuitive file upload interface
- **Visual Results**: Clear presentation of analysis results with confidence metrics

## Architecture

### Frontend (React + TypeScript)
- Modern, responsive UI with Tailwind CSS
- File upload with drag & drop support
- Real-time analysis results display
- Error handling and loading states

### Backend (FastAPI + TensorFlow)
- RESTful API for document analysis
- Convolutional autoencoder model
- Automatic model training and retraining
- Support for multiple file formats

### AI Model
- **Autoencoder Architecture**: Learns to reconstruct normal documents
- **Anomaly Detection**: High reconstruction error indicates potential fraud
- **Confidence Scoring**: Provides reliability metrics for each prediction
- **Adaptive Threshold**: Automatically calculated based on training data

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Poppler (for PDF processing)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fraud-detection-system
```

2. **Install frontend dependencies**
```bash
npm install
```

3. **Set up Python backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Install Poppler (for PDF support)**
   - **Windows**: Download from https://poppler.freedesktop.org/
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

### Running the Application

1. **Start the backend server**
```bash
cd backend
python main.py
```

2. **Start the frontend development server**
```bash
npm run dev
```

3. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Usage

1. **Upload Documents**: Drag and drop or select files to upload
2. **Analyze**: Click "Analyze for Fraud" to process documents
3. **Review Results**: View detailed analysis with confidence scores
4. **Interpret Results**:
   - ✅ **Genuine**: Low reconstruction error, appears authentic
   - 🚨 **Fraud Detected**: High reconstruction error, potential forgery

## How It Works

### Training Phase
1. The system learns from genuine documents in the `genuine_documents` folder
2. A convolutional autoencoder is trained to reconstruct normal document patterns
3. The model learns to minimize reconstruction error for authentic documents

### Detection Phase
1. New documents are processed through the trained autoencoder
2. Reconstruction error is calculated for each document
3. Documents with error above the threshold are flagged as potential fraud
4. Confidence scores are calculated based on deviation from the threshold

### Technical Details
- **Model**: Convolutional Autoencoder with encoder-decoder architecture
- **Input**: 256x256 grayscale images
- **Threshold**: Mean + 3×Standard Deviation of genuine document errors
- **Confidence**: Calculated based on distance from threshold

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - System health and model status
- `POST /analyze` - Analyze uploaded documents
- `POST /retrain` - Retrain model with new genuine documents

## File Structure

```
├── src/                    # React frontend source
│   ├── App.tsx            # Main application component
│   ├── main.tsx           # Application entry point
│   └── index.css          # Global styles
├── backend/               # Python backend
│   ├── main.py           # FastAPI application
│   ├── requirements.txt  # Python dependencies
│   └── genuine_documents/ # Training data folder
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the repository or contact the development team.
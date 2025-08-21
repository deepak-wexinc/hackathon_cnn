import React, { useState } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Loader2, Shield, Eye, Brain } from 'lucide-react';

interface AnalysisResult {
  document_id: number;
  filename: string;
  mse_error: number;
  is_fraud: boolean;
  confidence: number;
}

interface AnalysisResponse {
  threshold: number;
  results: AnalysisResult[];
  total_documents: number;
  fraud_detected: number;
}

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    setFiles(selectedFiles);
    setResults(null);
    setError(null);
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const droppedFiles = Array.from(event.dataTransfer.files);
    setFiles(droppedFiles);
    setResults(null);
    setError(null);
  };

  const analyzeDocuments = async () => {
    if (files.length === 0) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: AnalysisResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Fraud Detection using Visual Anomaly Detection
              </h1>
              <p className="text-gray-600 mt-1">
                Advanced AI-powered document authenticity verification
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <Eye className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="font-semibold text-gray-900">Visual Analysis</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Advanced computer vision algorithms analyze document patterns and structures
            </p>
          </div>
          
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-indigo-600" />
              </div>
              <h3 className="font-semibold text-gray-900">Deep Learning</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Convolutional autoencoder neural networks detect anomalies in document images
            </p>
          </div>
          
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-green-600" />
              </div>
              <h3 className="font-semibold text-gray-900">Real-time Detection</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Instant fraud detection with confidence scores and detailed analysis
            </p>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Documents for Analysis</h2>
          
          <div
            className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors duration-200"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Drop your documents here or click to browse
            </h3>
            <p className="text-gray-600 mb-4">
              Supports PDF, JPG, PNG, BMP, and TIFF formats
            </p>
            <input
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors duration-200 cursor-pointer"
            >
              <Upload className="w-5 h-5 mr-2" />
              Choose Files
            </label>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Selected Files ({files.length})
              </h3>
              <div className="space-y-3">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200"
                  >
                    <div className="flex items-center space-x-3">
                      <FileText className="w-5 h-5 text-gray-500" />
                      <div>
                        <p className="font-medium text-gray-900">{file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="text-red-500 hover:text-red-700 font-medium text-sm"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>

              <button
                onClick={analyzeDocuments}
                disabled={isAnalyzing}
                className="mt-6 w-full flex items-center justify-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing Documents...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5 mr-2" />
                    Analyze for Fraud
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-6 h-6 text-red-600" />
              <div>
                <h3 className="font-semibold text-red-900">Analysis Error</h3>
                <p className="text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Analysis Results</h2>
            
            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-600">{results.total_documents}</div>
                <div className="text-sm text-blue-800">Total Documents</div>
              </div>
              <div className="bg-red-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-red-600">{results.fraud_detected}</div>
                <div className="text-sm text-red-800">Fraud Detected</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600">
                  {results.total_documents - results.fraud_detected}
                </div>
                <div className="text-sm text-green-800">Genuine Documents</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-gray-600">
                  {results.threshold.toFixed(4)}
                </div>
                <div className="text-sm text-gray-800">Detection Threshold</div>
              </div>
            </div>

            {/* Individual Results */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Document Analysis</h3>
              {results.results.map((result, index) => (
                <div
                  key={index}
                  className={`p-6 rounded-lg border-2 ${
                    result.is_fraud
                      ? 'bg-red-50 border-red-200'
                      : 'bg-green-50 border-green-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      {result.is_fraud ? (
                        <AlertTriangle className="w-8 h-8 text-red-600" />
                      ) : (
                        <CheckCircle className="w-8 h-8 text-green-600" />
                      )}
                      <div>
                        <h4 className="font-semibold text-gray-900">
                          {result.filename}
                        </h4>
                        <p className={`text-sm font-medium ${
                          result.is_fraud ? 'text-red-700' : 'text-green-700'
                        }`}>
                          {result.is_fraud ? 'Forgery Detected! 🚨' : 'Appears Genuine ✅'}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-gray-900">
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Confidence</div>
                      <div className="text-xs text-gray-500 mt-1">
                        MSE: {result.mse_error.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
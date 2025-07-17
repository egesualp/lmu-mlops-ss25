# Financial Sentiment Classifier - Serving

This directory contains the serving components for the Financial Sentiment Classifier, including both the frontend (Streamlit) and backend (FastAPI) applications.

## Live Demo

**Frontend Application**: [https://frontend-891096115648rope-west1.run.app/](https://frontend-891096115648rope-west1p/)

## Project Structure

```
serving/
├── app.py                 # Streamlit frontend application
├── backend.py             # FastAPI backend application
├── requirements_frontend.txt  # Frontend dependencies
├── requirements_backend.txt   # Backend dependencies
└── README.md              # This file
```

## Architecture

- **Frontend**: Streamlit web application for user interaction
- **Backend**: FastAPI REST API serving ONNX model predictions
- **Model**: ONNX-optimized transformer model for sentiment classification

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- ONNX model files in the `onnx/` directory

### Local Development

#### 1Backend Setup

```bash
# Install backend dependencies
pip install -r serving/requirements_backend.txt

# Start the FastAPI backend
uvicorn serving.backend:app --host 0.0 --port 8000

The backend will be available at `http://localhost:800

####2. Frontend Setup

```bash
# Install frontend dependencies
pip install -r serving/requirements_frontend.txt

# Set backend URL environment variable
export BACKEND=http://localhost:8000

# Start the Streamlit frontend
streamlit run serving/app.py
```

The frontend will be available at `http://localhost:8501`

### Docker Deployment

#### Backend Container

```bash
# Build backend image
docker build -f dockerfiles/backend.dockerfile -t sentiment-backend .

# Run backend container
docker run -d -p 8000:80 backend sentiment-backend
```

#### Frontend Container

```bash
# Build frontend image
docker build -f dockerfiles/frontend.dockerfile -t streamlit-frontend .

# Run frontend container
docker run -p8511-e BACKEND=http://localhost:800streamlit-frontend
```

## Configuration

### Environment Variables

- `BACKEND`: URL of the backend service (required for frontend)

### Model Files

The backend expects the following files in the `onnx/` directory:
- `model.onnx`: ONNX model file
- `tokenizer.json`: Tokenizer configuration
- `tokenizer_config.json`: Tokenizer metadata
- `special_tokens_map.json`: Special tokens mapping
- `vocab.txt`: Vocabulary file

## API Endpoints

### Backend API

- `GET /`: Health check endpoint
- `POST /predict/`: Sentiment classification endpoint

#### Request Format
```json
{
  "text: our financial text here"
}
```

#### Response Format
```json
{
  "text: our financial text here,prediction": positive|neutral|negative",
  probabilities": 0.100.20.7```

## Testing

### Test Backend API

```bash
# Test health endpoint
curl http://localhost:8000/

# Test prediction endpoint
curl -X POST http://localhost:800/predict/ \
  -H "Content-Type: application/json\
  -d '{textThe stock market is performing well today."}'
```

### Test Frontend
1. Open `http://localhost:8501` in your browser
2ter financial text in the text area3lick Predict Sentimentto see results

## Production Deployment

### Google Cloud Run

The application is deployed on Google Cloud Run:

- **Frontend**: [https://frontend-891096115648rope-west1.run.app/](https://frontend-891096115648rope-west1.run.app/)
- **Backend**: Deployed as a separate Cloud Run service

### Deployment Commands

```bash
# Deploy backend
gcloud run deploy backend \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated

# Deploy frontend
gcloud run deploy frontend \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars BACKEND="https://backend-url"
```

## Troubleshooting

### Common Issues

1. **Backend Connection Error**: Ensure the backend is running and the `BACKEND` environment variable is set correctly
2odel Loading Error**: Verify that all ONNX model files are present in the `onnx/` directory
3. **Port Conflicts**: Make sure ports 80 (backend) and 8501end) are available

### Debug Mode

The frontend displays the backend URL for debugging purposes. Check the info box in the Streamlit interface.

## 📊 Features

- **Real-time Sentiment Analysis**: Instant classification of financial text
- **Probability Visualization**: Bar chart showing confidence scores
- **User-friendly Interface**: Clean Streamlit UI
- **RESTful API**: Standardized API endpoints
- **Containerized Deployment**: Docker support for easy deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4 locally
5. Submit a pull request

## 📄 License

This project is part of the LMU MLOps course. See the main LICENSE file for details.

# Financial Sentiment Classifier - Serving

This directory contains the Streamlit frontend for the Financial Sentiment Classifier.

## Live Demo

**Frontend Application**: [https://frontend-891096115648rope-west1.run.app/](https://frontend-891096115648rope-west1.run.app/)

## Project Structure

```
serving/
├── app.py                   # Streamlit frontend application
├── requirements_frontend.txt  # Frontend dependencies
└── README.md                # This file
```

## Architecture

- **Frontend**: Streamlit web application for user interaction
- **Backend**: [Public API](https://financial-sentiment-api-687370715419.europe-west3.run.app) (no local backend required)

## Quick Start

### Prerequisites

- Python 3.12
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Install frontend dependencies
pip install -r frontend/requirements_frontend.txt

# Start the Streamlit frontend
streamlit run frontend/app.py
```

The frontend will be available at `http://localhost:8501`

### Docker Deployment

```bash
# Build frontend image
docker build -f dockerfiles/frontend.dockerfile -t streamlit-frontend .

# Run frontend container
# (No BACKEND env needed, backend is static in the code)
docker run -p 8501:8501 streamlit-frontend
```

## API Info

The frontend uses the following public API for predictions:

- **API Base URL:** [https://financial-sentiment-api-687370715419.europe-west3.run.app](https://financial-sentiment-api-687370715419.europe-west3.run.app)
- **Endpoint:** `POST /predict`
- **Request:**
  ```json
  { "text": "Your financial text here" }
  ```
- **Response:**
  ```json
  { "label": "positive|neutral|negative", "score": 0.87 }
  ```

## Features

- **Real-time Sentiment Analysis**: Instant classification of financial text
- **Confidence Score**: Smiley face and confidence visualization
- **User-friendly Interface**: Clean Streamlit UI
- **No Backend Setup**: Uses a public API for predictions
- **Containerized Deployment**: Docker support for easy deployment

## Troubleshooting

- If you see connection errors, check your internet connection or the public API status.
- The frontend displays the backend URL for debugging purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is part of the LMU MLOps course. See the main LICENSE file for details.

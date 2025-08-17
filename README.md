# MLOps Project - Financial Sentiment Analysis

<img width="1004" height="857" alt="Screenshot 2025-08-17 152730" src="https://github.com/user-attachments/assets/c6b7cf73-01c7-4307-84c5-27f98a398283" />

## Live Demo

**Frontend Application**: [https://frontend-891096115648.europe-west1.run.app/](https://frontend-891096115648.europe-west1.run.app/)

**Inference API**: [https://financial-sentiment-api-687370715419.europe-west3.run.app/](https://financial-sentiment-api-687370715419.europe-west3.run.app/)

## Goal

This is the project description for the Machine Learning Operations course (SS25) at Ludwig Maximilian University. 
The goal of this project is to implement a comprehensive MLOps pipeline for **Financial Sentiment Analysis** using HuggingFace Transformers with the Kaggle financial-sentiment-dataset. 
The primary focus is on creating a robust and scalable MLOps cycle that encompasses data versioning, model training, experiment tracking, automated testing, continuous integration/deployment, and production monitoring. 
This project demonstrates best practices in machine learning engineering including reproducible training pipelines, automated model evaluation, containerized deployments, and cloud-based inference services.

## Framework

This project leverages the **HuggingFace Transformers** library as its core framework for natural language processing and model development. 
We specifically chose **BERT (Bidirectional Encoder Representations from Transformers)** from HuggingFace's model hub as our foundation model, utilizing the `bert-base-uncased` variant. 
The framework supports both PyTorch Lightning and native HuggingFace Trainer implementations, allowing for flexible experimentation and training approaches while maintaining consistency in model architecture and tokenization.

## Data

The project utilizes the **"sbhatti/financial-sentiment-analysis"** dataset from Kaggle, 
which contains financial text data specifically curated for sentiment analysis tasks. 
The dataset structure includes two primary columns: `Sentence` (containing financial text snippets such as news headlines, analyst reports, and market commentary) and `Sentiment` (with three classes: positive, negative, and neutral). 
During preprocessing, these columns are renamed to `text` and `label` respectively. 
The dataset is automatically split into training (80%), validation (10%), and test (10%) sets to ensure proper model evaluation. 

## Model

We selected **BERT (Bidirectional Encoder Representations from Transformers)** as our base model for this financial sentiment classification task due to several key advantages. 

BERT's bidirectional attention mechanism allows it to capture contextual relationships from both directions in the input sequence,
which is crucial for understanding the nuanced sentiment expressions in financial text where context heavily influences meaning.
The model's pre-training on large-scale text corpora provides a strong foundation for transfer learning to the financial domain.

Additionally, BERT's sequence classification architecture with a linear classification head is well-suited for our three-class sentiment prediction task (positive, negative, neutral).

The `bert-base-uncased` variant provides an optimal balance between model performance and computational efficiency, while the extensive HuggingFace ecosystem ensures reliable tokenization, training utilities, and deployment capabilities essential for our MLOps pipeline.

## Project structure

The directory structure of the project looks like this:
```txt
├── .dvc/                     # DVC (Data Version Control) files
├── .github/                  # GitHub actions and workflows
│   └── workflows/
│       ├── data-change.yaml
│       ├── load_api_test.yaml
│       ├── model-change.yaml
│       ├── pre_commit.yaml
│       ├── tests.yaml
│       └── update_pre_commit.yaml
├── api/                      # API service modules
│   ├── __init__.py
│   ├── main.py
│   └── sentiment_monitoring.py
├── artifacts/                # Build and training artifacts
├── cloudbuild/              # Google Cloud Build configurations
│   ├── build_api_docker_image.yaml
│   ├── build_docker_images.yaml
│   ├── build_monitoring_docker_image.yaml
│   ├── cloudrun_monitoring.yaml
│   ├── vertex_config.yaml
│   └── vertex_sweep_config.yaml
├── conf/                    # Configuration files (Hydra configs)
│   ├── config_hf.yaml
│   ├── config_pytorch.yaml
│   ├── sweep.yaml
│   ├── sweep_pytorch.yaml
│   └── sweep_pytorch_2.yaml
├── data/                    # Data directory
│   ├── processed/          # Processed datasets (train, eval, test CSVs)
│   │   ├── eval.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── sentiment_data.csv.dvc
│   ├── raw/                # Raw datasets
│   │   └── data.csv
│   └── test_trigger.txt
├── dockerfiles/            # Docker containers
│   ├── api.dockerfile
│   ├── frontend.dockerfile
│   ├── sentiment_monitoring.dockerfile
│   └── train.dockerfile
├── frontend/               # Frontend application
│   ├── __init__.py
│   ├── app.py
│   ├── README.md
│   └── requirements_frontend.txt
├── models/                 # Trained models directory
├── reports/                # Reports and outputs
│   └── figures/           # Generated figures and plots
├── src/                    # Source code
│   ├── __init__.py
│   ├── data.py            # Data processing utilities
│   ├── eval.py            # Model evaluation
│   ├── model.py           # Model definitions
│   ├── stats.py           # Statistics utilities
│   ├── train_hf.py        # HuggingFace training script
│   └── train_lightning.py # PyTorch Lightning training script
├── tests/                  # Tests
│   ├── __init__.py
│   ├── locustfile.py      # Load testing
│   ├── README.md
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── utils/                  # Utility scripts
│   ├── check_onnx.py
│   ├── export_onnx.py
│   └── wandb_test.py
├── wandb/                  # Weights & Biases experiment tracking
├── .dvcignore             # DVC ignore patterns
├── .editorconfig          # Editor configuration
├── .gcloudignore          # Google Cloud ignore patterns
├── .gcloudignore.api      # API-specific Cloud ignore patterns
├── .gitignore             # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── coverage.txt           # Test coverage report
├── data.dvc              # DVC data pipeline definition
├── LICENSE               # Project license
├── main.py               # Main entry point
├── profile.out           # Profiling output
├── profile_output.prof   # Detailed profiling data
├── pyproject.toml        # Python project configuration
├── pytest.ini           # Pytest configuration
├── README.md             # Project README
├── requirements.txt      # Core project dependencies
├── requirements_dev.txt  # Development dependencies
├── requirements_tests.txt # Testing dependencies
└── tasks.py              # Project automation tasks (Invoke)
```
---

Created using mlops_template, a cookiecutter template for getting started with Machine Learning Operations (MLOps).
Edited according to project needs and structures.

 Usage & Quickstart

## Setup & Initialization

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/lmu-mlops-ss25.git
   cd lmu-mlops-ss25
   ```

2. **Create and activate a conda environment (Python 3.12):**
   ```bash
   conda create --name mlops_project python=3.12
   conda activate mlops_project
   ```

3. **Install dependencies:**
   - You can install dependencies directly:
     ```bash
     pip install -r requirements.txt
     ```
   - Or, install the project as a package (recommended, uses `pyproject.toml`):
     ```bash
     pip install -e .
     ```

4. **(Optional) Set up DVC and pull data:**
   - Make sure you have [DVC](https://dvc.org/doc/install) installed.
   - Configure remote storage (e.g., GCS bucket):
     ```bash
     dvc remote add -d myremote gcs://your-bucket/path
     dvc pull
     ```

5. **Download the dataset:**
   - Make sure you have [Invoke](https://www.pyinvoke.org/) installed:
     ```bash
     pip install invoke
     ```
   - Download the data using the provided task:
     ```bash
     invoke download
     ```
     This will download data from kaggle and store it under data/raw folder.

6. **Preprocess the data:**
   - Run the preprocessing task:
     ```bash
     invoke preprocess-data
     ```
     This will preprocess data, by splitting it into train, eval and test under data/processed folder.

## Data Management with DVC (Optional)

- Data and model artifacts are versioned using DVC and stored in a GCS bucket.
- You need to have permission to access our data. To reproduce the application without DVC, please follow Data Processing steps.
- To fetch the latest data:
  ```bash
  dvc pull
  ```
- To add new data:
  ```bash
  dvc add data/your_new_data.csv
  dvc push
  ```

## Training the Model

- Training scripts are located in `src/train_hf.py` (HuggingFace) and `src/train_lightning.py` (PyTorch Lightning).
- Be aware that some functionalities and arguments might not work if you train with lightning. 
- Example command:
  ```bash
  python src/train_hf.py --config-name config_hf
  ```
- Adjust configuration files in `conf/` as needed.

## Evaluating the Model

- Use the evaluation script:
  ```bash
  python src/eval.py --model-path models/final
  ```

## Export Steps

- Make sure you are logged in to [Weights & Biases (W&B)](https://wandb.ai/):
  ```bash
  wandb login
  ```

1. **Run the export script:**
   ```bash
   python utils/export_onnx.py
   ```
   This script will:
   - Download the trained model artifact from W&B.
   - Load the model and tokenizer.
   - Export the model to ONNX format and save it in `models/onnx/model.onnx`.
   - Save the tokenizer files in the same directory.
   - Upload the ONNX model and tokenizer as a new artifact to W&B.

2. **Result:**
   - The exported ONNX model and tokenizer will be available in the `models/onnx/` directory.
   - The ONNX model will also be uploaded to your W&B project as a new artifact.

### Customization

If you want to change the model or artifact names, edit the configuration section at the top of `utils/export_onnx.py`:
```python
PROJECT_NAME = "financial-sentiment-bert"
ENTITY = "your-wandb-entity"
MODEL_ARTIFACT = "entity/project/model-artifact:version"
ONNX_ARTIFACT_NAME = "your-onnx-artifact-name"
```

## Running with Docker

- **Build Docker images:**
  ```bash
  docker build -f dockerfiles/api.dockerfile -t mlops-api .
  ```

- **Run API container:**
  ```bash
  docker run -p 8000:8000 mlops-api
  ```

- See `cloudbuild/` for cloud deployment configurations.

## Inference & API Usage

- The API is built with FastAPI (`api/main.py`).
- Once running (locally or in Docker), access docs at: [http://localhost:8000/docs](http://localhost:8000/docs)
- Example request:
  ```bash
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Your input text"}'
  ```

## Testing

- Run tests using:
  ```bash
  pytest
  ```

## Frontend

Please refer to [frontend/README.md](frontend/README.md) for instructions and details about the frontend application.


## Troubleshooting

- If you encounter issues with DVC, check your remote configuration and credentials.
- For Docker issues, ensure Docker is running and you have sufficient permissions.
- For API errors, check logs in the container or `logs/` directory.

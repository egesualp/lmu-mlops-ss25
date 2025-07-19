# MLOps Project - Financial Sentiment Analysis

## Live Demo

**Frontend Application**: [https://frontend-891096115648.europe-west1.run.app/](https://frontend-891096115648rope-west1.run.app/)

## GOAL

This is the project description for the Machine Learning Operations course (SS25) at Ludwig Maximilian University. 
The goal of this project is to implement a comprehensive MLOps pipeline for **Financial Sentiment Analysis** using HuggingFace Transformers with the Kaggle financial-sentiment-dataset. 
The primary focus is on creating a robust and scalable MLOps cycle that encompasses data versioning, model training, experiment tracking, automated testing, continuous integration/deployment, and production monitoring. 
This project demonstrates best practices in machine learning engineering including reproducible training pipelines, automated model evaluation, containerized deployments, and cloud-based inference services.

## FRAMEWORK

This project leverages the **HuggingFace Transformers** library as its core framework for natural language processing and model development. 
The Transformers library provides state-of-the-art pre-trained models and seamless integration with modern ML infrastructure. 
We specifically chose **BERT (Bidirectional Encoder Representations from Transformers)** from HuggingFace's model hub as our foundation model, utilizing the `bert-base-uncased` variant. 
The framework supports both PyTorch Lightning and native HuggingFace Trainer implementations, allowing for flexible experimentation and training approaches while maintaining consistency in model architecture and tokenization.

## DATA

The project utilizes the **"sbhatti/financial-sentiment-analysis"** dataset from Kaggle, 
which contains financial text data specifically curated for sentiment analysis tasks. 
The dataset structure includes two primary columns: `Sentence` (containing financial text snippets such as news headlines, analyst reports, and market commentary) and `Sentiment` (with three classes: positive, negative, and neutral). 
During preprocessing, these columns are renamed to `text` and `label` respectively. 
The dataset is automatically split into training (80%), validation (10%), and test (10%) sets to ensure proper model evaluation. 
The financial domain specificity of this dataset makes it ideal for training models that can understand the nuanced sentiment expressions commonly found in financial communications and market analysis.

## MODEL

We selected **BERT (Bidirectional Encoder Representations from Transformers)** as our base model for this financial sentiment classification task due to several key advantages. 
BERT's bidirectional attention mechanism allows it to capture contextual relationships from both directions in the input sequence,
which is crucial for understanding the nuanced sentiment expressions in financial text where context heavily influences meaning.
The model's pre-training on large-scale text corpora provides a strong foundation for transfer learning to the financial domain. 
Additionally, BERT's sequence classification architecture with a linear classification head is well-suited for our three-class sentiment prediction task (positive, negative, neutral).
The `bert-base-uncased` variant provides an optimal balance between model performance and computational efficiency, while the extensive HuggingFace ecosystem ensures reliable tokenization,
training utilities, and deployment capabilities essential for our MLOps pipeline.

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

Created using mlops_template, a cookiecutter template for getting started with Machine Learning Operations (MLOps).
Edited according to project needs and structures.


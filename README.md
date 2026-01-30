# End-to-End Image Classification System

## Project Overview
This project implements a complete machine learning pipeline for multi-class image classification using Transfer Learning. It features an automated preprocessing script, a training pipeline using ResNet50, and a containerized REST API built with FastAPI and Docker.

## Features
* **Transfer Learning**: Fine-tunes a pre-trained ResNet50 model on the Caltech-101 dataset.
* **Automated Pipeline**: Scripts for downloading, splitting, and augmenting data.
* **REST API**: A FastAPI backend providing `/predict` and `/health` endpoints.
* **Containerization**: Fully Dockerized application managed via Docker Compose.
* **Robust Error Handling**: Graceful handling of invalid file types and missing uploads.

## Prerequisites
* Docker & Docker Compose
* Git

## Quick Start (Run with Docker)
The entire application can be launched with a single command.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Set up Environment Variables:**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

3.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    * The API will be available at `http://localhost:8000`.
    * The model will automatically load from the `model/` directory.

## API Documentation

### 1. Health Check
* **Endpoint**: `GET /health`
* **Description**: Verifies the service is running.
* **Response**: `{"status": "ok"}`

### 2. Predict Image
* **Endpoint**: `POST /predict`
* **Description**: Classifies an uploaded image.
* **Body**: `form-data` with key `file` (image file).
* **Success Response**:
    ```json
    {
      "predicted_class": "airplanes",
      "confidence": 0.98
    }
    ```
* **Error Response**: Returns `400 Bad Request` for missing or invalid files.

## Design Choices

### Model Architecture
I selected **ResNet50** as the base model.
* **Justification**: ResNet50 offers an excellent balance between accuracy and computational efficiency. It is deeper than VGG16 (providing better feature extraction) but lighter than larger variants, making it suitable for this deployment context.

### Training Strategy
* **Feature Extraction**: The convolutional base of ResNet50 was frozen to retain learned features from ImageNet.
* **Head Replacement**: The final fully connected layer was replaced to match the number of classes in the custom dataset.
* **Optimizer**: SGD with Momentum (0.9) and a step-based learning rate scheduler.

## Performance
Evaluation metrics on the validation set are stored in `results/metrics.json`.
* **Accuracy**: ~79% (varies by training run)
* **Precision/Recall**: Weighted averages calculated across all classes.

## Development Setup (Local)
If you wish to run scripts manually without Docker:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Preprocessing**:
    ```bash
    python src/preprocess.py
    ```
3.  **Run Training**:
    ```bash
    python src/train.py
    ```
4.  **Run Tests**:
    ```bash
    pytest tests/test_api.py
    ```# Image_classifier_API

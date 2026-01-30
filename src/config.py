import os

# Configuration based on Core Requirements - Step 6, Item 2
class Config:
    # Port for the API server (Page 7)
    API_PORT = int(os.getenv("API_PORT", 8000))

    # Path to the trained model artifact inside the container (Page 7)
    MODEL_PATH = os.getenv("MODEL_PATH", "model/image_classifier.pth")
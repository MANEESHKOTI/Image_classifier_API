import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from config import Config

# Initialize FastAPI app
app = FastAPI(title="Image Classification API")

# Global variables for model and classes
model = None
class_names = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If on Mac M-series, use MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")

def load_model_artifact():
    """
    Loads the trained model and class names on startup.
    Strictly follows Core Req 6 (Model Loading).
    """
    global model, class_names
    
    # 1. Load Class Names
    # We infer class names from the data directory structure, same as training
    # This ensures consistency between training and inference
    train_dir = os.path.join("data", "train")
    if os.path.exists(train_dir):
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    else:
        # Fallback if data dir is missing in production container (though it should be mounted)
        print("Warning: Data directory not found. Class names might be empty.")
        class_names = [] # You might hardcode these if strictly needed for production without data mount

    num_classes = len(class_names)
    print(f"Loading model for {num_classes} classes: {class_names}")

    # 2. Initialize Model Architecture (ResNet50)
    # Must match the architecture in train.py
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 3. Load Weights
    if os.path.exists(Config.MODEL_PATH):
        try:
            state_dict = torch.load(Config.MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval() # Set to evaluation mode
            print(f"Model loaded successfully from {Config.MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError("Could not load model artifact.")
    else:
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        # We don't raise error here to allow the app to start, but predict will fail
        
# Load model immediately when script is imported/run
load_model_artifact()

# Preprocessing transform (Must match validation transform)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

@app.get("/health")
async def health_check():
    """
    Core Req 11: Health check endpoint.
    Returns {"status": "ok"} (Page 14).
    """
    # Simple check to ensure model is loaded
    if model is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Model not loaded"})
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    """
    Core Req 8 & 9: Prediction endpoint.
    Accepts an image file, returns predicted class and confidence.
    """
    # Core Req 10: Error Handling
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate file type based on extension (basic check)
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")

    try:
        # 1. Read and Process Image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
             raise HTTPException(status_code=400, detail="Invalid image file content.")

        input_tensor = transform(image).unsqueeze(0).to(device)

        # 2. Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # 3. Format Response (Page 12 Schema)
        idx = predicted_idx.item()
        conf_score = confidence.item()
        
        # Safety check for index
        predicted_label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"

        return {
            "predicted_class": predicted_label,
            "confidence": conf_score
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        # General error handling
        print(f"Prediction Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # This block is for local debugging, Docker will use uvicorn command
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.API_PORT)
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from config import Config

def evaluate_model():
    # 1. Setup & Configuration
    # Ensure results directory exists
    results_dir = os.path.dirname("results/metrics.json")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Use the same device logic as training
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Evaluating on device: {device}")

    # 2. Data Preparation
    # Must use the same validation transformations as training
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join('data', 'val')
    if not os.path.exists(data_dir):
        print(f"Error: Validation data not found at {data_dir}")
        return

    val_dataset = datasets.ImageFolder(data_dir, data_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Evaluating on {len(val_dataset)} images across {num_classes} classes...")

    # 3. Load Model
    # Initialize the same architecture (ResNet50)
    model = models.resnet50(weights=None) # Weights=None because we load our own
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the trained weights
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate Metrics (Core Req 7)
    acc = accuracy_score(all_labels, all_preds)
    
    # 'weighted' calculates metrics for each label, and finds their average weighted by support
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Confusion Matrix (NxN array)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # 6. Save Results
    # Formatting strictly according to Page 11 JSON Schema
    metrics_data = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "confusion_matrix": cm.tolist() # Convert numpy array to list for JSON serialization
    }

    output_path = "results/metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    evaluate_model()
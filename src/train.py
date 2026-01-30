import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from config import Config

def train_model():
    """
    Trains the model and saves the artifact.
    Strictly follows Core Requirements 4, 5, and 6.
    """
    # 1. Device Configuration
    # Ensures the script is "runnable" on different hardware (Req 5)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (GPU) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU) acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # Ensure model directory exists
    model_dir = os.path.dirname(Config.MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 2. Data Augmentation (Core Req 4)
    # Must apply at least two random augmentations.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),       # Augmentation 1 [cite: 253]
            transforms.RandomHorizontalFlip(),       # Augmentation 2 [cite: 253]
            transforms.RandomRotation(10),           # Augmentation 3 (Bonus)
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    if not os.path.exists(data_dir):
        print("Error: 'data/' directory not found. Run src/preprocess.py first.")
        return

    # Load Data (Core Req 5.1)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # num_workers=0 is safest for compatibility
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"Classes found: {num_classes}")

    # 3. Model Initialization (Core Req 5.2 & 5.3)
    # Using ResNet50 as suggested in the PDF [cite: 141]
    print("Initializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze base layers (Transfer Learning Strategy) [cite: 146]
    for param in model.parameters():
        param.requires_grad = False

    # Modify final layer (Req 5.3)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Training Loop (Core Req 5.4)
    num_epochs = 5 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:4f}')

    # 5. Save Artifact (Core Req 5.5 & 6)
    model.load_state_dict(best_model_wts)
    print(f"Saving model to {Config.MODEL_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
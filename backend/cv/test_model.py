import torch
import torchvision.models as models
import json
import os

print("--- Test Start ---")
try:
    df_model_path = r'D:\VTryOn\ml\checkpoints\df_model_epoch_5.pth'
    class_idx_path = r'D:\VTryOn\ml\class_to_idx.json'
    
    if not os.path.exists(df_model_path):
        print(f"Error: Model path does not exist: {df_model_path}")
    if not os.path.exists(class_idx_path):
        print(f"Error: JSON path does not exist: {class_idx_path}")
        
    with open(class_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    print(f"Loaded {len(class_to_idx)} classes")
    
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    
    print("Loading state dict...")
    state_dict = torch.load(df_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("Success: Model loaded correctly")
except Exception as e:
    print(f"Error: {e}")
print("--- Test End ---")

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
try:
    from rembg import remove
except ImportError:
    remove = None

import io
import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from unet_model import SimpleUNet
import json
try:
    import smplx
except ImportError:
    smplx = None

# Init Models safely
unet_model = None
df_model = None
df_class_names = []
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml/VITON_Training/viton_unet_model.pth'))
    if os.path.exists(model_path):
        unet_model = SimpleUNet()
        unet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        unet_model.eval()
        print("Trained UNet successfully loaded.")
except Exception as e:
    print(f"UNet initialization error: {e}")

try:
    df_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml/checkpoints/df_model_epoch_5.pth'))
    class_idx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml/class_to_idx.json'))
    
    if os.path.exists(class_idx_path):
        with open(class_idx_path, 'r') as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            df_class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
            
        # Use same architecture as training
        df_model = models.resnet50(weights=None)
        df_model.fc = torch.nn.Linear(df_model.fc.in_features, len(class_to_idx))
        
        if os.path.exists(df_model_path):
            checkpoint = torch.load(df_model_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                df_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                df_model.load_state_dict(checkpoint)
            df_model.eval()
            print("Trained DeepFashion ResNet successfully loaded.")
        else:
            print(f"DeepFashion model file not found at: {df_model_path}")
    else:
        print(f"Class mapping file not found at: {class_idx_path}")
except Exception as e:
    print(f"DeepFashion model initialization error: {e}")

# Preprocessing transforms for our Try-ON UNet
unet_transforms = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

df_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def unnormalize_tensor(tensor):
    return (tensor * 0.5) + 0.5

app = FastAPI(title="Virtual Try-On CV Backend", description="Handles garment segmentation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "cv-backend"}

@app.post("/api/cv/segment-garment")
async def segment_garment(file: UploadFile = File(...)):
    if remove is None:
        return Response(status_code=500, content="Background removal library failed to load.")
    
    contents = await file.read()
    
    # Process the image to remove background using U2-Net
    output_image = remove(contents)
    
    return Response(content=output_image, media_type="image/png")

@app.post("/api/cv/try-on")
async def advanced_try_on(clothing: UploadFile = File(...), person: UploadFile = File(...)):
    if unet_model is None:
        return Response(status_code=500, content="Advanced Try-On models are not available. Ensure models are trained and present.")
        
    cloth_bytes = await clothing.read()
    cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
    
    # Process through UNet
    input_tensor = unet_transforms(cloth_img).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = unet_model(input_tensor)
        
    out_tensor = unnormalize_tensor(output_tensor.squeeze(0))
    out_pil = transforms.ToPILImage()(out_tensor)
    
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/api/cv/classify-garment")
async def classify_garment(file: UploadFile = File(...)):
    if df_model is None:
        return Response(status_code=500, content="DeepFashion model is not available. Ensure models are trained and present.")
        
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    input_tensor = df_transforms(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = df_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
        
        top5 = []
        for i in range(5):
            idx = top5_idx[0][i].item()
            conf = top5_prob[0][i].item()
            name = df_class_names[idx] if idx < len(df_class_names) else "Unknown"
            top5.append({"rank": i + 1, "class_name": name, "class_index": idx, "confidence": round(conf * 100, 2)})
        
        predicted_idx = top5_idx[0][0].item()
        
    class_name = df_class_names[predicted_idx] if predicted_idx < len(df_class_names) else "Unknown"
    
    return {
        "predicted_class": class_name,
        "predicted_index": predicted_idx,
        "confidence": round(top5_prob[0][0].item() * 100, 2),
        "top5_predictions": top5
    }

@app.post("/api/cv/smpl-estimate")
async def smpl_estimate():
    # SMPL 3D Body Representation layer
    # Requires official SMPL .pkl models installed in local environment to deform garment meshes onto bodies dynamically
    try:
        # Pseudo SMPL Layer Implementation Setup:
        # body_model = smplx.create(model_path='models', model_type='smpl')
        return {"status": "SMPL endpoint is active.", "message": "SMPL requires model parameter files. Provide 3D coordinates from the frontend WARP array."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

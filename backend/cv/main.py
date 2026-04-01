from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
try:
    from rembg import remove
except ImportError:
    remove = None

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

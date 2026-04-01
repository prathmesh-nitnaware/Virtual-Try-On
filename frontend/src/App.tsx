import React, { useEffect, useRef, useState } from 'react';
import { drawDenseWarp } from './warpUtils';

import './index.css';

type PoseResults = any;

// Base64 encoded sample SVG T-Shirt (Proper white t-shirt shape)
const TSHIRT_SRC = "/garment_fixed.png";

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [garmentUrl, setGarmentUrl] = useState<string>(TSHIRT_SRC);
  
  const tshirtImgRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Calibration State for AI or Manual Fit
  const [fitScale, setFitScale] = useState(1.0);
  const [fitOffsetY, setFitOffsetY] = useState(0);
  const [isCalibrating, setIsCalibrating] = useState(false);
  
  // Model References to keep them from garbage collection
  const poseModelRef = useRef<any>(null);
  const latestPoseRef = useRef<PoseResults | null>(null);

  useEffect(() => {
    // Preload T-Shirt image whenever garmentUrl changes
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = garmentUrl;
    img.onload = () => {
      tshirtImgRef.current = img;
    };
  }, [garmentUrl]);

  useEffect(() => {
    // Initialize MediaPipe Pose
    const PoseObj = (window as any).Pose;
    const pose = new PoseObj({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1, 
      smoothLandmarks: true,
      enableSegmentation: true, 
      smoothSegmentation: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results: PoseResults) => {
      latestPoseRef.current = results;
    });

    poseModelRef.current = pose;

    return () => {
      if (poseModelRef.current) poseModelRef.current.close();
    };
  }, []);

  const drawFrame = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const video = videoRef.current;
    
    if (!canvas || !ctx || !video) return;

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const results = latestPoseRef.current;

    if (results && results.poseLandmarks && tshirtImgRef.current) {
      const landmarks = results.poseLandmarks;
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12];
      const leftHip = landmarks[23];
      const rightHip = landmarks[24];

      let tempCanvas = offscreenCanvasRef.current;
      if (!tempCanvas) {
        tempCanvas = document.createElement('canvas');
        offscreenCanvasRef.current = tempCanvas;
      }
      
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tCtx = tempCanvas.getContext('2d');
      
      if (tCtx) {
        // 1. Calculate Core Torso Framing with User/AI Calibration
        const shoulderDistX = Math.abs(rightShoulder.x - leftShoulder.x);
        const baseWidth = shoulderDistX * canvas.width * 1.6; // Increased base width for sleeve coverage
        const shirtWidth = baseWidth * fitScale;
        
        const torsoCenterX = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
        const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        const hipY = (leftHip.y + rightHip.y) / 2;
        const torsoHeight = Math.abs(hipY - shoulderY) * canvas.height;
        
        const baseHeight = torsoHeight * 1.7; // Shirt is taller than purely hip distance
        const shirtHeight = baseHeight * fitScale;
        
        const shirtX = (torsoCenterX * canvas.width) - (shirtWidth / 2);
        // Base alignment at collarbone (15% up from shoulders) + User Offset
        const baseShirtY = (shoulderY * canvas.height) - (shirtHeight * 0.15);
        const shirtY = baseShirtY + fitOffsetY;
        
        tCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // 2. Draw Body Base (Isolates user)
        if (results.segmentationMask) {
          tCtx.save();
          tCtx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);
          tCtx.globalCompositeOperation = 'source-in';
          tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
          tCtx.restore();
        }

        // 3. Compute Deformation
        const turnedRatio = Math.abs(rightShoulder.z - leftShoulder.z) * 2.5;
        const hipToShoulderRatio = (Math.abs(rightHip.x - leftHip.x) / shoulderDistX);
        const pinchAmount = shirtWidth * 0.12 * Math.max(0, 1 - (hipToShoulderRatio - 0.75)); 
        const chestBulge = shirtHeight * (0.07 + turnedRatio * 0.05);

        // 4. Draw Garment with depth
        tCtx.save();
        if (results.segmentationMask) {
           tCtx.globalCompositeOperation = 'source-atop'; 
        }
        
        const imgW = tshirtImgRef.current.width || 240;
        const imgH = tshirtImgRef.current.height || 280;
        
        tCtx.shadowColor = 'rgba(0,0,0,0.3)';
        tCtx.shadowBlur = 15;
        tCtx.shadowOffsetY = 6;
        
        drawDenseWarp(tCtx, tshirtImgRef.current, imgW, imgH, shirtX, shirtY, shirtWidth, shirtHeight, pinchAmount, chestBulge);
        tCtx.restore();

        // 5. Ambient Shadow Blend (Proper Option 3)
        tCtx.save();
        tCtx.globalCompositeOperation = 'multiply';
        tCtx.globalAlpha = 0.15; // Realistic clothing-to-body shadow integration
        tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        tCtx.restore();

        ctx.drawImage(tempCanvas, 0, 0);
      }
      
      ctx.restore();
    } else {
      ctx.restore();
    }

    requestAnimationFrame(drawFrame);
  };

  const handleSmartFit = async () => {
    if (!canvasRef.current) return;
    setIsCalibrating(true);

    try {
      // 1. Capture snapshot as base64
      const imageBase64 = canvasRef.current.toDataURL('image/jpeg', 0.7).split(',')[1];

      // 2. Call Ollama Local API (llava model)
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llava',
          prompt: `Look at this person and the shirt overlay. Currently the shirt fits at Scale ${fitScale} and OffsetY ${fitOffsetY}. 
          Return ONLY JSON in this format: {"scaleFactor": <float>, "yOffsetDelta": <int>}. 
          If the shirt is too small, increase scaleFactor. If it's too high on the neck, increase yOffsetDelta (positive shifts down). 
          Keep your scaleFactor between 0.8 and 2.5.`,
          images: [imageBase64],
          stream: false,
          format: 'json'
        }),
      });

      const data = await response.json();
      const aiResponse = JSON.parse(data.response);

      if (aiResponse.scaleFactor) setFitScale(aiResponse.scaleFactor);
      if (aiResponse.yOffsetDelta !== undefined) setFitOffsetY(prev => prev + aiResponse.yOffsetDelta);

    } catch (err) {
      console.error("AI Smart Fit failed:", err);
      alert("AI Fit failed. Ensure Ollama is running with 'llava' model and OLLAMA_ORIGINS='*' is set.");
    } finally {
      setIsCalibrating(false);
    }
  };

  const startCamera = async () => {
    if (!videoRef.current || !poseModelRef.current) return;
    setIsModelLoading(true);

    try {
      const CameraObj = (window as any).Camera;
      const camera = new CameraObj(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current && poseModelRef.current) {
            await poseModelRef.current.send({ image: videoRef.current });
          }
        },
        width: 1280,
        height: 720,
        facingMode: "user"
      });

      await camera.start();
      setIsCameraActive(true);
      setIsModelLoading(false);
      requestAnimationFrame(drawFrame);
      
    } catch (err) {
      console.error("Error starting camera: ", err);
      setIsModelLoading(false);
      alert("Please allow camera access.");
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/cv/segment-garment', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to segment garment');

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      setGarmentUrl(objectUrl);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Failed to process image. Make sure the CV Backend is running on port 8000.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="tryon-container">
      <div className="header">
        <h1 className="title">Virtual Try-On</h1>
        <p className="subtitle">Real-time AR with Local AI Smart Calibration.</p>
      </div>

      <div className="camera-box">
        <div className="status-badge">
          <span className={`status-dot ${isCameraActive ? 'active' : ''}`}></span>
          {isCameraActive ? 'LIVE AR ENGINE' : 'OFFLINE'}
        </div>

        {isModelLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Warming up AI Engine...</p>
          </div>
        )}

        <video ref={videoRef} className="video-element" playsInline style={{ display: 'none' }}></video>
        <canvas ref={canvasRef} className="output-canvas"></canvas>
      </div>

      {isCameraActive && (
        <div className="calibration-panel">
          <div className="tool-row">
             <button className="btn api" onClick={handleSmartFit} disabled={isCalibrating}>
               {isCalibrating ? 'Ollama Analyzing...' : '✨ Smart AI Fit (Ollama)'}
             </button>
          </div>
          <div className="tool-row">
            <div className="nudge-group">
               <span>Size:</span>
               <button onClick={() => setFitScale(s => s + 0.05)}>+</button>
               <button onClick={() => setFitScale(s => Math.max(0.2, s - 0.05))}>-</button>
            </div>
            <div className="nudge-group">
               <span>Height:</span>
               <button onClick={() => setFitOffsetY(y => y - 5)}>↑</button>
               <button onClick={() => setFitOffsetY(y => y + 5)}>↓</button>
            </div>
          </div>
        </div>
      )}

      <div className="controls">
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileUpload} 
          accept="image/*" 
          style={{ display: 'none' }} 
        />
        
        <button 
          className="btn secondary" 
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
        >
          {isUploading ? 'Segmenting...' : 'Upload T-Shirt'}
        </button>

        {!isCameraActive && (
          <button className="btn primary" onClick={startCamera}>
             Launch AR Experience
          </button>
        )}
        {isCameraActive && (
          <button className="btn" onClick={() => window.location.reload()}>
             End Session
          </button>
        )}
      </div>
    </div>
  );
};

export default App;

import React, { useEffect, useRef, useState } from 'react';

import './index.css';

type PoseResults = any;
type SegResults = any;

// Base64 encoded sample SVG T-Shirt (Proper white t-shirt shape)
const TSHIRT_SRC = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQwIiBoZWlnaHQ9IjI4MCIgdmlld0JveD0iMCAwIDI0MCAyODAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTSA4MCAxMCBRIDEyMCA0MCAxNjAgMTAgTCAyMDAgMjAgTCAyMzAgNzAgTCAxODAgMTEwIEwgMTcwIDgwIEwgMTcwIDI3MCBMIDcwIDI3MCBMIDcwIDgwIEwgNjAgMTEwIEwgMTAgNzAgTCA0MCAyMCBaIiBmaWxsPSIjRjhGOEY4IiBzdHJva2U9IiNEMEQwRDAiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPjwvc3ZnPg==";

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  
  const tshirtImgRef = useRef<HTMLImageElement | null>(null);
  
  // Model References to keep them from garbage collection
  const poseModelRef = useRef<any>(null);
  const latestPoseRef = useRef<PoseResults | null>(null);

  useEffect(() => {
    // Preload T-Shirt image
    const img = new Image();
    img.src = TSHIRT_SRC;
    tshirtImgRef.current = img;

    // Initialize MediaPipe Pose
    const PoseObj = (window as any).Pose;
    const pose = new PoseObj({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1, // 0 = Fast, 1 = Better, 2 = Best (Avoid 2 for web real-time)
      smoothLandmarks: true,
      enableSegmentation: true, // Pose has built-in segmentation! We can use this instead of a 2nd model right away
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

    // Match canvas internal resolution to video source resolution for sharp rendering
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Camera Feed
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const results = latestPoseRef.current;

    // 2. Compute T-Shirt transformation if pose is found
    if (results && results.poseLandmarks && tshirtImgRef.current) {
      const landmarks = results.poseLandmarks;
      
      // Get Shoulder and Hip coordinates (normalized 0 to 1)
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12];
      const leftHip = landmarks[23];
      // Calculate width and scale to 1.1x shoulder width
      const shirtWidth = Math.abs(rightShoulder.x - leftShoulder.x) * canvas.width * 1.1;
      
      // Note: Since MediaPipe coordinates might be horizontally flipped depending on the camera,
      // Left shoulder might have a higher X value than Right shoulder. We use Math.min to find the 
      // left-most visible shoulder on the canvas to start the rect.
      const minX = Math.min(leftShoulder.x, rightShoulder.x);
      
      const shirtX = minX * canvas.width - shirtWidth * 0.05; // Center the slight 1.1x overlap
      const shirtY = Math.min(leftShoulder.y, rightShoulder.y) * canvas.height;
      
      // T-shirt is roughly 1.2x taller than wide
      const shirtHeight = shirtWidth * 1.2;
      
      // --- ADVANCED OCCLUSION / MASKING OPPORTUNITY HERE ---
      // If you want the shirt to be clipped strictly to the torso, you can use the Segmentation Mask
      // from `results.segmentationMask`. 
      
      ctx.drawImage(
        tshirtImgRef.current, 
        shirtX, 
        shirtY, 
        shirtWidth, 
        shirtHeight
      );
      
      ctx.restore();
    } else {
      ctx.restore();
    }

    // Loop
    requestAnimationFrame(drawFrame);
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
      
      // Start render loop
      requestAnimationFrame(drawFrame);
      
    } catch (err) {
      console.error("Error starting camera: ", err);
      setIsModelLoading(false);
      alert("Please allow camera access to load the try-on experience.");
    }
  };

  return (
    <div className="tryon-container">
      <div className="header">
        <h1 className="title">Virtual Try-On</h1>
        <p className="subtitle">Real-time AR powered by MediaPipe Pose.</p>
      </div>

      <div className="camera-box">
        <div className="status-badge">
          <span className={`status-dot ${isCameraActive ? 'active' : ''}`}></span>
          {isCameraActive ? 'LIVE' : 'OFFLINE'}
        </div>

        {isModelLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Warming up AI Engine...</p>
          </div>
        )}

        {/* Hidden video source */}
        <video ref={videoRef} className="video-element" playsInline></video>

        {/* The compositing canvas */}
        <canvas ref={canvasRef} className="output-canvas"></canvas>
      </div>

      <div className="controls">
        {!isCameraActive && (
          <button className="btn primary" onClick={startCamera}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
              <circle cx="12" cy="13" r="4"></circle>
            </svg>
            Start Camera & AR Engine
          </button>
        )}
        {isCameraActive && (
          <button className="btn" onClick={() => window.location.reload()}>
             Stop Session
          </button>
        )}
      </div>
    </div>
  );
};

export default App;

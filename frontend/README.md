# 🖥️ V-TryOn Frontend — Real-Time AR Client

<div align="center">

![React](https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20WASM-4285F4?style=for-the-badge&logo=google&logoColor=white)

**High-performance, client-side Augmented Reality (AR) try-on interface capable of 30+ FPS pose tracking and dynamic mesh deformation directly in the browser.**

</div>

---

## 🌟 Key Capabilities

- **Zero-Latency In-Browser Landmark Detection:** Executes Google MediaPipe Pose via WebAssembly (WASM), detecting 33 distinct full-body 3D keypoints without sending video streams to a remote server.
- **Mathematical Dense Mesh Warping (`warpUtils.ts`):** Implements dynamic perspective warping, shoulder-to-hip vector alignment, and non-linear torso scaling on HTML5 2D Canvas.
- **Real-Time Calibration HUD:** Interactive slider controls allowing users and testers to calibrate fit scaling, Y-axis vertical offset, and view live FPS metrics.
- **Garment Upload & Matting Integration:** Seamless file upload handler connecting directly to the CV backend's U2-Net matting endpoint for instant background removal and apparel preview.

---

## 🏗️ Architecture & Component Flow

```text
[Webcam Stream]
      │
      ▼
[MediaPipe Pose WASM] ──(33 Landmarks)──► [Landmark Smoothing & Vector Math]
                                                      │
                                                      ▼
[Garment Texture / SVG] ─────────────────► [Dense Mesh / Affine Warper]
                                                      │
                                                      ▼
                                            [HTML5 Canvas Render (30+ FPS)]
```

---

## 📁 Source Code Organization

```text
frontend/
├── public/                 # Static assets (default garment textures, icons)
├── src/
│   ├── App.tsx             # Primary AR canvas view, video processor & HUD
│   ├── warpUtils.ts        # Affine transform & dense mesh warping algorithms
│   ├── index.css           # Glassmorphism dark-mode theme & HUD controls
│   ├── main.tsx            # React application root
│   └── vite-env.d.ts       # Vite & MediaPipe TypeScript type declarations
├── package.json            # Dependencies & scripts
└── vite.config.ts          # Vite build & bundler configuration
```

---

## 🚀 Quickstart & Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Dev Server
```bash
npm run dev
```
The client will start at `http://localhost:5173`.

### 3. Build for Production
```bash
npm run build
```

---

## 🎯 Key Engineering Highlights (For Evaluators)

1. **Client-Side Compute Offloading:** Runs all real-time video frames through client-side WebAssembly, ensuring 100% user privacy and eliminating server compute costs for video streaming.
2. **Double-Buffering & Offscreen Canvas:** Minimizes paint thrashing and GPU overhead by offloading intermediate affine transforms to an offscreen canvas before rendering to the main display.

# ⚙️ V-TryOn Backend Microservices

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-Python%203.10-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Express](https://img.shields.io/badge/Express-Node.js-000000?style=for-the-badge&logo=express&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

**Decoupled microservice architecture separating user session management and transactional logic from GPU-accelerated computer vision inference.**

</div>

---

## 🏛️ Architecture Overview

The backend is split into two specialized microservices:

```
                          ┌───────────────────────┐
                          │   Client Application  │
                          └───────────┬───────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│   App Gateway (:4000) │                           │   CV Engine (:8000)   │
│   Node.js + Express   │                           │   FastAPI + PyTorch   │
├───────────────────────┤                           ├───────────────────────┤
│ • User Authentication │                           │ • U2-Net Matting      │
│ • Wardrobe Metadata   │                           │ • DeepFashion ResNet  │
│ • Avatar Data Model   │                           │ • CP-VTON+ Pipeline   │
│ • PostgreSQL Database │                           │ • SMPL-X 3D Fitting   │
└───────────────────────┘                           └───────────────────────┘
```

---

## 📂 Subdirectories

### 1. `backend/cv/` — Computer Vision & Deep Learning Engine
- **Framework:** FastAPI, PyTorch, torchvision, OpenCV, U2-Net (`rembg`).
- **Core Endpoints:**
  - `POST /api/cv/segment-garment`: Sub-second foreground cloth extraction.
  - `POST /api/cv/classify-garment`: DeepFashion ResNet-50 50-category classifier with top-5 confidence ranking.
  - `POST /api/cv/try-on`: U-Net 2D neural try-on inference.
  - `POST /api/cv/smpl-estimate`: 3D SMPL-X parametric body estimation.
- **Phase 4A Pipeline (`backend/cv/vton/`):** Isolated end-to-end VTON pipeline integrating SCHP human parsing, HRNet pose estimation, and CP-VTON+ (GMM + TOM).

### 2. `backend/app/` — Application Gateway & Data Service
- **Framework:** Node.js, Express 5, TypeScript, PostgreSQL.
- **Core Responsibilities:**
  - JWT-based authentication and user session control.
  - User profiles, wardrobe catalogs, and try-on session persistence.

---

## 🐳 Containerized Deployment

Both services along with the PostgreSQL database can be started using the root `docker-compose.yml`:

```bash
docker-compose up --build
```

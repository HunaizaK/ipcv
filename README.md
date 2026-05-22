# Elderly Care Monitoring System

An AI-powered real-time human action detection system for elderly care. Uses a 3D CNN model (X3D-M) to detect health-related actions from live video and automatically alerts caregivers when serious events are detected.

## Features

- Real-time action detection via webcam or uploaded video
- Detects 9 health-related actions with confidence scoring
- Automatic email alerts for 5 serious actions (fall, stagger, nausea, chest pain, headache)
- WebSocket-based live streaming to a React dashboard
- Prediction smoothing to reduce false positives

## Tech Stack

| Layer | Technology |
|---|---|
| Model | X3D-M (PyTorchVideo, pretrained on Kinetics) |
| Backend | FastAPI + Uvicorn |
| Video Processing | OpenCV, FFmpeg |
| Frontend | React 19, Tailwind CSS |
| Communication | WebSockets |

## Detected Actions

| Action | Triggers Alert |
|---|---|
| Sneeze / Cough | |
| **Staggering** | **Yes** |
| **Falling** | **Yes** |
| **Touch Head (headache)** | **Yes** |
| **Touch Chest (chest pain)** | **Yes** |
| Touch Back (backache) | |
| Touch Neck (neckache) | |
| **Nausea / Vomiting** | **Yes** |
| Use Fan / Feeling Warm | |

## Project Structure

```
IPCV-Project/
├── api/
│   ├── main.py              # FastAPI server, WebSocket handler, email alerts
│   ├── train.py             # Model training script
│   ├── inference.py         # Core inference functions
│   ├── inference2.py        # CLI tool for offline video analysis
│   ├── ws_client.py         # WebSocket test client
│   └── best_action_model.pth
├── frontend/
│   └── src/
│       ├── App.js
│       └── pages/
│           ├── ElderlyMonitoringSystem.jsx
│           ├── ElderlyCareHomePage.jsx
│           └── SettingsPage.jsx
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- FFmpeg installed and on PATH
- CUDA-capable GPU (optional, CPU works)

### Backend

```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| WebSocket | `/ws/stream` | Live webcam stream with predictions |
| POST | `/api/detect-video` | Upload video for action analysis |
| GET | `/videos/{filename}` | Download processed output video |

### WebSocket Message Format

Messages are sent as `"action|confidence"`, e.g. `"falling|0.78"`.

## Model

- **Architecture:** X3D-M with a custom 9-class head
- **Dataset:** NTU RGB+D (classes 41–49, subjects P001–P040)
- **Input:** 16 frames × 224×224 pixels
- **Training split:** 80% train / 10% val / 10% test (subject-based)
- **Optimizer:** AdamW, lr=5e-5, weight decay=1e-4
- **Scheduler:** CosineAnnealingLR with early stopping (patience=7)

### Training

```bash
cd api
python train.py
```

Outputs `best_action_model.pth`, `confusion_matrix.png`, and `training_curves.png`.

### Offline Video Analysis

```bash
cd api
python inference2.py --video path/to/video.mp4
```

## Configuration

Key parameters in `api/main.py`:

```python
CONFIDENCE_THRESHOLD = 0.35   # Minimum confidence to count a prediction
PREDICTION_HISTORY_SIZE = 7   # Smoothing window (last N predictions)
STRIDE = 4                    # Process every Nth frame
ALERT_COOLDOWN = 10           # Seconds between repeat alerts for same action
```

## Email Alerts

Configure sender credentials in `api/main.py` using environment variables or directly. Alerts are sent via Gmail SMTP (port 587, STARTTLS) when a serious action is detected above the confidence threshold, subject to the cooldown period.

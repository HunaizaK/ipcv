# Elderly Monitoring System — Project Explanation
### Image Processing & Computer Vision (IPCV) — 8th Semester

---

## Project Kya Hai?

Ye ek **AI-powered Real-Time Human Action Detection System** hai jo camera ya uploaded video mein insaan ki movements dekh kar automatically pehchanta hai ke wo koi serious health issue toh nahi face kar raha — jaise girna, chakkar aana, ya sine mein dard.

Agar koi serious action detect ho toh system **automatic email alert** bhejta hai caretaker ko.

---

## Files Overview

| File | Kaam |
|---|---|
| `inference.py` | AI model load karna aur prediction karna |
| `main.py` | FastAPI server — WebSocket + REST API |
| `train.py` | Model ko dataset pe train karna |
| `inference2.py` | Command-line video analysis tool |
| `best_action_model.pth` | Trained model weights (saved best model) |
| `confusion_matrix.png` | Training evaluation result (visual) |
| `training_curves.png` | Loss aur accuracy curves |

---

## 9 Detectable Actions (Classes)

| # | Action | Serious? |
|---|---|---|
| 1 | Sneeze / Cough | No |
| 2 | **Staggering** (larkharana) | **YES** |
| 3 | **Falling** (girna) | **YES** |
| 4 | **Touch Head** (sar dard) | **YES** |
| 5 | **Touch Chest** (sine / dil ka dard) | **YES** |
| 6 | Touch Back (kamar dard) | No |
| 7 | Touch Neck (gardan dard) | No |
| 8 | **Nausea / Vomiting** (ulti aana) | **YES** |
| 9 | Use a Fan / Feeling Warm | No |

> **5 Serious Actions** trigger karte hain automatic email alert.

---

## Pipeline — Step by Step Flow

```
+---------------------------+
|   Camera / Video Input    |
+---------------------------+
             |
             v
+---------------------------+
|  OpenCV: 16 Frames Buffer |
|  (har 4th frame pe process)|
+---------------------------+
             |
             v
+----------------------------------+
|  Preprocessing                   |
|  - BGR  -->  RGB convert         |
|  - Resize: 224 x 224 pixels      |
|  - Normalize (mean=0.45, std=0.225)|
+----------------------------------+
             |
             v
+----------------------------------+
|  X3D-M Model (3D CNN)            |
|  - Input:  (1, C, 16, 224, 224)  |
|  - Output: 9 class logits        |
+----------------------------------+
             |
             v
+----------------------------------+
|  Softmax                         |
|  - 9 probabilities (sum = 1.0)   |
|  - Top prediction = action       |
|  - Max probability = confidence  |
+----------------------------------+
             |
             v
+----------------------------------+
|  PredictionSmoother              |
|  - Last 7 predictions store karo |
|  - Confidence-weighted voting    |
|  - Flickering / noise remove     |
+----------------------------------+
             |
             v
+----------------------------------+
|  Output: Action + Confidence     |
+----------------------------------+
       |                |
       v                v
[Serious Action?]   [Frontend]
       |            WebSocket se
       v            screen pe show
[Email Alert]
[Gmail SMTP]
```

---

## inference.py — AI Brain

```
Model:        X3D-M (Facebook / PyTorchVideo)
Architecture: 3D Convolutional Neural Network
Input:        16 frames, each 224x224 pixels
Output:       9-class probability distribution
Device:       CUDA (GPU) if available, else CPU
```

### Functions

**`preprocess_video(frames)`**
- 16 frames leta hai
- BGR to RGB convert karta hai
- 224x224 resize karta hai
- Normalize karta hai
- PyTorch tensor banata hai: shape `(1, 3, 16, 224, 224)`

**`predict_action(frames, return_confidence=True)`**
- Preprocessed tensor model mein deta hai
- Softmax apply karta hai
- Action naam aur confidence return karta hai

**`predict_action_with_probabilities(frames)`**
- Sabhi 9 classes ki probabilities return karta hai
- Debugging ke liye useful

---

## main.py — FastAPI Server

### Configuration

| Parameter | Value | Matlab |
|---|---|---|
| `PREDICTION_HISTORY_SIZE` | 7 | Last 7 predictions smooth karo |
| `CONFIDENCE_THRESHOLD` | 0.35 (35%) | Is se kam confidence ignore karo |
| `STRIDE` | 4 | Har 4th frame pe process karo |
| `TARGET_FPS` | 30 | Webcam 30 FPS |
| `ALERT_COOLDOWN` | 10 sec | Ek action pe 10 sec mein ek alert |

### Endpoint 1: `/ws/stream` — Live WebSocket

```
Client connect kare
      |
      v
Webcam khule (640x480, 30fps)
      |
      v
Loop: Har frame read karo
      |
      v
16 frames buffer bharo
      |
      v
Har 4th frame pe:
  - predict_action() call karo
  - PredictionSmoother mein daalo
  - Smoothed result frontend ko bhejo: "action|confidence"
      |
      v
Serious action + confidence >= 35%?
  - Last alert se 10 sec guzar gaye?
    - YES -> Email bhejo, time record karo
```

### Endpoint 2: `/api/detect-video` — Video Upload

```
User video upload kare
      |
      v
Temp file mein save karo
      |
      v
Non-MP4? -> ffmpeg se MP4 convert karo
      |
      v
OpenCV se frame-by-frame read karo
      |
      v
Har frame pe:
  - 16-frame buffer bharo
  - Har 4th frame pe predict karo
  - Smoother se action lo
  - Frame pe text overlay lagao: "action (XX.X%)"
  - Serious action? -> Email bhejo
      |
      v
Temporary video save karo (OpenCV mp4v)
      |
      v
ffmpeg se H.264 encode karo (browser compatible)
      |
      v
Final URL return karo + stats
```

### PredictionSmoother Class

```
Purpose: Predictions stable karna (noise/flickering hatana)

How it works:
1. Last 7 predictions aur confidence store karo (deque)
2. Confidence < 35%? Ignore karo
3. Baaki predictions ko confidence ke hisaab se weight do
4. Sab se zyada weighted action = final result
5. Ye last_stable_action bhi track karta hai
```

---

## train.py — Model Training

### Dataset

```
Name:     NTU RGB+D Dataset
Classes:  41 to 49 (9 health-related actions)
Format:   .avi video files
Split:    Subject-based (person ID se)
```

| Split | Subjects | Purpose |
|---|---|---|
| Train | P001 - P032 | Model seekhe |
| Validation | P033 - P036 | Training monitor kare |
| Test | P037 - P040 | Final evaluation |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Model | X3D-M (pretrained=True) |
| Epochs | 10 max |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| LR Scheduler | CosineAnnealingLR |
| Gradient Clipping | max_norm = 1.0 |
| Early Stopping | patience = 7 |
| Loss Function | CrossEntropyLoss (weighted) |
| Input Frames | 16 per clip |
| Frame Size | 224 x 224 |

### Data Augmentation (Training only)

| Augmentation | Probability |
|---|---|
| Random Horizontal Flip | 50% |
| Random Brightness (0.8x - 1.2x) | 50% |

### Class Weights

Imbalanced dataset ke liye class weights calculate karta hai:
```
weight = total_samples / (num_classes × class_count)
```
Iska matlab: kam samples wali class ko zyada importance milti hai.

### Output Files

| File | Content |
|---|---|
| `best_action_model.pth` | Best validation accuracy wala model |
| `confusion_matrix.png` | 9x9 heatmap — kaunsa class kahan galat predict hua |
| `training_curves.png` | Loss curves, accuracy curves, confidence distribution |

---

## inference2.py — Standalone CLI Tool

### Usage

```bash
# Sliding window mode (default, zyada accurate)
python inference2.py video.mp4

# Uniform sampling mode (fast, single pass)
python inference2.py video.mp4 --mode uniform

# Custom model weights
python inference2.py video.mp4 --model my_model.pth
```

### Two Modes

**Sliding Window Mode:**
```
Poori video pe 16-frame window slide karo (step=8)
Har window pe prediction lo
Confidence >= 35%? List mein daalo
Confidence-weighted voting karo
Best action return karo
```

**Uniform Mode:**
```
Poori video mein se 16 frames uniformly sample karo
Sirf ek baar predict karo
Action return karo
```

### Output Example

```
==================================================
  Detected action : falling
  Confidence      : 78.3%
  Clips evaluated : 24
  Vote breakdown  :
    falling                                       82.1%
    staggering                                    17.9%
==================================================
```

---

## Complete Tech Stack

| Layer | Technology | Version/Detail |
|---|---|---|
| Language | Python | 3.11 / 3.12 |
| Backend Framework | FastAPI | Async, WebSocket support |
| AI Model | X3D-M | PyTorchVideo (Facebook Research) |
| Deep Learning | PyTorch | CUDA support |
| Video Processing | OpenCV (cv2) | Frame read/write/overlay |
| Video Encoding | FFmpeg | H.264, AAC, browser-compatible |
| Real-time Comm. | WebSocket | ws:// protocol |
| CORS | FastAPI Middleware | All origins allowed |
| Email Alerts | smtplib + Gmail SMTP | TLS port 587 |
| Data Augmentation | NumPy | Flip, brightness |
| Metrics/Plots | matplotlib, seaborn | Confusion matrix, curves |
| Model Evaluation | sklearn | classification_report |

---

## Model Architecture: X3D-M

```
X3D = Expanding Architectures for Efficient Video Understanding
M   = Medium variant

Input:  (Batch, 3 channels, 16 frames, 224px, 224px)
          |
    [3D Convolutions]
    (spatial + temporal features ek saath)
          |
    [5 Residual Blocks]
          |
    [Block 5 Projection Head]  <-- Yahan hum ne modify kiya
    Original: 400 classes (Kinetics dataset)
    Modified: 9 classes (hamare actions)
          |
    [Softmax]
          |
    Output: 9 probabilities
```

---

## Alert System

```
Trigger Conditions:
1. Action = serious action (5 mein se koi ek)
2. Confidence >= 35%
3. Last alert se 10+ seconds guzar gaye hon

Email Details:
- From:    aliya10akhtar3a@gmail.com
- To:      aminah30akhtar3a@gmail.com
- Subject: "Alert: Serious Action Detected - [action]"
- Body:    Action naam + timestamp
- Server:  smtp.gmail.com:587 (STARTTLS)
```

---

## API Endpoints Summary

| Method | Endpoint | Kaam |
|---|---|---|
| GET | `/` | API info + configuration |
| GET | `/health` | Server health check |
| WebSocket | `/ws/stream` | Live webcam streaming |
| POST | `/api/detect-video` | Video file upload + analysis |
| Static | `/videos/{filename}` | Processed video download |

---

## Data Flow Diagram

```
                    +-------------+
                    |   Frontend  |
                    | (React.js)  |
                    +------+------+
                           |
              WebSocket /ws/stream
              or POST /api/detect-video
                           |
                    +------v------+
                    |   FastAPI   |
                    |  (main.py)  |
                    +------+------+
                           |
                    +------v------+
                    | inference.py|
                    |  X3D-M AI   |
                    +------+------+
                           |
              +------------+------------+
              |                         |
    +---------v--------+    +-----------v--------+
    | PredictionSmoother|   | Email Alert System  |
    | (7-frame buffer)  |   | (Gmail SMTP)        |
    +------------------+    +--------------------+
              |
    +---------v--------+
    | Output to Client |
    | action|confidence|
    +------------------+
```

---

## Key Numbers at a Glance

| Metric | Value |
|---|---|
| Total Action Classes | 9 |
| Serious / Alert Actions | 5 |
| Input Frames per Prediction | 16 |
| Frame Resolution | 224 x 224 px |
| Confidence Threshold | 35% |
| Smoothing Window | 7 predictions |
| Processing Stride | Every 4th frame |
| Alert Cooldown | 10 seconds |
| Max Training Epochs | 10 |
| Batch Size | 4 |
| Learning Rate | 0.00005 |
| Training Split | 80% train / 10% val / 10% test |

---

*IPCV Project — 8th Semester*

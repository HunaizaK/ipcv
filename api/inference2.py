"""
Simple video action classifier.

Usage:
    python inference2.py input.mp4
    python inference2.py clip.avi --model best_action_model.pth
    python inference2.py clip.mp4 --mode sliding   # sliding-window vote (default)
    python inference2.py clip.mp4 --mode uniform   # single-shot uniform sample
"""

import argparse
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import Normalize
from torchvision.transforms import Compose

# ==================== CONFIG ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
FRAMES = 16
SIZE = 224
STRIDE = 8                  # sliding-window step (frames between clips)
CONFIDENCE_THRESHOLD = 0.35

CLASS_NAMES = [
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea/vomiting",
    "use a fan / feeling warm",
]
# ================================================

_transform = Compose([
    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
])


def load_model(weights_path: str) -> torch.nn.Module:
    print(f"Loading model on {DEVICE} from '{weights_path}' ...")
    model = x3d_m(pretrained=False)
    model.blocks[5].proj = torch.nn.Linear(
        model.blocks[5].proj.in_features, NUM_CLASSES
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert a list of BGR frames to a normalised (1, C, T, H, W) tensor."""
    processed = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (SIZE, SIZE))
        processed.append(resized)
    clip = np.stack(processed)                                  # (T, H, W, C)
    clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2)  # (C, T, H, W)
    clip = _transform(clip)
    return clip.unsqueeze(0).to(DEVICE)


def predict_clip(model: torch.nn.Module,
                 frames: List[np.ndarray]) -> Tuple[str, float]:
    """Run the model on exactly FRAMES frames. Returns (action, confidence)."""
    tensor = frames_to_tensor(frames)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs).item()
        idx = torch.argmax(logits, dim=1).item()
    return CLASS_NAMES[idx], confidence


def read_all_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def infer_uniform(model: torch.nn.Module, video_path: str) -> dict:
    """Sample FRAMES frames uniformly across the video — single inference pass."""
    all_frames = read_all_frames(video_path)
    n = len(all_frames)
    if n == 0:
        raise RuntimeError("Video has no readable frames.")

    if n < FRAMES:
        # repeat last frame to pad
        all_frames += [all_frames[-1]] * (FRAMES - n)
        indices = list(range(FRAMES))
    else:
        indices = np.linspace(0, n - 1, FRAMES, dtype=int).tolist()

    clip = [all_frames[i] for i in indices]
    action, conf = predict_clip(model, clip)
    return {"action": action, "confidence": conf, "clips_evaluated": 1}


def infer_sliding(model: torch.nn.Module, video_path: str) -> dict:
    """
    Slide a 16-frame window over the video, collect predictions, return the
    action with the highest weighted vote (confidence-weighted majority).
    """
    all_frames = read_all_frames(video_path)
    n = len(all_frames)
    if n == 0:
        raise RuntimeError("Video has no readable frames.")

    if n < FRAMES:
        all_frames += [all_frames[-1]] * (FRAMES - n)
        n = len(all_frames)

    predictions: List[Tuple[str, float]] = []
    start = 0
    while start + FRAMES <= n:
        clip = all_frames[start: start + FRAMES]
        action, conf = predict_clip(model, clip)
        if conf >= CONFIDENCE_THRESHOLD:
            predictions.append((action, conf))
        start += STRIDE

    if not predictions:
        # fall back: just use the last clip even if below threshold
        clip = all_frames[-FRAMES:]
        action, conf = predict_clip(model, clip)
        predictions = [(action, conf)]

    # Confidence-weighted vote
    weights: dict = {}
    for action, conf in predictions:
        weights[action] = weights.get(action, 0.0) + conf

    best_action = max(weights, key=lambda a: weights[a])
    total_weight = sum(weights.values())
    avg_conf = weights[best_action] / len(predictions)

    return {
        "action": best_action,
        "confidence": round(avg_conf, 4),
        "clips_evaluated": len(predictions),
        "vote_breakdown": {a: round(w / total_weight, 3) for a, w in weights.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Classify action in a video file.")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--model", default="best_action_model.pth",
                        help="Path to trained weights (default: best_action_model.pth)")
    parser.add_argument("--mode", choices=["sliding", "uniform"], default="sliding",
                        help="'sliding' = window vote (default), 'uniform' = single-shot")
    args = parser.parse_args()

    model = load_model(args.model)

    print(f"Running {args.mode} inference on '{args.input}' ...")
    if args.mode == "uniform":
        result = infer_uniform(model, args.input)
    else:
        result = infer_sliding(model, args.input)

    print("\n" + "=" * 50)
    print(f"  Detected action : {result['action']}")
    print(f"  Confidence      : {result['confidence'] * 100:.1f}%")
    print(f"  Clips evaluated : {result['clips_evaluated']}")
    if "vote_breakdown" in result:
        print("  Vote breakdown  :")
        for action, share in result["vote_breakdown"].items():
            print(f"    {action:<45} {share * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

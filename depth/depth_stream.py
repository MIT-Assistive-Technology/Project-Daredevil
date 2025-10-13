#!/usr/bin/env python3
"""
depth_stream.py — Live monocular depth using Hugging Face DPT model via transformers.

Model: Intel/dpt-hybrid-midas (a MiDaS 3.0 DPT model)
Docs: https://huggingface.co/Intel/dpt-hybrid-midas
API: https://huggingface.co/docs/transformers/model_doc/dpt

What this does currently:
- Captures video from the built-in or external camera
- Runs MiDaS (DPT) monocular depth on each frame
    - Source: https://github.com/isl-org/MiDaS
    - A bit slow this way, we can see the delay
- Displays side-by-side comparison of regular feed and colorized depth feed
- Prints simple depth stats for a center ROI each frame
    - ROI region of interest will eventually be the object we want to "hone in" on, or
        rather strengthen the signal from that area (extracting the object of interest from
        the background noise)
- Optional saving of raw depth (float32 .npy) and PNGs.

To do:
- Test how fast and lightweight the model really can be--we are running this on a laptop, after all
- Integrate with object detection to isolate the object of interest
- Work on filtering and stabilizing the depth signal of an object of interest over time
- Provide relative depth information based on known object sizes, or even just based on
    the size of the object in the frame (e.g. a face) relative to some baseline or reference
    in the background (like a wall, floor, table, etc.)
- Play around with the relative values, because just from looking at the ROI stats,
    it seems that the final signal used for the spatial audio is not just the median value,
    but it needs to be relative to the background or something else...

Installation:
    python3 -m venv env && source env/bin/activate
    pip install --upgrade pip
    pip install numpy opencv-python torch torchvision transformers huggingface_hub timm

Flags for running the script:
    # Basic run (with default camera and size):
    python depth_stream.py

    # Custom camera, size, horizontal flip, and save snapshots to 'out' directory:
    python3 depth_stream.py --camera 0 --width 1280 --height 720 --flip --save-dir out

    # Useful flags:
    #   --camera 0                  # choose camera index
    #   --width 1280 --height 720   # request capture size
    #   --model Intel/dpt-hybrid-midas  # model ID from HuggingFace
    #   --roi 0.2                   # center ROI fraction (0..0.6)
    #   --save-dir out              # save snapshots with 's'
    #   --flip                      # horizontal flip the RGB feed (selfie view)

Key commands during program execution:
    q  #  Quit
    s  #  Save current RGB, depth-color PNG, and raw depth .npy to --save-dir

Notes:
- Monocular depth is RELATIVE (up to scale/shift) --> so do not treat values as any particular units.
- Use raw float map for logic, as the colorized map is for visualization (but it's fun to look at for sure).
- First run will hopefully download model weights to the HuggingFace cache, then it works offline? But
    am unsure how lightweight the model really is...
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation

# ----- Depth estimation and visualization via Hugging Face DPT model -----

class HFDepthEstimator:
    """
    Hugging Face DPT depth estimator (Intel/dpt-hybrid-midas), fully using transformers
    """

    def __init__(self, model_id: str = "Intel/dpt-hybrid-midas", device: torch.device | None = None):
        self.device = device or self.pick_device()
        # Image processor handles resize/normalize and post-process back to original size
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = DPTForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

        # Warmup to reduce first-frame latency
        with torch.no_grad():
            dummy = np.zeros((384, 384, 3), dtype=np.uint8)
            inputs = self.processor(images=dummy, return_tensors="pt").to(self.device)
            _ = self.model(**inputs)

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Single-frame depth inference.

        Args:
            frame_bgr: HxWx3 uint8 (OpenCV BGR)
        Returns:
            depth_f32: HxW float32 relative depth, aligned to input size
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Post-process to the original frame size (H, W)
        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(frame_rgb.shape[0], frame_rgb.shape[1])]
        )
        # Predicted depth is a torch Tensor HxW
        depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        return depth

def colorize_depth(depth: np.ndarray, clip_percentile: float = 0.05) -> np.ndarray:
    """
    Colorizes the depth map using percentile clipping and the MAGMA colormap.
    """
    d = np.nan_to_num(depth, nan=float(np.nanmedian(depth)))
    lo = np.percentile(d, 100 * clip_percentile)
    hi = np.percentile(d, 100 * (1.0 - clip_percentile))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d_8u = (d_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_8u, cv2.COLORMAP_MAGMA)

# ----- Region of Interest (ROI) stats and drawing -----

def center_roi_stats(depth: np.ndarray, roi_frac: float = 0.2) -> dict:
    """
    Compute robust stats over a centered square ROI (fraction of min(H, W)).
    """
    h, w = depth.shape
    size = int(min(h, w) * roi_frac)
    if size < 2:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}

    cx, cy = w // 2, h // 2
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    roi = depth[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    roi = roi[np.isfinite(roi)]

    if roi.size == 0:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(roi)),
        "p10": float(np.percentile(roi, 10)),
        "p90": float(np.percentile(roi, 90)),
    }

def draw_center_roi(img: np.ndarray, roi_frac: float = 0.2, color=(255, 255, 255)) -> None:
    """
    Drawing a region of interest (ROI) rectangle that will eventually be the object we care about
    """
    h, w = img.shape[:2]
    size = int(min(h, w) * roi_frac)
    if size < 2:
        return
    cx, cy = w // 2, h // 2
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

# ----- Camera handling and snapshot saving -----

def open_camera(idx: int, width: int, height: int):
    """
    OpenCV camera handling and snapshot saving
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try --camera 1 or another index.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    return cap

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon acceleration
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_snapshot(save_dir: str, rgb_bgr: np.ndarray, depth_f32: np.ndarray, depth_color_bgr: np.ndarray):
    """
    Saves RGB PNG, colorized depth PNG, and raw depth .npy to the specified directory.
    """
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(save_dir, f"rgb_{ts}.png"), rgb_bgr)
    cv2.imwrite(os.path.join(save_dir, f"depth_color_{ts}.png"), depth_color_bgr)
    np.save(os.path.join(save_dir, f"depth_raw_{ts}.npy"), depth_f32)
    print(f"[saved] rgb_{ts}.png, depth_color_{ts}.png, depth_raw_{ts}.npy")

# ----- Running the depth streaming test -----

def parse_args():
    p = argparse.ArgumentParser(description="Live depth via transformers.")
    p.add_argument("--camera", type=int, default=0, help="Camera index.")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    p.add_argument("--height", type=int, default=720, help="Requested capture height.")
    p.add_argument("--flip", action="store_true", help="Flip camera feed horizontally.")
    p.add_argument("--roi", type=float, default=0.2, help="Center ROI fraction (0..0.6).")
    p.add_argument("--save-dir", type=str, default="", help="Directory to save snapshots, using 's' command.")
    return p.parse_args()

def main():
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")

    estimator = HFDepthEstimator(model_id="Intel/dpt-hybrid-midas", device=device)
    cap = open_camera(args.camera, args.width, args.height)

    cv2.namedWindow("RGB | Depth", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            if args.flip:
                frame_bgr = cv2.flip(frame_bgr, 1)

            depth = estimator.infer(frame_bgr)  # HxW float32 (relative)
            depth_color = colorize_depth(depth, clip_percentile=0.05)

            # Overlays
            draw_center_roi(frame_bgr, args.roi)
            draw_center_roi(depth_color, args.roi)
            stats = center_roi_stats(depth, args.roi)
            cv2.putText(
                frame_bgr,
                f"Intel/dpt-hybrid-midas",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_color,
                f"ROI median: {stats['median']:.3f} | p10: {stats['p10']:.3f} p90: {stats['p90']:.3f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Putting streams side-by-side
            if frame_bgr.shape != depth_color.shape:
                depth_color = cv2.resize(depth_color, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            stacked = np.hstack([frame_bgr, depth_color])
            cv2.imshow("RGB | Depth", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and args.save_dir:
                save_snapshot(args.save_dir, frame_bgr, depth, depth_color)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp


# ------------------------------
# Utility: filesystem
# ------------------------------
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ------------------------------
# Detector: FaceDetection -> Crop -> FaceMesh
# More robust than FaceMesh alone (works when face is small/off-center)
# ------------------------------
class FaceCropMeshDetector:
    """
    Step 1: MediaPipe FaceDetection -> bounding box
    Step 2: Crop ROI -> MediaPipe FaceMesh on ROI
    Return selected landmarks back in ORIGINAL image pixel coordinates.
    """

    def __init__(
        self,
        min_det_conf: float = 0.3,
        min_mesh_det_conf: float = 0.3,
        min_mesh_track_conf: float = 0.3,
        refine_landmarks: bool = True,
        model_selection: int = 0,   # 0 short-range, 1 long-range
        pad: float = 0.25          # crop padding ratio
    ):
        self.face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_det_conf
        )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_mesh_det_conf,
            min_tracking_confidence=min_mesh_track_conf,
        )

        # Eyes, nose, mouth points (few points = lightweight)
        self.idx = [33, 133, 362, 263, 1, 61, 291, 13, 14]
        self.pad = pad

    def detect(self, bgr_frame: np.ndarray):
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # --- Step 1: Face detection ---
        det_res = self.face_det.process(rgb)
        if not det_res.detections:
            return None

        best = max(det_res.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Expand bbox by padding
        pad = self.pad
        x1 = int(x1 - pad * bw)
        y1 = int(y1 - pad * bh)
        x2 = int(x1 + (1 + 2 * pad) * bw)
        y2 = int(y1 + (1 + 2 * pad) * bh)

        # Clamp to image
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi_bgr = bgr_frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        # --- Step 2: Face mesh on ROI ---
        mesh_res = self.face_mesh.process(roi_rgb)
        if not mesh_res.multi_face_landmarks:
            return None

        face = mesh_res.multi_face_landmarks[0].landmark
        rh, rw = roi_bgr.shape[:2]

        pts = []
        for i in self.idx:
            x = int(face[i].x * rw) + x1
            y = int(face[i].y * rh) + y1
            pts.append([x, y])

        return np.array(pts, dtype=np.float32)


# ------------------------------
# Similarity alignment (Procrustes-like)
# Removes normal head motion (translation/rotation/scale)
# ------------------------------
def estimate_similarity_transform(A: np.ndarray, B: np.ndarray):
    """
    Estimate similarity transform mapping A to B:
    scale + rotation + translation.
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    A0 = A - mu_A
    B0 = B - mu_B

    cov = (A0.T @ B0) / A.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_A = (A0 ** 2).sum() / A.shape[0]
    s = (S.sum() / var_A) if var_A > 1e-9 else 1.0
    t = mu_B - s * (R @ mu_A)

    return s, R, t


def apply_similarity_transform(A: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    """Apply similarity transform to points A."""
    return (s * (A @ R.T)) + t


def jitter_score(A: np.ndarray, B: np.ndarray) -> float:
    """
    Jitter score = RMS residual after aligning A -> B.
    Higher = more non-rigid inconsistency (deepfake-like warping/jitter).
    """
    s, R, t = estimate_similarity_transform(A, B)
    A_aligned = apply_similarity_transform(A, s, R, t)
    residual = B - A_aligned
    rms = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
    return float(rms)


# ------------------------------
# Debug visualization
# ------------------------------
def draw_landmarks(frame: np.ndarray, pts: np.ndarray):
    """Draw selected landmarks on frame."""
    out = frame.copy()
    for (x, y) in pts.astype(int):
        cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
    return out


# ------------------------------
# Main pipeline: extract golden frames + save CSV
# ------------------------------
def extract_golden_frames(
    video_path: str,
    out_dir: str,
    save_debug_landmarks: bool = False,
    save_all_frames: bool = False,
    max_frames: int = None,
    threshold_mode: str = "percentile",
    threshold_value: float = 90.0,
    min_gap: int = 3,
    sample_fps: float = None,     # if None -> process all frames
):
    """
    Extract golden frames using landmark jitter sampling.

    Saves:
      - output/<video_name>/scores.csv
      - output/<video_name>/golden_frames/*.jpg
      - output/<video_name>/landmarks_debug/*.jpg (optional)
      - output/<video_name>/frames_all/*.jpg (optional)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_out = os.path.join(out_dir, video_name)

    golden_dir = os.path.join(base_out, "golden_frames")
    dbg_dir = os.path.join(base_out, "landmarks_debug")
    all_dir = os.path.join(base_out, "frames_all")

    ensure_dir(base_out)
    ensure_dir(golden_dir)
    if save_debug_landmarks:
        ensure_dir(dbg_dir)
    if save_all_frames:
        ensure_dir(all_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, Frames: {total}")
    if sample_fps is not None:
        step = max(int(round(fps / sample_fps)), 1)
        print(f"[INFO] Sampling: {sample_fps} fps (every {step} frames)")
    else:
        step = 1
        print("[INFO] Sampling: ALL frames")

    detector = FaceCropMeshDetector(
        min_det_conf=0.3,
        min_mesh_det_conf=0.3,
        min_mesh_track_conf=0.3,
        model_selection=1,   # long-range helps when face is smaller
        pad=0.30
    )

    frame_idx = 0
    processed = 0

    prev_pts = None
    scores = []

    detected_frames = 0
    valid_pairs = 0

    pbar = tqdm(total=total if total > 0 else None, desc=f"Processing {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # sampling logic
        if frame_idx % step != 0:
            frame_idx += 1
            if total > 0:
                pbar.update(1)
            continue

        processed += 1

        if save_all_frames:
            cv2.imwrite(os.path.join(all_dir, f"{frame_idx:06d}.jpg"), frame)

        pts = detector.detect(frame)

        if pts is not None:
            detected_frames += 1

            if save_debug_landmarks:
                dbg = draw_landmarks(frame, pts)
                cv2.imwrite(os.path.join(dbg_dir, f"{frame_idx:06d}.jpg"), dbg)

        score = None
        if pts is not None and prev_pts is not None:
            score = jitter_score(prev_pts, pts)
            valid_pairs += 1

        scores.append({
            "frame_idx": frame_idx,
            "jitter_score": score
        })

        # update prev only when face detected
        if pts is not None:
            prev_pts = pts

        frame_idx += 1
        if total > 0:
            pbar.update(1)

    pbar.close()
    cap.release()

    df = pd.DataFrame(scores)
    csv_path = os.path.join(base_out, "scores.csv")
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Sampled frames processed: {processed}")
    print(f"[INFO] Frames with landmarks: {detected_frames}")
    print(f"[INFO] Valid consecutive pairs: {valid_pairs}")
    print(f"[INFO] Scores saved: {csv_path}")

    valid = df.dropna(subset=["jitter_score"]).copy()
    if len(valid) == 0:
        print("[ERROR] No valid landmark transitions found. No golden frames extracted.")
        print("        (Face detection/mesh is failing OR face is missing in video.)")
        return

    # Threshold
    if threshold_mode == "percentile":
        thr = np.percentile(valid["jitter_score"].values, threshold_value)
    elif threshold_mode == "mean_std":
        mu = valid["jitter_score"].mean()
        sd = valid["jitter_score"].std(ddof=0)
        thr = mu + threshold_value * sd
    elif threshold_mode == "fixed":
        thr = float(threshold_value)
    else:
        raise ValueError("threshold_mode must be: percentile | mean_std | fixed")

    df["selected"] = False
    df.loc[df["jitter_score"].notna() & (df["jitter_score"] >= thr), "selected"] = True

    # Enforce min_gap between selected frames
    selected_rows = df.index[df["selected"]].tolist()
    kept = []
    last_kept = -10**9
    for r in selected_rows:
        f = int(df.loc[r, "frame_idx"])
        if f - last_kept >= min_gap * step:
            kept.append(r)
            last_kept = f

    df["selected"] = False
    df.loc[kept, "selected"] = True

    # Save final CSV with selected flag
    df.to_csv(csv_path, index=False)

    keep_frames = set(df.loc[df["selected"], "frame_idx"].astype(int).tolist())
    if len(keep_frames) == 0:
        print("[WARN] Threshold produced 0 golden frames. Try lowering threshold_value.")
        return

    # Re-read video and save selected frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    frame_idx = 0
    pbar = tqdm(total=len(keep_frames), desc=f"Saving golden frames {video_name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in keep_frames:
            cv2.imwrite(os.path.join(golden_dir, f"{frame_idx:06d}.jpg"), frame)
            pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    print(f"[DONE] Golden frames saved: {len(keep_frames)}")
    print(f"[DONE] Threshold mode={threshold_mode}, threshold={thr:.4f}")
    print(f"[DONE] Output folder: {base_out}")


# ------------------------------
# CLI entry point
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Golden Frames using Landmark Jitter Sampling")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--debug_landmarks", action="store_true", help="Save debug landmark plots")
    parser.add_argument("--save_all", action="store_true", help="Save all frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Process at most N frames")

    parser.add_argument("--threshold_mode", type=str, default="percentile",
                        choices=["percentile", "mean_std", "fixed"])
    parser.add_argument("--threshold_value", type=float, default=90.0,
                        help="Percentile (0-100) OR std-multiplier OR fixed RMS threshold")

    parser.add_argument("--min_gap", type=int, default=3, help="Min gap between golden frames (in sampled frames)")
    parser.add_argument("--sample_fps", type=float, default=None,
                        help="If set (e.g., 1), process ~1 frame per second. If None, process all frames.")

    args = parser.parse_args()

    extract_golden_frames(
        video_path=args.video,
        out_dir=args.out,
        save_debug_landmarks=args.debug_landmarks,
        save_all_frames=args.save_all,
        max_frames=args.max_frames,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        min_gap=args.min_gap,
        sample_fps=args.sample_fps
    )
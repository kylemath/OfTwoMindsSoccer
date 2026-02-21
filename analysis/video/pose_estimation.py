"""
MediaPipe Pose estimation pipeline for penalty kick video clips.

Processes each penalty clip through MediaPipe Pose to extract 33 body landmarks
per frame, handling variable frame rates and multiple people in frame.

Supports multi-person detection via YOLOv8 (person detection) + MediaPipe Pose
per person. GK/striker classification and IoU tracking maintain consistent IDs.
Falls back to single-person MediaPipe when YOLOv8 is unavailable.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

# COCO class id for person
YOLO_PERSON_CLASS_ID = 0

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union for two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detect_persons_yolo(frame: np.ndarray, yolo_model) -> list[tuple[int, int, int, int, float]]:
    """Run YOLOv8 and return list of (x1, y1, x2, y2, conf) for person detections."""
    results = yolo_model(frame, verbose=False)[0]
    boxes = []
    if results.boxes is None:
        return boxes
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == YOLO_PERSON_CLASS_ID:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf))
    return boxes


def _crop_with_padding(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding_frac: float = 0.2,
) -> tuple[np.ndarray, int, int]:
    """Crop frame to bounding box with padding. Returns (cropped, offset_x, offset_y)."""
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * padding_frac)
    pad_h = int(bh * padding_frac)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w, x2 + pad_w)
    ny2 = min(h, y2 + pad_h)
    cropped = frame[ny1:ny2, nx1:nx2].copy()
    return cropped, nx1, ny1


def _run_single_person_mediapipe(
    frame: np.ndarray,
    pose,
    frame_idx: int,
    timestamp_ms: float,
    person_id: int = 0,
    min_landmark_visibility: float = 0.3,
) -> list[dict]:
    """Run MediaPipe on full frame (single-person mode). Returns list of row dicts."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    rows = []
    if results.pose_landmarks:
        h, w = frame.shape[:2]
        for lm_id, lm in enumerate(results.pose_landmarks.landmark):
            vis = lm.visibility if lm.visibility >= min_landmark_visibility else 0.0
            rows.append({
                "frame": frame_idx,
                "timestamp_ms": round(timestamp_ms, 2),
                "person_id": person_id,
                "landmark_id": lm_id,
                "landmark_name": LANDMARK_NAMES[lm_id] if lm_id < len(LANDMARK_NAMES) else f"landmark_{lm_id}",
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(lm.z, 6),
                "visibility": round(vis, 4),
            })
    return rows


def _run_multi_person_mediapipe(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    person_ids: list[int],
    pose,
    frame_idx: int,
    timestamp_ms: float,
    full_h: int,
    full_w: int,
    min_landmark_visibility: float,
    track_ids: list[int] | None = None,
) -> list[dict]:
    """Run MediaPipe on each cropped person. Map landmarks to full-frame normalized coords."""
    rows = []
    for idx, ((x1, y1, x2, y2, _), pid) in enumerate(zip(boxes, person_ids)):
        cropped, cx, cy = _crop_with_padding(frame, x1, y1, x2, y2, padding_frac=0.2)
        ch, cw = cropped.shape[:2]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            tid = track_ids[idx] if track_ids is not None else None
            for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                # Map from crop-normalized [0,1] to full-frame normalized [0,1]
                px = lm.x * cw + cx
                py = lm.y * ch + cy
                norm_x = px / full_w
                norm_y = py / full_h
                vis = lm.visibility if lm.visibility >= min_landmark_visibility else 0.0
                r = {
                    "frame": frame_idx,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "person_id": pid,
                    "landmark_id": lm_id,
                    "landmark_name": LANDMARK_NAMES[lm_id] if lm_id < len(LANDMARK_NAMES) else f"landmark_{lm_id}",
                    "x": round(norm_x, 6),
                    "y": round(norm_y, 6),
                    "z": round(lm.z, 6),
                    "visibility": round(vis, 4),
                }
                if tid is not None:
                    r["_track_id"] = tid
                rows.append(r)
    return rows


def _classify_gk_striker(
    frame_boxes: list[list[tuple[int, int, int, int, float]]],
    total_frames: int,
) -> dict[int, int]:
    """
    Classify persons as GK (person_id=0) or striker (person_id=1).
    Uses first 25% of frames: person with smaller average y (higher in frame)
    is the goalkeeper.
    Returns mapping from internal track_id -> person_id (0=GK, 1=striker).
    """
    n_classify = max(1, int(total_frames * 0.25))
    cy_per_track: dict[int, list[float]] = {}
    for f_idx, boxes in enumerate(frame_boxes):
        if f_idx >= n_classify:
            break
        for track_id, (x1, y1, x2, y2, _) in boxes:
            cy = (y1 + y2) / 2
            if track_id not in cy_per_track:
                cy_per_track[track_id] = []
            cy_per_track[track_id].append(cy)
    if not cy_per_track:
        return {}
    avg_cy = {tid: np.mean(cys) for tid, cys in cy_per_track.items()}
    sorted_tracks = sorted(avg_cy.keys(), key=lambda t: avg_cy[t])
    # Smallest avg y (higher in frame) = goalkeeper = person_id 0
    mapping = {}
    for i, tid in enumerate(sorted_tracks):
        mapping[tid] = min(i, 1)  # 0=GK, 1=striker; if more than 2, extra get 1
    return mapping


def process_single_clip(
    video_path: Path,
    output_dir: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    min_landmark_visibility: float = 0.3,
) -> Path | None:
    """
    Run MediaPipe Pose on a single video clip.

    When YOLOv8 is available and 2+ persons are detected, uses multi-person pipeline:
    YOLOv8 person detection, GK/striker classification, IoU tracking, per-person
    MediaPipe. Otherwise falls back to single-person full-frame MediaPipe.

    Returns path to the output CSV, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        logger.warning("Could not determine FPS for %s; defaulting to %.0f", video_path.name, fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Processing %s — %d frames @ %.1f fps", video_path.name, total_frames, fps)

    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            logger.warning("Could not load YOLOv8: %s. Using single-person mode.", e)

    mp_pose = mp.solutions.pose
    rows = []
    prev_tracks: list[tuple[int, tuple[int, int, int, int, float], int]] = []
    next_track_id = 0
    track_last_seen: dict[int, int] = {}
    max_frames_lost = 10
    frame_boxes_for_classify: list[list[tuple[int, tuple[int, int, int, int, float]]]] = []
    track_to_person: dict[int, int] | None = None
    n_classify = max(1, int(total_frames * 0.25))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            h, w = frame.shape[:2]

            if yolo_model is not None:
                boxes_raw = _detect_persons_yolo(frame, yolo_model)
                if len(boxes_raw) >= 2:
                    # IoU tracking
                    current_tracks: list[tuple[int, tuple[int, int, int, int, float]]] = []
                    used_prev = set()

                    for (x1, y1, x2, y2, conf) in boxes_raw:
                        box_a = np.array([x1, y1, x2, y2])
                        best_iou, best_prev_idx = 0.0, -1
                        for pi, (tid, pbox, _) in enumerate(prev_tracks):
                            if pi in used_prev:
                                continue
                            iou = _box_iou(box_a, np.array([pbox[0], pbox[1], pbox[2], pbox[3]]))
                            if iou > best_iou:
                                best_iou, best_prev_idx = iou, pi

                        if best_prev_idx >= 0 and best_iou > 0.3:
                            tid = prev_tracks[best_prev_idx][0]
                            used_prev.add(best_prev_idx)
                            current_tracks.append((tid, (x1, y1, x2, y2, conf)))
                        else:
                            current_tracks.append((next_track_id, (x1, y1, x2, y2, conf)))
                            next_track_id += 1
                        track_last_seen[current_tracks[-1][0]] = frame_idx

                    for pi, (tid, pbox, last_seen) in enumerate(prev_tracks):
                        if pi in used_prev:
                            continue
                        if frame_idx - last_seen <= max_frames_lost:
                            current_tracks.append((tid, pbox))
                            track_last_seen[tid] = frame_idx

                    prev_tracks = [(tid, box, track_last_seen[tid]) for tid, box in current_tracks]

                    if frame_idx < n_classify:
                        frame_boxes_for_classify.append(current_tracks)

                    if frame_idx >= n_classify and track_to_person is None and frame_boxes_for_classify:
                        track_to_person = _classify_gk_striker(frame_boxes_for_classify, total_frames)
                        if not track_to_person:
                            all_tids = {tid for row in frame_boxes_for_classify for tid, _ in row}
                            track_to_person = {tid: 0 for tid in all_tids}

                    # Process only boxes from fresh YOLO detections (not held-over tracks)
                    det_boxes = []
                    det_track_ids = []
                    for (x1, y1, x2, y2, c) in boxes_raw:
                        box_a = np.array([x1, y1, x2, y2])
                        for tid, (px1, py1, px2, py2, _) in current_tracks:
                            if _box_iou(box_a, np.array([px1, py1, px2, py2])) > 0.5:
                                det_boxes.append((x1, y1, x2, y2, c))
                                det_track_ids.append(tid)
                                break

                    if det_boxes:
                        person_ids = [
                            track_to_person.get(tid, 0) if track_to_person else tid
                            for tid in det_track_ids
                        ]
                        track_ids_arg = det_track_ids if track_to_person is None else None
                        frame_rows = _run_multi_person_mediapipe(
                            frame, det_boxes, person_ids, pose,
                            frame_idx, timestamp_ms, h, w, min_landmark_visibility,
                            track_ids=track_ids_arg,
                        )
                        rows.extend(frame_rows)
                else:
                    frame_rows = _run_single_person_mediapipe(
                        frame, pose, frame_idx, timestamp_ms, person_id=0,
                        min_landmark_visibility=min_landmark_visibility,
                    )
                    rows.extend(frame_rows)
            else:
                frame_rows = _run_single_person_mediapipe(
                    frame, pose, frame_idx, timestamp_ms, person_id=0,
                    min_landmark_visibility=min_landmark_visibility,
                )
                rows.extend(frame_rows)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info("  %s: frame %d / %d", video_path.name, frame_idx, total_frames)

    cap.release()

    # Remap track_id -> person_id for rows from first 25% (before classification)
    for r in rows:
        if "_track_id" in r:
            tid = r["_track_id"]
            r["person_id"] = track_to_person.get(tid, 0) if track_to_person else tid
            del r["_track_id"]

    if not rows:
        logger.warning("No pose landmarks detected in %s", video_path.name)
        return None

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_pose.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d landmark rows to %s", len(df), out_path)
    return out_path


def _process_single_clip_wrapper(args: tuple) -> Path | None:
    """Wrapper for parallel processing: (video_path, output_dir, kwargs) -> output path."""
    video_path, output_dir, kwargs = args
    return process_single_clip(video_path, output_dir, **kwargs)


def process_directory(
    clips_dir: Path,
    output_dir: Path | None = None,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    parallel: int | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    min_landmark_visibility: float = 0.3,
) -> list[Path]:
    """
    Process all video clips in a directory. Returns list of output CSV paths.

    Args:
        clips_dir: Directory containing video files.
        output_dir: Output directory for pose CSVs (default: <clips_dir>/../pose_data).
        extensions: Video file extensions to process.
        parallel: If set, number of worker processes for parallel processing. None = sequential.
    """
    clips_dir = Path(clips_dir)
    if output_dir is None:
        output_dir = clips_dir.parent / "pose_data"

    video_files = sorted(
        f for f in clips_dir.iterdir()
        if f.suffix.lower() in extensions and not f.name.startswith(".")
    )

    if not video_files:
        logger.warning("No video files found in %s", clips_dir)
        return []

    logger.info("Found %d video files in %s", len(video_files), clips_dir)
    kwargs = {
        "min_detection_confidence": min_detection_confidence,
        "min_tracking_confidence": min_tracking_confidence,
        "min_landmark_visibility": min_landmark_visibility,
    }

    if parallel is not None and parallel >= 1:
        results = []
        tasks = [(vf, output_dir, kwargs) for vf in video_files]
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_process_single_clip_wrapper, t) for t in tasks]
            for future in as_completed(futures):
                try:
                    out = future.result()
                    if out is not None:
                        results.append(out)
                except Exception as e:
                    logger.exception("Worker failed: %s", e)
        logger.info("Processed %d / %d clips successfully.", len(results), len(video_files))
        return results

    results = []
    for vf in video_files:
        out = process_single_clip(vf, output_dir, **kwargs)
        if out is not None:
            results.append(out)

    logger.info("Processed %d / %d clips successfully.", len(results), len(video_files))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe Pose on penalty kick clips.")
    parser.add_argument("--clips-dir", type=Path, default=SCRIPT_DIR / "clips",
                        help="Directory containing video clips")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for pose CSVs (default: <clips_dir>/../pose_data)")
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--min-landmark-visibility", type=float, default=0.3,
                        help="Landmarks below this visibility are marked as 0 in output")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of worker processes for parallel processing (default: sequential)")
    args = parser.parse_args()

    process_directory(
        args.clips_dir,
        args.output_dir,
        parallel=args.parallel,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        min_landmark_visibility=args.min_landmark_visibility,
    )


if __name__ == "__main__":
    main()

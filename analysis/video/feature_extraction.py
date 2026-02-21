"""
Feature extraction pipeline for penalty kick pose data.

Computes theory-relevant kinematic features from raw MediaPipe pose CSVs,
mapping Tafazoli et al. (2026) shared neural subspace predictions to
measurable video-derived quantities.

Features extracted:
  P7 — Head-body dissociation angle
  P3 — Dive commitment onset timing
  P2 — Pre-dive postural asymmetry
  P3 — Striker hip rotation timing relative to GK dive commitment
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

# MediaPipe landmark indices
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _pivot_frame(df: pd.DataFrame, person_id: int = 0) -> pd.DataFrame:
    """Pivot raw pose CSV so each row is a frame with columns per landmark coordinate."""
    sub = df[df["person_id"] == person_id].copy()
    pivoted = sub.pivot_table(
        index=["frame", "timestamp_ms"],
        columns="landmark_name",
        values=["x", "y", "z", "visibility"],
        aggfunc="first",
    )
    pivoted.columns = [f"{val}_{lm}" for val, lm in pivoted.columns]
    return pivoted.reset_index().sort_values("frame")


def _smooth(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smooth (or uniform filter fallback for very short series)."""
    if len(series) < window:
        return series
    if window >= len(series):
        window = len(series) if len(series) % 2 == 1 else len(series) - 1
    try:
        return savgol_filter(series, window_length=max(window, 3), polyorder=min(2, max(window, 3) - 1))
    except Exception:
        return uniform_filter1d(series.astype(float), size=min(window, len(series)))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def compute_head_body_dissociation(pf: pd.DataFrame) -> pd.DataFrame:
    """
    P7: Head-body dissociation angle over time.

    Head direction = vector from mid-shoulder to nose (normalized image coords).
    Body momentum  = vector from hip-center(t-3) to hip-center(t).
    Returns DataFrame with columns: frame, timestamp_ms, head_angle_deg, body_angle_deg, dissociation_deg.
    """
    mid_shoulder_x = (pf["x_left_shoulder"] + pf["x_right_shoulder"]) / 2
    mid_shoulder_y = (pf["y_left_shoulder"] + pf["y_right_shoulder"]) / 2
    hip_center_x = (pf["x_left_hip"] + pf["x_right_hip"]) / 2
    hip_center_y = (pf["y_left_hip"] + pf["y_right_hip"]) / 2

    head_dx = pf["x_nose"].values - mid_shoulder_x.values
    head_dy = pf["y_nose"].values - mid_shoulder_y.values
    head_angles = np.degrees(np.arctan2(head_dy, head_dx))

    lag = 3
    body_dx = hip_center_x.values[lag:] - hip_center_x.values[:-lag]
    body_dy = hip_center_y.values[lag:] - hip_center_y.values[:-lag]
    body_angles = np.degrees(np.arctan2(body_dy, body_dx))
    body_angles = np.concatenate([np.full(lag, np.nan), body_angles])

    dissociation = np.abs(head_angles - body_angles)
    dissociation = np.where(dissociation > 180, 360 - dissociation, dissociation)

    return pd.DataFrame({
        "frame": pf["frame"].values,
        "timestamp_ms": pf["timestamp_ms"].values,
        "head_angle_deg": _smooth(head_angles),
        "body_angle_deg": _smooth(np.nan_to_num(body_angles, nan=0.0)),
        "dissociation_deg": _smooth(np.nan_to_num(dissociation, nan=0.0)),
    })


def compute_dive_commitment(pf: pd.DataFrame, fps: float = 30.0) -> dict:
    """
    P3: Dive commitment onset — frame where lateral hip velocity > 2 SD above baseline.

    Returns dict with commitment_frame, commitment_time_ms, baseline_mean, baseline_std,
    threshold, and full lateral velocity series.
    """
    hip_center_x = (pf["x_left_hip"].values + pf["x_right_hip"].values) / 2
    lateral_vel = np.gradient(_smooth(hip_center_x)) * fps
    baseline_window = max(1, len(lateral_vel) // 4)
    baseline = lateral_vel[:baseline_window]
    mean_b, std_b = np.mean(np.abs(baseline)), np.std(np.abs(baseline))
    threshold = mean_b + 2 * max(std_b, 1e-4)

    commitment_frame = None
    for i in range(baseline_window, len(lateral_vel)):
        if abs(lateral_vel[i]) > threshold:
            commitment_frame = int(pf["frame"].iloc[i])
            break

    commitment_time_ms = None
    if commitment_frame is not None:
        row = pf[pf["frame"] == commitment_frame]
        if len(row):
            commitment_time_ms = float(row["timestamp_ms"].iloc[0])

    return {
        "commitment_frame": commitment_frame,
        "commitment_time_ms": commitment_time_ms,
        "baseline_mean": float(mean_b),
        "baseline_std": float(std_b),
        "threshold": float(threshold),
        "lateral_velocity": lateral_vel,
    }


def compute_postural_asymmetry(pf: pd.DataFrame, goal_center_x: float = 0.5) -> dict:
    """
    P2: Pre-dive postural asymmetry — lateral offset of GK center-of-mass from goal center.

    We use the first 20% of frames as the "pre-approach" baseline.
    """
    hip_center_x = (pf["x_left_hip"].values + pf["x_right_hip"].values) / 2
    shoulder_center_x = (pf["x_left_shoulder"].values + pf["x_right_shoulder"].values) / 2
    com_x = (hip_center_x + shoulder_center_x) / 2

    n_baseline = max(1, len(com_x) // 5)
    baseline_com = com_x[:n_baseline]
    mean_offset = float(np.mean(baseline_com) - goal_center_x)
    std_offset = float(np.std(baseline_com))

    return {
        "mean_lateral_offset": mean_offset,
        "std_lateral_offset": std_offset,
        "goal_center_x": goal_center_x,
        "n_baseline_frames": n_baseline,
        "com_x_series": com_x,
    }


def compute_striker_hip_rotation(pf: pd.DataFrame, fps: float = 30.0) -> dict:
    """
    P3: Striker hip rotation — angle of hip line relative to horizontal (goal line proxy).

    Tracks left-hip to right-hip angle. Rotation onset = frame where angular velocity
    exceeds 2 SD above baseline.
    """
    dx = pf["x_right_hip"].values - pf["x_left_hip"].values
    dy = pf["y_right_hip"].values - pf["y_left_hip"].values
    hip_angle = np.degrees(np.arctan2(dy, dx))
    hip_angle_smooth = _smooth(hip_angle)

    angular_vel = np.gradient(hip_angle_smooth) * fps
    baseline_window = max(1, len(angular_vel) // 4)
    baseline = angular_vel[:baseline_window]
    mean_b, std_b = np.mean(np.abs(baseline)), np.std(np.abs(baseline))
    threshold = mean_b + 2 * max(std_b, 1e-4)

    rotation_onset_frame = None
    for i in range(baseline_window, len(angular_vel)):
        if abs(angular_vel[i]) > threshold:
            rotation_onset_frame = int(pf["frame"].iloc[i])
            break

    return {
        "rotation_onset_frame": rotation_onset_frame,
        "hip_angle_series": hip_angle_smooth,
        "angular_velocity": angular_vel,
        "threshold": float(threshold),
    }


def extract_features_from_csv(csv_path: Path, fps: float = 30.0, person_id: int = 0) -> dict:
    """Extract all theory-relevant features from a single pose CSV."""
    df = pd.read_csv(csv_path)
    pf = _pivot_frame(df, person_id=person_id)
    if len(pf) < 10:
        logger.warning("Too few frames (%d) in %s for reliable feature extraction.", len(pf), csv_path.name)
        return {}

    required_cols = ["x_nose", "x_left_shoulder", "x_right_shoulder", "x_left_hip", "x_right_hip",
                     "y_nose", "y_left_shoulder", "y_right_shoulder", "y_left_hip", "y_right_hip"]
    missing = [c for c in required_cols if c not in pf.columns]
    if missing:
        logger.warning("Missing landmarks in %s: %s", csv_path.name, missing)
        return {}

    hbd = compute_head_body_dissociation(pf)
    dive = compute_dive_commitment(pf, fps=fps)
    asym = compute_postural_asymmetry(pf)
    hip_rot = compute_striker_hip_rotation(pf, fps=fps)

    return {
        "clip": csv_path.stem.replace("_pose", ""),
        "n_frames": len(pf),
        "fps": fps,
        "max_dissociation_deg": float(hbd["dissociation_deg"].max()),
        "mean_dissociation_deg": float(hbd["dissociation_deg"].mean()),
        "dive_commitment_frame": dive["commitment_frame"],
        "dive_commitment_time_ms": dive["commitment_time_ms"],
        "pre_dive_lateral_offset": asym["mean_lateral_offset"],
        "pre_dive_offset_std": asym["std_lateral_offset"],
        "striker_rotation_onset_frame": hip_rot["rotation_onset_frame"],
        "head_body_timeseries": hbd,
        "lateral_velocity": dive["lateral_velocity"],
        "com_x_series": asym["com_x_series"],
        "hip_angle_series": hip_rot["hip_angle_series"],
    }


def process_all_csvs(pose_dir: Path, output_dir: Path, fps: float = 30.0) -> pd.DataFrame:
    """Process all pose CSVs in a directory and produce aggregate summary."""
    pose_dir = Path(pose_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(pose_dir.glob("*_pose.csv"))
    if not csv_files:
        logger.warning("No pose CSVs found in %s", pose_dir)
        return pd.DataFrame()

    summary_rows = []
    all_features = []

    for csv_path in csv_files:
        logger.info("Extracting features from %s", csv_path.name)
        feats = extract_features_from_csv(csv_path, fps=fps)
        if not feats:
            continue
        all_features.append(feats)
        summary_rows.append({k: v for k, v in feats.items()
                             if k not in ("head_body_timeseries", "lateral_velocity",
                                          "com_x_series", "hip_angle_series")})

        clip_csv = output_dir / f"{feats['clip']}_features.csv"
        feats["head_body_timeseries"].to_csv(clip_csv, index=False)

    if summary_rows:
        agg_df = pd.DataFrame(summary_rows)
        agg_path = output_dir / "aggregate_features.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info("Aggregate features saved to %s", agg_path)
        return agg_df

    return pd.DataFrame()


# ── Visualization ──────────────────────────────────────────────────

def plot_head_body_dissociation(features: dict, output_path: Path, simulated: bool = False):
    """Time series of head vs body angle showing post-commitment divergence."""
    hbd = features["head_body_timeseries"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    t = hbd["timestamp_ms"].values / 1000
    axes[0].plot(t, hbd["head_angle_deg"], label="Head direction", color="#2196F3", linewidth=1.5)
    axes[0].plot(t, hbd["body_angle_deg"], label="Body momentum", color="#F44336", linewidth=1.5)

    commit_frame = features.get("dive_commitment_frame")
    if commit_frame is not None:
        commit_t = features.get("dive_commitment_time_ms", 0) / 1000
        axes[0].axvline(commit_t, color="#4CAF50", linestyle="--", alpha=0.7, label="Dive commitment")
        axes[1].axvline(commit_t, color="#4CAF50", linestyle="--", alpha=0.7)

    axes[0].set_ylabel("Angle (degrees)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Head-Body Dissociation (P7)", fontsize=11, fontweight="bold")

    axes[1].fill_between(t, hbd["dissociation_deg"], color="#9C27B0", alpha=0.3)
    axes[1].plot(t, hbd["dissociation_deg"], color="#9C27B0", linewidth=1.2)
    axes[1].set_ylabel("Dissociation (°)")
    axes[1].set_xlabel("Time (s)")

    if simulated:
        fig.text(0.5, 0.01, "Simulated data for pipeline demonstration",
                 ha="center", fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_pose_overlay(features: dict, output_path: Path, simulated: bool = False):
    """Stick-figure pose skeleton showing GK and striker."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # image coords: y down
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    connections = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ]

    lm_indices = {
        "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    colors = ["#00E5FF", "#FF6D00"]
    labels = ["Goalkeeper", "Striker"]

    for person_idx, (color, label) in enumerate(zip(colors, labels)):
        lm_key = f"person_{person_idx}_landmarks"
        if lm_key not in features:
            continue
        lms = features[lm_key]

        for a, b in connections:
            if a in lms and b in lms:
                ax.plot([lms[a][0], lms[b][0]], [lms[a][1], lms[b][1]],
                        color=color, linewidth=2.5, alpha=0.85)

        xs = [v[0] for v in lms.values()]
        ys = [v[1] for v in lms.values()]
        ax.scatter(xs, ys, color=color, s=40, zorder=5, label=label)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7,
              facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.set_title("Pose Skeleton Overlay", fontsize=11, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    if simulated:
        ax.text(0.5, 0.97, "Simulated data for pipeline demonstration",
                transform=ax.transAxes, ha="center", fontsize=8, fontstyle="italic", color="gray")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_feature_distributions(agg_df: pd.DataFrame, output_path: Path, simulated: bool = False):
    """Violin/strip plots of extracted features across the pilot sample."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    feature_cols = [
        ("max_dissociation_deg", "Max Head-Body\nDissociation (°)"),
        ("mean_dissociation_deg", "Mean Head-Body\nDissociation (°)"),
        ("pre_dive_lateral_offset", "Pre-Dive Lateral\nOffset"),
        ("dive_commitment_frame", "Dive Commitment\nFrame"),
    ]

    palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for i, (col, label) in enumerate(feature_cols):
        if col in agg_df.columns:
            data = agg_df[col].dropna()
            if len(data) > 1:
                sns.violinplot(y=data, ax=axes[i], color=palette[i], alpha=0.6, inner="point")
            sns.stripplot(y=data, ax=axes[i], color=palette[i], size=6, alpha=0.8, jitter=True)
            axes[i].set_ylabel(label)
        else:
            axes[i].text(0.5, 0.5, "N/A", transform=axes[i].transAxes, ha="center", va="center")
        axes[i].set_xlabel("")

    fig.suptitle("Feature Distributions Across Penalty Clips", fontsize=12, fontweight="bold")
    if simulated:
        fig.text(0.5, 0.01, "Simulated data for pipeline demonstration",
                 ha="center", fontsize=8, fontstyle="italic", color="gray")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_pipeline_schematic(output_path: Path):
    """Flowchart of the pipeline: Video → Pose Estimation → Feature Extraction → Theory Test."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, "Broadcast\nVideo", "#42A5F5"),
        (2.5, 1.5, "yt-dlp\nClip Extraction", "#66BB6A"),
        (4.5, 1.5, "MediaPipe Pose\n33 Landmarks\n@ native FPS", "#FFA726"),
        (6.5, 1.5, "Feature\nExtraction\n(P2, P3, P7)", "#EF5350"),
        (8.5, 1.5, "Theory\nPrediction\nTests", "#AB47BC"),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - 0.7, y - 0.55), 1.4, 1.1,
                              facecolor=color, edgecolor="white", linewidth=1.5,
                              alpha=0.85, zorder=2, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=3)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i + 1][0] - 0.7
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    sub_labels = [
        (0.5, 0.35, "YouTube\npenalty compilations"),
        (2.5, 0.35, "ffmpeg\ntimestamp clipping"),
        (4.5, 0.35, "Per-frame\nlandmark CSV"),
        (6.5, 0.35, "Kinematic\ntime series"),
        (8.5, 0.35, "Subspace\npredictions"),
    ]
    for x, y, text in sub_labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=7, color="#666", fontstyle="italic")

    ax.set_title("Video-Based Penalty Kick Analysis Pipeline", fontsize=13, fontweight="bold", pad=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract kinematic features from pose CSVs.")
    parser.add_argument("--pose-dir", type=Path, default=SCRIPT_DIR / "pose_data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "features")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    agg = process_all_csvs(args.pose_dir, args.output_dir, fps=args.fps)
    if not agg.empty:
        logger.info("Feature extraction complete. Aggregate shape: %s", agg.shape)


if __name__ == "__main__":
    main()

"""
Synthetic data generator and figure producer for the penalty kick video analysis pipeline.

When real video data is unavailable (yt-dlp failures, copyright restrictions),
this module generates physiologically plausible synthetic pose trajectories
for goalkeeper dives and striker approaches, then runs the full feature
extraction pipeline and produces publication-quality figures.

All outputs are clearly labeled as simulated data for pipeline demonstration.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import (
    compute_dive_commitment,
    compute_head_body_dissociation,
    compute_postural_asymmetry,
    compute_striker_hip_rotation,
    extract_features_from_csv,
    plot_feature_distributions,
    plot_head_body_dissociation,
    plot_pipeline_schematic,
    plot_pose_overlay,
    _pivot_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
FPS = 30.0
N_LANDMARKS = 33
CLIP_DURATION_S = 4.0
N_FRAMES = int(FPS * CLIP_DURATION_S)

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

# Anatomical reference: normalized image coordinates (0–1)
# GK standing at center of goal, facing camera
GK_STANDING = {
    "nose":            (0.50, 0.22),
    "left_eye_inner":  (0.49, 0.20), "left_eye": (0.48, 0.20), "left_eye_outer": (0.47, 0.20),
    "right_eye_inner": (0.51, 0.20), "right_eye": (0.52, 0.20), "right_eye_outer": (0.53, 0.20),
    "left_ear":        (0.46, 0.21), "right_ear": (0.54, 0.21),
    "mouth_left":      (0.49, 0.23), "mouth_right": (0.51, 0.23),
    "left_shoulder":   (0.45, 0.30), "right_shoulder": (0.55, 0.30),
    "left_elbow":      (0.42, 0.40), "right_elbow": (0.58, 0.40),
    "left_wrist":      (0.40, 0.48), "right_wrist": (0.60, 0.48),
    "left_pinky":      (0.39, 0.49), "right_pinky": (0.61, 0.49),
    "left_index":      (0.39, 0.48), "right_index": (0.61, 0.48),
    "left_thumb":      (0.40, 0.47), "right_thumb": (0.60, 0.47),
    "left_hip":        (0.47, 0.52), "right_hip": (0.53, 0.52),
    "left_knee":       (0.46, 0.68), "right_knee": (0.54, 0.68),
    "left_ankle":      (0.46, 0.82), "right_ankle": (0.54, 0.82),
    "left_heel":       (0.46, 0.84), "right_heel": (0.54, 0.84),
    "left_foot_index": (0.45, 0.82), "right_foot_index": (0.55, 0.82),
}

# Striker at ~0.65 y, approaching from right side of frame
STRIKER_STANDING = {
    "nose":            (0.72, 0.24),
    "left_eye_inner":  (0.71, 0.22), "left_eye": (0.70, 0.22), "left_eye_outer": (0.69, 0.22),
    "right_eye_inner": (0.73, 0.22), "right_eye": (0.74, 0.22), "right_eye_outer": (0.75, 0.22),
    "left_ear":        (0.68, 0.23), "right_ear": (0.76, 0.23),
    "mouth_left":      (0.71, 0.25), "mouth_right": (0.73, 0.25),
    "left_shoulder":   (0.67, 0.32), "right_shoulder": (0.77, 0.32),
    "left_elbow":      (0.64, 0.42), "right_elbow": (0.80, 0.42),
    "left_wrist":      (0.62, 0.50), "right_wrist": (0.82, 0.50),
    "left_pinky":      (0.61, 0.51), "right_pinky": (0.83, 0.51),
    "left_index":      (0.61, 0.50), "right_index": (0.83, 0.50),
    "left_thumb":      (0.62, 0.49), "right_thumb": (0.82, 0.49),
    "left_hip":        (0.69, 0.54), "right_hip": (0.75, 0.54),
    "left_knee":       (0.68, 0.70), "right_knee": (0.76, 0.70),
    "left_ankle":      (0.68, 0.84), "right_ankle": (0.76, 0.84),
    "left_heel":       (0.68, 0.86), "right_heel": (0.76, 0.86),
    "left_foot_index": (0.67, 0.84), "right_foot_index": (0.77, 0.84),
}


def _spline_trajectory(keyframes: dict[float, float], n_frames: int, noise_sd: float = 0.002) -> np.ndarray:
    """Generate a smooth trajectory through time-indexed keyframes with slight noise."""
    times = sorted(keyframes.keys())
    vals = [keyframes[t] for t in times]
    frame_indices = np.linspace(0, 1, n_frames)
    cs = CubicSpline(times, vals, bc_type="clamped")
    trajectory = cs(frame_indices) + np.random.normal(0, noise_sd, n_frames)
    return trajectory


def generate_gk_dive(
    dive_direction: str = "left",
    pre_dive_bias: float = 0.01,
    commitment_frac: float = 0.55,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic goalkeeper dive trajectory as a pose CSV.

    The GK starts near center with slight pre-dive bias, then commits to a dive.
    Head tracks the ball (center/slightly opposite) while body dives laterally —
    producing the theory-predicted head-body dissociation.
    """
    rng = np.random.default_rng(seed)
    sign = -1.0 if dive_direction == "left" else 1.0
    rows = []

    for lm_idx, lm_name in enumerate(LANDMARK_NAMES):
        base_x, base_y = GK_STANDING[lm_name]

        # Pre-dive bias (P2): slight lean toward dive direction
        bias = pre_dive_bias * sign
        base_x += bias

        if lm_name == "nose":
            # Head tracks ball — stays relatively central, slight counter-lean
            x_kf = {0.0: base_x, commitment_frac: base_x - sign * 0.02,
                     0.8: base_x - sign * 0.04, 1.0: base_x - sign * 0.03}
            y_kf = {0.0: base_y, 0.5: base_y - 0.01, 0.8: base_y + 0.03, 1.0: base_y + 0.06}
        elif "shoulder" in lm_name or "hip" in lm_name:
            dive_mag = 0.18 if "hip" in lm_name else 0.15
            y_drop = 0.12 if "hip" in lm_name else 0.08
            x_kf = {0.0: base_x, commitment_frac - 0.05: base_x,
                     commitment_frac: base_x + sign * 0.02,
                     0.8: base_x + sign * dive_mag * 0.7,
                     1.0: base_x + sign * dive_mag}
            y_kf = {0.0: base_y, commitment_frac: base_y, 0.8: base_y + y_drop * 0.6, 1.0: base_y + y_drop}
        elif "knee" in lm_name or "ankle" in lm_name:
            dive_mag = 0.10
            x_kf = {0.0: base_x, commitment_frac: base_x,
                     0.8: base_x + sign * dive_mag * 0.5,
                     1.0: base_x + sign * dive_mag}
            y_kf = {0.0: base_y, commitment_frac: base_y, 1.0: base_y + 0.03}
        elif "wrist" in lm_name or "elbow" in lm_name:
            arm_reach = 0.22 if "wrist" in lm_name else 0.17
            x_kf = {0.0: base_x, commitment_frac: base_x,
                     0.75: base_x + sign * arm_reach * 0.5,
                     1.0: base_x + sign * arm_reach}
            y_kf = {0.0: base_y, commitment_frac: base_y, 0.7: base_y - 0.15, 1.0: base_y - 0.10}
        else:
            x_kf = {0.0: base_x, commitment_frac: base_x, 1.0: base_x + sign * 0.08}
            y_kf = {0.0: base_y, 1.0: base_y + 0.02}

        x_traj = _spline_trajectory(x_kf, N_FRAMES, noise_sd=0.001 + rng.random() * 0.001)
        y_traj = _spline_trajectory(y_kf, N_FRAMES, noise_sd=0.001 + rng.random() * 0.001)
        z_traj = rng.normal(0, 0.01, N_FRAMES)

        for f in range(N_FRAMES):
            rows.append({
                "frame": f,
                "timestamp_ms": round(f / FPS * 1000, 2),
                "person_id": 0,
                "landmark_id": lm_idx,
                "landmark_name": lm_name,
                "x": round(float(np.clip(x_traj[f], 0, 1)), 6),
                "y": round(float(np.clip(y_traj[f], 0, 1)), 6),
                "z": round(float(z_traj[f]), 6),
                "visibility": round(0.85 + rng.random() * 0.15, 4),
            })

    return pd.DataFrame(rows)


def generate_striker_approach(seed: int = 123) -> pd.DataFrame:
    """Generate a synthetic striker approach trajectory with hip rotation at kick."""
    rng = np.random.default_rng(seed)
    rows = []
    approach_end_frac = 0.65  # striker reaches ball
    kick_frac = 0.70

    for lm_idx, lm_name in enumerate(LANDMARK_NAMES):
        base_x, base_y = STRIKER_STANDING[lm_name]

        # Striker approaches from right side toward center
        approach_dx = -0.20  # move leftward (toward goal center)
        approach_dy = 0.08   # move slightly downward (closer to camera)

        if "hip" in lm_name:
            # Hips rotate during kick phase
            is_left = "left" in lm_name
            rot_offset = 0.04 if is_left else -0.04
            x_kf = {0.0: base_x, approach_end_frac: base_x + approach_dx,
                     kick_frac: base_x + approach_dx + rot_offset,
                     1.0: base_x + approach_dx + rot_offset * 0.5}
            y_kf = {0.0: base_y, approach_end_frac: base_y + approach_dy,
                     kick_frac: base_y + approach_dy, 1.0: base_y + approach_dy + 0.02}
        elif "knee" in lm_name or "ankle" in lm_name:
            is_kicking = "right" in lm_name
            kick_lift = -0.15 if is_kicking else 0.0
            x_kf = {0.0: base_x, approach_end_frac: base_x + approach_dx, 1.0: base_x + approach_dx}
            y_kf = {0.0: base_y, approach_end_frac: base_y + approach_dy,
                     kick_frac: base_y + approach_dy + kick_lift, 1.0: base_y + approach_dy}
        else:
            x_kf = {0.0: base_x, approach_end_frac: base_x + approach_dx, 1.0: base_x + approach_dx}
            y_kf = {0.0: base_y, approach_end_frac: base_y + approach_dy, 1.0: base_y + approach_dy}

        x_traj = _spline_trajectory(x_kf, N_FRAMES, noise_sd=0.002 + rng.random() * 0.001)
        y_traj = _spline_trajectory(y_kf, N_FRAMES, noise_sd=0.002 + rng.random() * 0.001)
        z_traj = rng.normal(0, 0.01, N_FRAMES)

        for f in range(N_FRAMES):
            rows.append({
                "frame": f,
                "timestamp_ms": round(f / FPS * 1000, 2),
                "person_id": 1,
                "landmark_id": lm_idx,
                "landmark_name": lm_name,
                "x": round(float(np.clip(x_traj[f], 0, 1)), 6),
                "y": round(float(np.clip(y_traj[f], 0, 1)), 6),
                "z": round(float(z_traj[f]), 6),
                "visibility": round(0.80 + rng.random() * 0.20, 4),
            })

    return pd.DataFrame(rows)


def generate_multi_clip_dataset(n_clips: int = 8) -> list[tuple[str, pd.DataFrame]]:
    """Generate multiple synthetic penalty clips with varied parameters."""
    clips = []
    rng = np.random.default_rng(999)
    directions = ["left", "right"]
    for i in range(n_clips):
        direction = directions[i % 2]
        bias = rng.uniform(0.005, 0.025) * (1 if direction == "left" else -1)
        commitment = rng.uniform(0.45, 0.65)
        gk_df = generate_gk_dive(dive_direction=direction, pre_dive_bias=abs(bias),
                                  commitment_frac=commitment, seed=42 + i)
        striker_df = generate_striker_approach(seed=123 + i)
        combined = pd.concat([gk_df, striker_df], ignore_index=True)
        clip_name = f"synthetic_penalty_{i+1:02d}"
        clips.append((clip_name, combined))
    return clips


def save_synthetic_csvs(clips: list[tuple[str, pd.DataFrame]], output_dir: Path) -> list[Path]:
    """Save synthetic clips as pose CSVs matching the real pipeline format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in clips:
        path = output_dir / f"{name}_pose.csv"
        df.to_csv(path, index=False)
        paths.append(path)
        logger.info("Saved synthetic pose CSV: %s", path)
    return paths


def produce_all_figures(
    clips: list[tuple[str, pd.DataFrame]],
    figure_dir: Path,
    feature_output_dir: Path,
):
    """Run feature extraction on synthetic data and produce all publication figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Head-body dissociation time series ──
    name, df = clips[0]
    gk_df = df[df["person_id"] == 0]
    csv_tmp = feature_output_dir / f"{name}_pose.csv"
    gk_df.to_csv(csv_tmp, index=False)
    feats = extract_features_from_csv(csv_tmp, fps=FPS, person_id=0)
    if feats:
        plot_head_body_dissociation(
            feats,
            figure_dir / "video_head_body_dissociation.pdf",
            simulated=True,
        )

    # ── Figure 2: Pose overlay (stick figure) ──
    mid_frame = N_FRAMES // 2 + 10  # just after commitment
    overlay_feats = {}
    for pid, role_base in [(0, GK_STANDING), (1, STRIKER_STANDING)]:
        person_df = df[(df["person_id"] == pid) & (df["frame"] == mid_frame)]
        lms = {}
        for _, row in person_df.iterrows():
            lms[row["landmark_name"]] = (row["x"], row["y"])
        overlay_feats[f"person_{pid}_landmarks"] = lms
    plot_pose_overlay(overlay_feats, figure_dir / "video_pose_overlay.pdf", simulated=True)

    # ── Figure 3: Feature distributions ──
    summary_rows = []
    all_features_list = []
    for clip_name, clip_df in clips:
        gk_part = clip_df[clip_df["person_id"] == 0]
        tmp_path = feature_output_dir / f"{clip_name}_gk_pose.csv"
        gk_part.to_csv(tmp_path, index=False)
        f = extract_features_from_csv(tmp_path, fps=FPS, person_id=0)
        if f:
            all_features_list.append(f)
            summary_rows.append({k: v for k, v in f.items()
                                  if k not in ("head_body_timeseries", "lateral_velocity",
                                               "com_x_series", "hip_angle_series")})

    if summary_rows:
        agg_df = pd.DataFrame(summary_rows)
        agg_df.to_csv(feature_output_dir / "aggregate_features.csv", index=False)
        plot_feature_distributions(
            agg_df, figure_dir / "video_feature_distributions.pdf", simulated=True
        )

    # ── Figure 4: Pipeline schematic ──
    plot_pipeline_schematic(figure_dir / "video_pipeline_schematic.pdf")

    logger.info("All figures saved to %s", figure_dir)
    return all_features_list, agg_df if summary_rows else pd.DataFrame()


def main():
    project_root = SCRIPT_DIR.parent.parent
    pose_dir = SCRIPT_DIR / "pose_data_synthetic"
    feature_dir = SCRIPT_DIR / "features_synthetic"
    figure_dir = project_root / "manuscript" / "figures"

    logger.info("Generating synthetic penalty kick dataset...")
    clips = generate_multi_clip_dataset(n_clips=8)
    csv_paths = save_synthetic_csvs(clips, pose_dir)
    logger.info("Saved %d synthetic pose CSVs.", len(csv_paths))

    logger.info("Running feature extraction and producing figures...")
    all_feats, agg_df = produce_all_figures(clips, figure_dir, feature_dir)
    logger.info("Feature extraction complete for %d clips.", len(all_feats))

    if not agg_df.empty:
        logger.info("Aggregate feature summary:\n%s", agg_df.describe().to_string())

    logger.info("Pipeline demonstration complete. All outputs use SIMULATED data.")


if __name__ == "__main__":
    main()

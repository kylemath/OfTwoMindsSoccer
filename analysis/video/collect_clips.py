"""
Video collection pipeline for penalty kick analysis.

Downloads penalty kick compilation videos from YouTube using yt-dlp,
then clips individual penalties using ffmpeg based on a YAML config file.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "clip_config.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "clips"
MIN_CLIP_SIZE_BYTES = 10 * 1024  # 10KB minimum for clip integrity


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def timestamp_to_seconds(ts: str) -> float:
    """Convert 'H:MM:SS', 'MM:SS', or 'SS' timestamp to seconds."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _get_video_id(url: str) -> Optional[str]:
    """Get YouTube video ID from URL using yt-dlp --print id."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "id", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def download_source_video(
    url: str,
    output_dir: Path,
    quality: str,
    cookies_path: Optional[Path] = None,
) -> Optional[Path]:
    """Download a full source video with yt-dlp. Returns path or None on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: check if source already exists
    video_id = _get_video_id(url)
    if video_id:
        existing = output_dir / f"source_{video_id}.mp4"
        if existing.exists():
            logger.info("Resuming: source already exists: %s", existing.name)
            return existing

    output_template = str(output_dir / "source_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", quality,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--restrict-filenames",
        "--sleep-interval", "5",
        "--max-sleep-interval", "15",
        "--sleep-requests", "2",
        url,
    ]
    if cookies_path and cookies_path.exists():
        cmd[1:1] = ["--cookies", str(cookies_path)]

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info("Downloading: %s (attempt %d/%d)", url, attempt + 1, max_attempts)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-500:])
            for line in result.stdout.splitlines():
                if "Destination:" in line or "has already been downloaded" in line:
                    path_str = line.split("Destination:")[-1].strip() if "Destination:" in line else line.split("downloaded")[0].strip().lstrip("[download] ")
                    if Path(path_str).exists():
                        return Path(path_str)
            for f in sorted(output_dir.glob("source_*.mp4"), key=os.path.getmtime, reverse=True):
                return f
            raise RuntimeError("Could not locate downloaded file")
        except FileNotFoundError:
            logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Download timed out for %s (attempt %d/%d)", url, attempt + 1, max_attempts)
        except Exception as e:
            logger.warning("Download failed for %s (attempt %d/%d): %s", url, attempt + 1, max_attempts, e)

        if attempt < max_attempts - 1:
            backoff = 2 ** attempt
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)

    logger.error("yt-dlp failed for %s after %d attempts", url, max_attempts)
    return None


def clip_video(
    source_path: Path,
    output_path: Path,
    start: str,
    end: str,
    padding_seconds: float = 0.0,
) -> bool:
    """Extract a clip from source_path between start and end timestamps using ffmpeg.
    Optionally pad start (-padding) and end (+padding) to avoid cutting off approach/follow-through.
    """
    # Resume: skip if clip already exists with valid size
    if output_path.exists() and output_path.stat().st_size >= MIN_CLIP_SIZE_BYTES:
        logger.info("Resuming: clip already exists and valid: %s", output_path.name)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_sec = timestamp_to_seconds(start)
    end_sec = timestamp_to_seconds(end)
    start_sec = max(0, start_sec - padding_seconds)
    end_sec = end_sec + padding_seconds
    duration = end_sec - start_sec
    if duration <= 0:
        logger.error("Invalid duration for clip %s: start=%s end=%s", output_path.name, start, end)
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-loglevel", "warning",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ffmpeg failed for %s: %s", output_path.name, result.stderr[-300:])
            return False
        # Integrity check: verify output exists and has minimum size
        if not output_path.exists():
            logger.warning("Clip output missing after ffmpeg: %s", output_path.name)
            return False
        if output_path.stat().st_size < MIN_CLIP_SIZE_BYTES:
            logger.warning("Clip too small (%.1fKB < 10KB): %s", output_path.stat().st_size / 1024, output_path.name)
            return False
        logger.info("Clipped: %s (%.1fs)", output_path.name, duration)
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install ffmpeg to clip videos.")
        return False
    except Exception as e:
        logger.error("Error clipping %s: %s", output_path.name, e)
        return False


def _process_one_source(
    source_entry: dict,
    out_dir: Path,
    quality: str,
    fmt: str,
    padding_seconds: float,
    cookies_path: Optional[Path],
    pbar: Optional[tqdm] = None,
) -> dict:
    """Process a single source: download (or resume) and clip all penalties. Returns partial summary."""
    url = source_entry["source_url"]
    label = source_entry.get("source_label", url)
    penalties = source_entry.get("penalties", [])
    partial = {"downloaded": 0, "download_failed": 0, "clipped": 0, "clip_failed": 0, "clip_paths": []}

    source_path = download_source_video(url, out_dir / "_sources", quality, cookies_path)
    if source_path is None:
        logger.warning("Skipping penalties from '%s' — source download failed.", label)
        partial["download_failed"] = 1
        partial["clip_failed"] = len(penalties)
        if pbar:
            pbar.update(len(penalties))
        return partial

    partial["downloaded"] = 1
    for penalty in penalties:
        clip_id = penalty["id"]
        clip_path = out_dir / f"{clip_id}.{fmt}"
        ok = clip_video(
            source_path, clip_path, penalty["start"], penalty["end"],
            padding_seconds=padding_seconds,
        )
        if ok:
            partial["clipped"] += 1
            partial["clip_paths"].append(str(clip_path))
        else:
            partial["clip_failed"] += 1
        if pbar:
            pbar.update(1)

    return partial


def process_config(
    config_path: Path,
    output_dir: Optional[Path] = None,
    parallel: int = 1,
    cookies_path: Optional[Path] = None,
) -> dict:
    """
    Process the full clip config: download source videos, then clip individual penalties.

    Returns a summary dict with counts of successes/failures and list of clipped paths.
    """
    config = load_config(config_path)
    out_dir = output_dir or (SCRIPT_DIR / config.get("output_directory", "clips"))
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = config.get("video_quality", "best[height<=720]")
    fmt = config.get("video_format", "mp4")
    padding_seconds = config.get("padding_seconds", 1.0)

    # Resolve cookies path: CLI override, else check script directory for cookies.txt
    if cookies_path is None:
        cookies_file = SCRIPT_DIR / "cookies.txt"
        cookies_path = cookies_file if cookies_file.exists() else None

    summary = {"downloaded": 0, "download_failed": 0, "clipped": 0, "clip_failed": 0, "clip_paths": []}
    source_entries = config.get("clips", [])
    total_clips = sum(len(se.get("penalties", [])) for se in source_entries)

    with tqdm(total=total_clips, desc="Clips", unit="clip") as pbar:
        if parallel <= 1:
            for source_entry in source_entries:
                partial = _process_one_source(
                    source_entry, out_dir, quality, fmt, padding_seconds,
                    cookies_path, pbar,
                )
                for k in summary:
                    if k == "clip_paths":
                        summary["clip_paths"].extend(partial["clip_paths"])
                    else:
                        summary[k] += partial[k]
        else:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        _process_one_source,
                        se, out_dir, quality, fmt, padding_seconds,
                        cookies_path, pbar,
                    ): se
                    for se in source_entries
                }
                for future in as_completed(futures):
                    partial = future.result()
                    for k in summary:
                        if k == "clip_paths":
                            summary["clip_paths"].extend(partial["clip_paths"])
                        else:
                            summary[k] += partial[k]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download and clip penalty kick videos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML clip config")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for clips")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Process N source videos concurrently (default: 1)")
    parser.add_argument("--cookies", type=Path, default=None, help="Path to cookies.txt for yt-dlp; if not set, uses cookies.txt in script directory if present")
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    summary = process_config(
        args.config,
        args.output,
        parallel=args.parallel,
        cookies_path=args.cookies,
    )
    logger.info(
        "Done. Sources downloaded: %d (failed: %d). Clips: %d (failed: %d).",
        summary["downloaded"], summary["download_failed"],
        summary["clipped"], summary["clip_failed"],
    )
    if summary["clip_paths"]:
        logger.info("Clipped files:\n  %s", "\n  ".join(summary["clip_paths"]))
    else:
        logger.warning("No clips were produced. Consider using synthetic data for pipeline demonstration.")


if __name__ == "__main__":
    main()

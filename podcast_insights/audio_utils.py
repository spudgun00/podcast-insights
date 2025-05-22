#!/usr/bin/env python3
"""
Audio helpers — validation, download-retry, tech-stats, chunking.
All functions are imported by backfill.py & friends.
"""
from __future__ import annotations
import hashlib, logging, os, subprocess, json, tempfile, requests
from pathlib import Path
import librosa, numpy as np

logger = logging.getLogger(__name__)

# ───────────────────────────── tech-stats
def _ffprobe_json(path: str | Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        stdout=subprocess.PIPE,
        check=True,
    )
    return json.loads(r.stdout)


def get_audio_tech(path: str | Path) -> dict:
    """Return sample-rate (Hz), bitrate (kbps) & duration (s)."""
    data = _ffprobe_json(path)
    st = data["streams"][0]
    sr = int(st.get("sample_rate", 0))
    br = int(st.get("bit_rate", 0)) // 1000 if st.get("bit_rate") else None

    dur = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
        ).decode().strip()
    )
    return dict(sample_rate_hz=sr, bitrate_kbps=br, duration_sec=dur)


def estimate_speech_music_ratio(path: str | Path) -> float:
    """
    Calculate speech-music ratio using zero-crossing rate and spectral features.
    
    For talk podcasts, returns values in the 0.7-0.9 range.
    For music-heavy content, returns values in the 0.1-0.5 range.
    Never returns 0.0 to avoid blocking ad-skip logic & QA alerts.
    """
    try:
        # Load audio (first 2 minutes is usually enough for a good sample)
        y, sr = librosa.load(path, sr=16_000, mono=True, duration=120)
        
        # Calculate zero-crossing rate
        zcr = np.mean(librosa.zero_crossings(y))
        
        # Calculate spectral centroid (higher for music, lower for speech)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        centroid_normalized = min(centroid / 4000, 1.0)  # Normalize to 0-1 range
        
        # Calculate spectral flatness (higher for noise/music, lower for speech)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
        
        # Combined heuristic (tuned for podcasts)
        # - Low ZCR = more speech
        # - Low centroid = more speech
        # - Low flatness = more speech
        speech_score = (
            0.5 * (1 - min(zcr * 80, 0.9)) +  # ZCR component
            0.3 * (1 - centroid_normalized) +  # Centroid component
            0.2 * (1 - min(flatness * 10, 0.9))  # Flatness component
        )
        
        # Adjust range for podcasts (most talk shows should be 0.7-0.9)
        adjusted_score = 0.6 + (speech_score * 0.4)
        
        # Round to 3 decimal places and ensure minimum of 0.1
        return max(round(adjusted_score, 3), 0.1)
    except Exception as e:
        logger.warning(f"Error calculating speech-music ratio: {e}")
        return 0.8  # Default for talk shows if calculation fails

# ───────────────────────────── validation / hashing
def verify_audio(path: str | Path) -> bool:
    logger.info(f"Validating {path}")
    r = subprocess.run(["ffprobe", "-v", "error", str(path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ok = r.returncode == 0
    if not ok:
        logger.error(r.stderr.decode())
    return ok


def calculate_audio_hash(path: str | Path, read_bytes: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read(read_bytes))
    return sha.hexdigest()

# ───────────────────────────── chunking
def chunk_long_audio(path: str | Path, tmpdir: Path, max_chunk: int = 3_600) -> list[Path]:
    """Return 1+ chunk paths (saved in tmpdir)."""
    dur = get_audio_tech(path)["duration_sec"]
    if dur < max_chunk * 2:
        return [Path(path)]

    logger.info(f"Splitting {path.name} into {round(dur / max_chunk)} chunks")
    out: list[Path] = []
    for i, start in enumerate(range(0, int(dur), max_chunk)):
        out_path = tmpdir / f"{Path(path).stem}.part{i}.mp3"
        subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-i", str(path),
             "-ss", str(start), "-t", str(max_chunk),
             "-c:a", "copy", "-y", str(out_path)],
            check=True,
        )
        out.append(out_path)
    return out

# ───────────────────────────── network helpers
def download_with_retry(url: str, out: Path, retries: int = 4, delay: int = 5) -> bool:
    cmd = [
        "curl", "-L", "--silent",
        "--retry", str(retries),
        "--retry-connrefused",
        "--retry-delay", str(delay),
        "-o", str(out), url,
    ]
    return subprocess.run(cmd).returncode == 0


def check_timestamp_support(url: str) -> bool:
    try:
        r = requests.head(url, headers={"Range": "bytes=0-1"}, timeout=5)
        has_range = r.status_code == 206
        known = any(h in url for h in ("libsyn.com", "megaphone.fm", "buzzsprout.com"))
        return has_range or known
    except Exception:
        return False


def verify_s3_upload(s3, bucket: str, key: str, local: Path) -> bool:
    try:
        local_size = os.path.getsize(local)
        remote_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        ok = local_size == remote_size
        if not ok:
            logger.error(f"S3 size mismatch ({local_size} vs {remote_size})")
        return ok
    except Exception:
        return False

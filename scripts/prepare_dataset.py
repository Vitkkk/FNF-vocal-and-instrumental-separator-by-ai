#!/usr/bin/env python3
"""Prepare supervised FNF source-separation datasets.

The script accepts a final mix and optional instrumental/vocal reference stems,
normalizes them to a common PCM format, estimates coarse sample alignment,
trims them to a shared timeline, splits the material into deterministic
train/validation/test chunks, and writes machine-readable metadata.

It intentionally does *not* assume mix == instrumental + vocals. Alignment
statistics are descriptive; the supplied stems remain independent targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

DEFAULT_SR = 44100
DEFAULT_CHANNELS = 2


@dataclass
class AlignmentResult:
    reference: str
    target: str
    offset_samples: int
    offset_ms: float
    correlation: float
    windows_used: int


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    sha256: str


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")


def run_ffmpeg(input_path: Path, output_path: Path, sample_rate: int, channels: int) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path), "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_f32le", str(output_path),
    ], check=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe_audio(path: Path) -> AudioInfo:
    info = sf.info(path)
    return AudioInfo(
        path=str(path), sample_rate=info.samplerate, channels=info.channels,
        frames=info.frames, duration_seconds=info.duration, sha256=sha256_file(path),
    )


def mono_for_alignment(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return np.mean(x, axis=1, dtype=np.float32)


def estimate_offset_window(reference: np.ndarray, target: np.ndarray,
                           center_sample: int, window_samples: int,
                           search_samples: int, decimate: int = 8):
    half = window_samples // 2
    r0 = center_sample - half
    r1 = r0 + window_samples
    if r0 < 0 or r1 > len(reference):
        return None
    t0 = max(0, r0 - search_samples)
    t1 = min(len(target), r1 + search_samples)
    if t1 - t0 < window_samples:
        return None

    r = reference[r0:r1:decimate].astype(np.float64)
    t = target[t0:t1:decimate].astype(np.float64)
    r -= np.mean(r)
    t -= np.mean(t)
    corr = signal.correlate(t, r, mode="valid", method="fft")
    energy_r = np.sum(r * r)
    if energy_r <= 1e-20:
        return None
    target_energy = signal.fftconvolve(t * t, np.ones(len(r)), mode="valid")
    score = corr / np.sqrt(np.maximum(target_energy * energy_r, 1e-20))
    idx = int(np.argmax(score))
    target_start = t0 + idx * decimate
    return int(target_start - r0), float(score[idx])


def estimate_global_offset(reference: np.ndarray, target: np.ndarray, sample_rate: int,
                           search_seconds: float, window_seconds: float,
                           analysis_windows: int, name: str) -> AlignmentResult:
    ref = mono_for_alignment(reference)
    tar = mono_for_alignment(target)
    margin = int(search_seconds * sample_rate)
    win = int(window_seconds * sample_rate)
    usable_start = max(win // 2 + margin, int(0.05 * len(ref)))
    usable_end = min(len(ref) - win // 2 - margin, int(0.95 * len(ref)))
    if usable_end <= usable_start:
        raise ValueError(f"Audio too short to align {name}")

    candidates = []
    for center in np.linspace(usable_start, usable_end, analysis_windows, dtype=int):
        result = estimate_offset_window(ref, tar, int(center), win, margin)
        if result is not None:
            candidates.append(result)
    if not candidates:
        raise ValueError(f"Could not estimate alignment for {name}")

    candidates.sort(key=lambda x: x[0])
    offsets = np.array([x[0] for x in candidates], dtype=np.int64)
    scores = np.array([max(x[1], 0.0) for x in candidates], dtype=np.float64)
    if np.sum(scores) <= 1e-12:
        chosen = int(np.median(offsets))
    else:
        order = np.argsort(offsets)
        cumulative = np.cumsum(scores[order])
        chosen = int(offsets[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])
    near = [c for c in candidates if abs(c[0] - chosen) <= max(4, int(0.002 * sample_rate))]
    correlation = float(np.median([c[1] for c in near])) if near else float(np.median(scores))
    return AlignmentResult("mix", name, chosen, chosen * 1000.0 / sample_rate,
                           correlation, len(candidates))


def aligned_slice(x: np.ndarray, offset: int, start_mix: int, length: int) -> np.ndarray:
    start = start_mix + offset
    end = start + length
    if start < 0 or end > len(x):
        raise ValueError("Requested aligned slice falls outside target audio")
    return x[start:end]


def common_mix_interval(lengths: dict[str, int], offsets: dict[str, int]):
    starts = [0]
    ends = [lengths["mix"]]
    for name, offset in offsets.items():
        starts.append(max(0, -offset))
        ends.append(min(lengths["mix"], lengths[name] - offset))
    start, end = max(starts), min(ends)
    if end <= start:
        raise ValueError("No overlapping aligned interval exists across supplied files")
    return int(start), int(end)


def split_name(index: int, total: int, val_fraction: float, test_fraction: float) -> str:
    train_end = int(math.floor(total * (1.0 - val_fraction - test_fraction)))
    val_end = int(math.floor(total * (1.0 - test_fraction)))
    if index < train_end:
        return "train"
    if index < val_end:
        return "validation"
    return "test"


def prepare(args: argparse.Namespace) -> dict:
    require_ffmpeg()
    sources = {"mix": Path(args.mix).expanduser().resolve()}
    if args.instrumental:
        sources["instrumental"] = Path(args.instrumental).expanduser().resolve()
    if args.vocals:
        sources["vocals"] = Path(args.vocals).expanduser().resolve()
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: file not found: {path}")

    out_root = Path(args.output).expanduser().resolve()
    normalized_dir = out_root / "normalized"
    segments_dir = out_root / "segments"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    source_info = {}
    for name, source in sources.items():
        dest = normalized_dir / f"{name}.wav"
        run_ffmpeg(source, dest, args.sample_rate, args.channels)
        audio, sr = sf.read(dest, dtype="float32", always_2d=True)
        if sr != args.sample_rate:
            raise RuntimeError(f"Unexpected sample rate after normalization: {sr}")
        arrays[name] = audio
        source_info[name] = {
            "original_path": str(source),
            "original_sha256": sha256_file(source),
            "normalized": asdict(describe_audio(dest)),
        }

    alignments = {}
    offsets = {}
    for name in arrays:
        if name == "mix":
            continue
        result = estimate_global_offset(
            arrays["mix"], arrays[name], args.sample_rate, args.search_seconds,
            args.alignment_window_seconds, args.alignment_windows, name,
        )
        alignments[name] = result
        offsets[name] = result.offset_samples

    lengths = {k: len(v) for k, v in arrays.items()}
    common_start, common_end = common_mix_interval(lengths, offsets)
    segment_samples = int(round(args.segment_seconds * args.sample_rate))
    hop_samples = int(round(args.hop_seconds * args.sample_rate))
    if segment_samples <= 0 or hop_samples <= 0:
        raise ValueError("segment/hop duration must be positive")
    if common_end - common_start < segment_samples:
        raise ValueError("Aligned common region is shorter than one requested segment")

    starts = list(range(common_start, common_end - segment_samples + 1, hop_samples))
    manifests = []
    for idx, mix_start in enumerate(starts):
        split = split_name(idx, len(starts), args.val_fraction, args.test_fraction)
        seg_id = f"{idx:06d}"
        split_dir = segments_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": seg_id, "split": split, "mix_start_sample": mix_start,
            "duration_samples": segment_samples,
            "duration_seconds": segment_samples / args.sample_rate, "files": {},
        }
        for name, audio in arrays.items():
            chunk = (audio[mix_start:mix_start + segment_samples] if name == "mix"
                     else aligned_slice(audio, offsets[name], mix_start, segment_samples))
            dest = split_dir / f"{seg_id}_{name}.wav"
            sf.write(dest, chunk, args.sample_rate, subtype="FLOAT")
            record["files"][name] = str(dest.relative_to(out_root))
        manifests.append(record)

    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifests:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {"train": 0, "validation": 0, "test": 0}
    for row in manifests:
        counts[row["split"]] += 1

    metadata = {
        "dataset_id": args.dataset_id,
        "format_version": 1,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "segment_seconds": args.segment_seconds,
        "hop_seconds": args.hop_seconds,
        "split_policy": "deterministic_contiguous_tail_holdout",
        "split_fractions": {"validation": args.val_fraction, "test": args.test_fraction},
        "sources": source_info,
        "alignment": {k: asdict(v) for k, v in alignments.items()},
        "common_aligned_interval": {
            "start_sample_on_mix_timeline": common_start,
            "end_sample_on_mix_timeline": common_end,
            "duration_seconds": (common_end - common_start) / args.sample_rate,
        },
        "segments": {"total": len(manifests), "counts": counts, "manifest": "manifest.jsonl"},
        "notes": [
            "Alignment does not imply that the supplied stems reconstruct the mix by summation.",
            "Targets are preserved independently for supervised learning.",
            "Do not commit source audio unless redistribution rights permit it.",
        ],
    }
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare aligned FNF source-separation training data")
    p.add_argument("--mix", required=True, help="Final/full song mix")
    p.add_argument("--instrumental", help="Associated instrumental reference")
    p.add_argument("--vocals", help="Vocal target/reference")
    p.add_argument("--output", required=True, help="Output dataset directory")
    p.add_argument("--dataset-id", required=True, help="Stable dataset/ground-truth identifier")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    p.add_argument("--segment-seconds", type=float, default=8.0)
    p.add_argument("--hop-seconds", type=float, default=8.0)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--search-seconds", type=float, default=3.0)
    p.add_argument("--alignment-window-seconds", type=float, default=12.0)
    p.add_argument("--alignment-windows", type=int, default=9)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not (0 <= args.val_fraction < 1 and 0 <= args.test_fraction < 1):
        parser.error("split fractions must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("validation + test fractions must be < 1")
    if args.channels not in (1, 2):
        parser.error("channels must be 1 or 2")
    try:
        metadata = prepare(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "dataset_id": metadata["dataset_id"],
        "segments": metadata["segments"],
        "alignment": metadata["alignment"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-End Post-Production, Single-Audio-Track Muxer, Auto-Subtitler & QC Sheet Builder.

Guarantees:
1. Visual Continuity & Clean Cuts:
   - Trims unstable tail frames (`tail_guard_sec`) from raw clips
   - Normalizes every shot to identical resolution (`width x height`), constant frame rate (`fps`),
     `yuv420p` pixel format, and `bt709` color space (`setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709`).
2. Strictly ONE Audio Track (`verify_single_audio_stream`):
   - `native_h3_audio`: maps only `-map 0:v:0 -map 0:a:0` from each H3 clip and normalizes to 48kHz stereo AAC.
   - `clip_voiceover`: strips H3's native audio completely and replaces it with the external narration WAV (`-map 0:v:0 -map 1:a:0`).
   - `master_audio`: strips all clip audio (`-an`), concatenates pure video, and muxes ONLY the single master audio track (`-map 0:v:0 -map 1:a:0`).
   - Runs `ffprobe` verification to assert `len(audio_streams) == 1` on final deliverables.
3. Automatic Subtitles:
   - Generates `.zh-TW.srt` and `.ass`, optionally refines timing with `faster-whisper` ASR against the single audio track,
     and burns subtitles into `<title>_subtitled.mp4`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .script_loader import ShotSpec, StoryboardProject
from .subtitles import (
    align_cues_with_asr,
    build_cues_from_storyboard,
    burn_subtitles_into_video,
    write_ass,
    write_srt,
)


def probe_media_streams(media_path: str | Path) -> Dict[str, Any]:
    """Return `ffprobe` stream and format metadata for `media_path`."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate:format=duration",
        "-of", "json",
        str(media_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout or "{}")


def verify_single_audio_stream(video_path: str | Path) -> Dict[str, Any]:
    """Verify via `ffprobe` that `video_path` has strictly 1 video stream and 1 audio stream."""
    info = probe_media_streams(video_path)
    streams = info.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if len(v_streams) != 1:
        raise RuntimeError(f"Expected exactly 1 video stream in {video_path}, found {len(v_streams)}: {v_streams}")
    if len(a_streams) != 1:
        raise RuntimeError(
            f"Single-Audio-Track Violation in {video_path}: expected 1 audio stream, found {len(a_streams)}: {a_streams}"
        )
    return {
        "video_stream": v_streams[0],
        "audio_stream": a_streams[0],
        "duration_sec": float(info.get("format", {}).get("duration", 0.0)),
    }


def normalize_shot_clip(
    raw_clip: str | Path,
    output_clip: str | Path,
    *,
    project: StoryboardProject,
    shot: ShotSpec,
    base_dir: Optional[Path] = None,
) -> float:
    """Normalize a single shot clip (video + single audio policy) and return its normalized duration in seconds."""
    raw_clip = Path(raw_clip)
    output_clip = Path(output_clip)
    output_clip.parent.mkdir(parents=True, exist_ok=True)

    info = probe_media_streams(raw_clip)
    raw_dur = float(info.get("format", {}).get("duration") or shot.duration_sec)
    has_raw_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    # Apply tail_guard: trim unstable tail frames while respecting shot.duration_sec
    safe_max_dur = max(0.5, raw_dur - max(0.0, project.tail_guard_sec))
    target_dur = min(float(shot.duration_sec), safe_max_dur) if raw_dur > shot.duration_sec else safe_max_dur

    vf = (
        f"scale={project.width}:{project.height}:force_original_aspect_ratio=decrease,"
        f"pad={project.width}:{project.height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={project.fps},format=yuv420p,"
        "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
    )

    if project.audio_mode == "master_audio":
        # Strip all clip audio (-an); master audio will be muxed once at final assembly
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_clip),
            "-t", f"{target_dur:.3f}",
            "-vf", vf,
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            str(output_clip),
        ]
        subprocess.run(cmd, check=True)
    elif project.audio_mode == "clip_voiceover":
        # Replace H3 audio completely with external narration WAV (or silent stereo if shot has no WAV)
        wav_path: Optional[Path] = None
        if shot.narration_wav:
            cand = Path(shot.narration_wav)
            if not cand.is_absolute() and base_dir:
                cand = base_dir / cand
            if cand.exists():
                wav_path = cand

        if wav_path is not None:
            w_info = probe_media_streams(wav_path)
            w_dur = float(w_info.get("format", {}).get("duration") or target_dur)
            final_dur = min(safe_max_dur, max(target_dur, w_dur))
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_clip),
                "-i", str(wav_path),
                "-t", f"{final_dur:.3f}",
                "-vf", vf,
                "-af", f"aresample={project.sample_rate},aformat=sample_fmts=fltp:channel_layouts=stereo,apad=whole_dur={final_dur:.3f}",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                str(output_clip),
            ]
            subprocess.run(cmd, check=True)
            target_dur = final_dur
        else:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_clip),
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={project.sample_rate}",
                "-t", f"{target_dur:.3f}",
                "-vf", vf,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                str(output_clip),
            ]
            subprocess.run(cmd, check=True)
    else:
        # native_h3_audio: keep ONLY stream 0:a:0 from H3 clip (or silent fallback if missing)
        if has_raw_audio:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_clip),
                "-t", f"{target_dur:.3f}",
                "-vf", vf,
                "-af", f"aresample={project.sample_rate},aformat=sample_fmts=fltp:channel_layouts=stereo,apad=whole_dur={target_dur:.3f}",
                "-map", "0:v:0",
                "-map", "0:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                str(output_clip),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_clip),
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={project.sample_rate}",
                "-t", f"{target_dur:.3f}",
                "-vf", vf,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                str(output_clip),
            ]
        subprocess.run(cmd, check=True)

    out_info = probe_media_streams(output_clip)
    return float(out_info.get("format", {}).get("duration") or target_dur)


def build_contact_sheet(
    video_path: str | Path,
    output_jpg: str | Path,
    *,
    num_Shots: int = 6,
) -> Optional[Path]:
    """Generate a visual QC grid contact sheet (`_review_sheet.jpg`) to inspect continuity at a glance."""
    video_path = Path(video_path)
    output_jpg = Path(output_jpg)
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    info = probe_media_streams(video_path)
    dur = float(info.get("format", {}).get("duration") or 6.0)
    cols = 3
    rows = max(1, (max(1, num_Shots) + cols - 1) // cols)
    interval = max(0.5, dur / max(1, cols * rows))
    vf = f"fps=1/{interval:.3f},scale=420:-1,tile={cols}x{rows}:padding=6:margin=6"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-frames:v", "1",
        str(output_jpg),
    ]
    try:
        subprocess.run(cmd, check=True)
        return output_jpg if output_jpg.exists() else None
    except Exception:
        return None


def assemble_movie(
    project: StoryboardProject,
    raw_clips: Sequence[str | Path],
    output_dir: str | Path,
    *,
    base_dir: Optional[Path] = None,
    use_asr_alignment: bool = True,
    burn_subtitles: bool = True,
) -> Dict[str, Any]:
    """Normalize clips, assemble single-track MP4, generate & burn subtitles, and verify single audio stream."""
    project.validate()
    if len(raw_clips) != len(project.shots):
        raise ValueError(f"Expected {len(project.shots)} clips for {len(project.shots)} shots, got {len(raw_clips)}")

    out_root = Path(output_dir)
    norm_dir = out_root / "normalized_clips"
    norm_dir.mkdir(parents=True, exist_ok=True)

    clip_durations: List[float] = []
    norm_paths: List[Path] = []

    for shot, raw_p in zip(project.shots, raw_clips):
        norm_p = norm_dir / f"{shot.index:02d}_{shot.shot_id}_norm.mp4"
        dur = normalize_shot_clip(raw_p, norm_p, project=project, shot=shot, base_dir=base_dir)
        clip_durations.append(dur)
        norm_paths.append(norm_p)

    concat_txt = out_root / "concat_list.txt"
    concat_lines = [f"file '{p.resolve().as_posix()}'" for p in norm_paths]
    concat_txt.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    clean_mp4 = out_root / f"{project.title}_clean.mp4"

    if project.audio_mode == "master_audio":
        video_only_mp4 = out_root / f"{project.title}_video_only.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_txt),
                "-c", "copy",
                str(video_only_mp4),
            ],
            check=True,
        )
        master_audio = Path(project.master_audio_path) if project.master_audio_path else None
        if master_audio and not master_audio.is_absolute() and base_dir:
            master_audio = base_dir / master_audio

        total_dur = sum(clip_durations)
        if master_audio and master_audio.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(video_only_mp4),
                    "-i", str(master_audio),
                    "-t", f"{total_dur:.3f}",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                    "-movflags", "+faststart",
                    str(clean_mp4),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(video_only_mp4),
                    "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={project.sample_rate}",
                    "-t", f"{total_dur:.3f}",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                    "-movflags", "+faststart",
                    str(clean_mp4),
                ],
                check=True,
            )
    else:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_txt),
                "-map", "0:v:0",
                "-map", "0:a:0",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(project.sample_rate), "-ac", "2",
                "-movflags", "+faststart",
                str(clean_mp4),
            ],
            check=True,
        )

    clean_probe = verify_single_audio_stream(clean_mp4)

    # Generate subtitles (.zh-TW.srt and .ass)
    cues = build_cues_from_storyboard(project, clip_durations=clip_durations)
    if use_asr_alignment and cues:
        cues = align_cues_with_asr(cues, clean_mp4)

    srt_path = write_srt(cues, out_root / f"{project.title}.zh-TW.srt")
    ass_path = write_ass(
        cues,
        out_root / f"{project.title}.zh-TW.ass",
        width=project.width,
        height=project.height,
    )

    subtitled_mp4: Optional[Path] = None
    subtitled_probe: Optional[Dict[str, Any]] = None
    if burn_subtitles and cues:
        subtitled_mp4 = out_root / f"{project.title}_subtitled.mp4"
        burn_subtitles_into_video(clean_mp4, ass_path, subtitled_mp4)
        subtitled_probe = verify_single_audio_stream(subtitled_mp4)

    review_sheet = build_contact_sheet(
        subtitled_mp4 or clean_mp4,
        out_root / f"{project.title}_review_sheet.jpg",
        num_Shots=len(project.shots),
    )

    manifest = {
        "title": project.title,
        "audio_mode": project.audio_mode,
        "continuity_mode": project.continuity_mode,
        "clean_mp4": str(clean_mp4),
        "subtitled_mp4": str(subtitled_mp4) if subtitled_mp4 else None,
        "srt_path": str(srt_path),
        "ass_path": str(ass_path),
        "review_sheet": str(review_sheet) if review_sheet else None,
        "clip_durations": clip_durations,
        "clean_stream_verification": clean_probe,
        "subtitled_stream_verification": subtitled_probe,
    }
    (out_root / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest

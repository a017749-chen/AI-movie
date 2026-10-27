"""Post-production: trim tails, normalise clips, enforce ONE audio track, subtitles, QC sheet.

Single-audio policy (checked with ffprobe on every deliverable):
  native_h3_audio : keep only stream 0:a:0 of each H3 clip, loudness-levelled per clip so voices match.
  clip_voiceover  : drop H3 audio, map one external WAV per shot (-map 0:v:0 -map 1:a:0); silence if none.
  master_audio    : drop all clip audio (-an), concatenate picture, lay ONE master track under it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .script_loader import ShotSpec, StoryboardProject, H3_FPS
from .subtitles import align_cues_with_asr, build_cues_from_storyboard, burn_subtitles_into_video, write_ass, write_srt

X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p"]


def _run(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError("ffmpeg failed:\n" + " ".join(cmd) + "\n" + r.stderr[-3000:])


def probe_media_streams(media_path: str | Path) -> Dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate:format=duration",
         "-of", "json", str(media_path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out or "{}")


def media_duration(path: str | Path) -> float:
    return float(probe_media_streams(path).get("format", {}).get("duration") or 0.0)


def verify_single_audio_stream(video_path: str | Path) -> Dict[str, Any]:
    """Raise unless the file has exactly one video stream and exactly one audio stream."""
    info = probe_media_streams(video_path)
    v = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    a = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if len(v) != 1 or len(a) != 1:
        raise RuntimeError(f"{video_path}: expected 1 video + 1 audio stream, found {len(v)} video / {len(a)} audio")
    return {"video_stream": v[0], "audio_stream": a[0], "duration_sec": float(info["format"]["duration"])}


def _has_audio(info: Dict[str, Any]) -> bool:
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def normalize_shot_clip(raw_clip: str | Path, output_clip: str | Path, *, project: StoryboardProject,
                        shot: ShotSpec, base_dir: Optional[Path] = None) -> float:
    """Trim the unstable tail, conform picture (size/fps/bt709) and apply the audio policy. Returns seconds."""
    raw_clip, output_clip = Path(raw_clip), Path(output_clip)
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    info = probe_media_streams(raw_clip)
    raw_dur = float(info.get("format", {}).get("duration") or shot.duration_sec)
    usable = max(0.5, raw_dur - project.tail_guard_sec)
    dur = min(shot.duration_sec, usable)
    W, H, sr = project.width, project.height, project.sample_rate
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
          f"fps={H3_FPS},format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    t = f"{dur:.3f}"
    afmt = f"aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo"
    aac = ["-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2"]
    silence = ["-f", "lavfi", "-t", t, "-i", f"anullsrc=channel_layout=stereo:sample_rate={sr}"]

    if project.audio_mode == "master_audio":
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_clip), "-t", t, "-vf", vf, "-map", "0:v:0", "-an", *X264]
    elif project.audio_mode == "clip_voiceover":
        wav = None
        if shot.narration_wav:
            cand = Path(shot.narration_wav)
            if not cand.is_absolute() and base_dir is not None:
                cand = Path(base_dir) / cand
            wav = cand if cand.is_file() else None
            if wav is None:
                print(f"  warning: {shot.shot_id}: narration file {shot.narration_wav} not found, using silence")
        if wav is not None:
            wav_dur = media_duration(wav)
            if wav_dur > dur + 0.05:
                print(f"  warning: {shot.shot_id}: narration {wav_dur:.2f}s is longer than the shot {dur:.2f}s "
                      f"and will be cut; lengthen the shot")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_clip), "-i", str(wav), "-t", t, "-vf", vf,
                   "-af", f"{afmt},apad", "-map", "0:v:0", "-map", "1:a:0", *X264, *aac]
        else:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_clip), *silence, "-t", t, "-vf", vf,
                   "-map", "0:v:0", "-map", "1:a:0", *X264, *aac]
    else:  # native_h3_audio
        if _has_audio(info):
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_clip), "-t", t, "-vf", vf,
                   "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,{afmt},apad", "-map", "0:v:0", "-map", "0:a:0",
                   *X264, *aac]
        else:
            print(f"  warning: {shot.shot_id}: H3 clip has no audio stream, using silence")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_clip), *silence, "-t", t, "-vf", vf,
                   "-map", "0:v:0", "-map", "1:a:0", *X264, *aac]
    _run(cmd + [str(output_clip)])
    return media_duration(output_clip)


def build_contact_sheet(clips: Sequence[str | Path], labels: Sequence[str], output_jpg: str | Path,
                        cols: int = 4, thumb_w: int = 384) -> Optional[Path]:
    """One labelled frame (middle of each clip) per shot - a quick look at character/scene drift."""
    from PIL import Image, ImageDraw

    output_jpg = Path(output_jpg)
    tmp = output_jpg.parent / "_sheet_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, c in enumerate(clips):
        png = tmp / f"{i:03d}.png"
        mid = max(0.0, media_duration(c) / 2)
        try:
            _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{mid:.2f}", "-i", str(c), "-frames:v", "1",
                  "-vf", f"scale={thumb_w}:-2", str(png)])
            thumbs.append(Image.open(png).convert("RGB"))
        except RuntimeError:
            continue
    if not thumbs:
        return None
    th = thumbs[0].height
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * 6, rows * (th + 26) + 6), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs):
        x = 6 + (i % cols) * (thumb_w + 6)
        y = 6 + (i // cols) * (th + 26)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + th + 4), str(labels[i] if i < len(labels) else i), fill=(230, 230, 230))
    sheet.save(output_jpg, quality=88)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    return output_jpg


def assemble_movie(project: StoryboardProject, raw_clips: Sequence[str | Path], output_dir: str | Path, *,
                   base_dir: Optional[Path] = None, use_asr_alignment: bool = False, asr_model: str = "small",
                   asr_device: str = "cpu", burn_subtitles: bool = True) -> Dict[str, Any]:
    """Normalise all clips, build the clean + subtitled MP4s, verify one audio stream, write a manifest."""
    project.validate()
    if len(raw_clips) != len(project.shots):
        raise ValueError(f"{len(project.shots)} shots but {len(raw_clips)} clips")
    out = Path(output_dir)
    norm_dir = out / "normalized_clips"
    norm_dir.mkdir(parents=True, exist_ok=True)

    durations: List[float] = []
    norm: List[Path] = []
    for shot, raw in zip(project.shots, raw_clips):
        p = norm_dir / f"{shot.index:02d}_{shot.shot_id}.mp4"
        durations.append(normalize_shot_clip(raw, p, project=project, shot=shot, base_dir=base_dir))
        norm.append(p)
        print(f"  normalised {shot.shot_id}: {durations[-1]:.2f}s")

    concat = out / "concat_list.txt"
    concat.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in norm), encoding="utf-8")
    clean = out / f"{project.title}_clean.mp4"
    total = sum(durations)
    sr = str(project.sample_rate)
    aac = ["-c:a", "aac", "-b:a", "192k", "-ar", sr, "-ac", "2", "-movflags", "+faststart"]

    if project.audio_mode == "master_audio":
        picture = out / f"{project.title}_picture.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy",
              str(picture)])
        master = Path(project.master_audio_path) if project.master_audio_path else None
        if master is not None and not master.is_absolute() and base_dir is not None:
            master = Path(base_dir) / master
        if master is None or not master.is_file():
            raise FileNotFoundError(f"master_audio mode needs MASTER_AUDIO_PATH (got {project.master_audio_path!r})")
        if media_duration(master) < total - 0.1:
            print(f"  warning: master audio is shorter than the picture ({total:.1f}s); the end is silent")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(master), "-t", f"{total:.3f}",
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", "apad", *aac, str(clean)])
        picture.unlink(missing_ok=True)
    else:
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
              "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", *aac, str(clean)])
    clean_check = verify_single_audio_stream(clean)

    cues = build_cues_from_storyboard(project, clip_durations=durations)
    if use_asr_alignment and cues and project.audio_mode != "master_audio":
        cues = align_cues_with_asr(cues, clean, model_size=asr_model, device=asr_device)
    srt = write_srt(cues, out / f"{project.title}.zh-TW.srt")
    ass = write_ass(cues, out / f"{project.title}.zh-TW.ass", width=project.width, height=project.height)

    subtitled, sub_check = None, None
    if burn_subtitles and cues:
        subtitled = burn_subtitles_into_video(clean, ass, out / f"{project.title}_subtitled.mp4")
        sub_check = verify_single_audio_stream(subtitled)

    sheet = build_contact_sheet(norm, [s.shot_id for s in project.shots], out / f"{project.title}_review_sheet.jpg")
    manifest = {
        "title": project.title, "audio_mode": project.audio_mode, "continuity_mode": project.continuity_mode,
        "inference_mode": project.inference_mode, "clean_mp4": str(clean),
        "subtitled_mp4": str(subtitled) if subtitled else None, "srt_path": str(srt), "ass_path": str(ass),
        "review_sheet": str(sheet) if sheet else None, "clip_durations": durations, "total_sec": total,
        "subtitle_cues": len(cues), "clean_stream_verification": clean_check,
        "subtitled_stream_verification": sub_check,
    }
    (out / "render_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest

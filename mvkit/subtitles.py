"""Automatic Subtitle Generator, ASR Forced Aligner, and FFmpeg ASS Burner for AI-Movie.

Supports:
1. Storyboard-driven subtitle generation (`.zh-TW.srt` and `.ass` with clean movie/teaching styling):
   - Splits long sentences by punctuation into readable clauses (<= 18 chars per line)
   - Proportionally allocates durations within each shot's actual normalized duration
   - Also supports song-level absolute lyric cues (`project.subtitle_cues`) from `autoip-mv-maker` `project.py`
2. Optional `faster-whisper` ASR timing alignment (`align_cues_with_asr`):
   - Transcribes the final single audio stream for word timestamps while keeping the user's
     exact Traditional Chinese script text (preventing ASR homophone typos).
   - Defaults to CPU (`device="cpu"`) so it never collides with ComfyUI VRAM or missing cuDNN DLLs.
3. FFmpeg ASS burn-in (`burn_subtitles_into_video`) that strictly preserves the single audio stream.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .script_loader import StoryboardProject, clean_dialogue


@dataclass
class SubtitleCue:
    index: int
    start_sec: float
    end_sec: float
    text: str
    shot_id: str = ""


def split_chinese_clauses(text: str, max_chars: int = 18) -> List[str]:
    """Split a paragraph of Traditional Chinese dialogue into readable subtitle clauses."""
    clean = clean_dialogue(text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return []

    raw_parts = re.split(r"(?<=[。！？；!?…，,、：:])\s*", clean)
    clauses: List[str] = []
    buf = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        stripped_part = re.sub(r"[，,、：:。！？；!?]+$", "", part).strip()
        if not stripped_part:
            continue
        if not buf:
            buf = stripped_part
        elif len(buf) + 1 + len(stripped_part) <= max_chars:
            buf = f"{buf}，{stripped_part}"
        else:
            clauses.append(buf)
            buf = stripped_part
    if buf:
        clauses.append(buf)

    final_clauses: List[str] = []
    for c in clauses:
        while len(c) > int(max_chars * 1.4):
            final_clauses.append(c[:max_chars])
            c = c[max_chars:]
        if c:
            final_clauses.append(c)
    return final_clauses


def build_cues_from_storyboard(
    project: StoryboardProject,
    clip_durations: Optional[Sequence[float]] = None,
    *,
    max_chars_per_line: int = 18,
    lead_in_sec: float = 0.12,
    tail_pad_sec: float = 0.12,
) -> List[SubtitleCue]:
    """Generate `SubtitleCue` items from the storyboard's shots (or absolute `project.subtitle_cues` for MVs)."""
    cues: List[SubtitleCue] = []
    cue_idx = 1

    # If the project has absolute-timed MV lyric cues and shots have no per-shot dialogue, use subtitle_cues
    if project.subtitle_cues and not any((s.subtitle_text or s.dialogue) for s in project.shots):
        total_limit = sum(clip_durations) if clip_durations else None
        for start_s, end_s, raw_text in project.subtitle_cues:
            if total_limit is not None and start_s >= total_limit:
                continue
            clauses = split_chinese_clauses(raw_text, max_chars=max_chars_per_line)
            if not clauses:
                continue
            e_clamped = min(end_s, total_limit) if total_limit is not None else end_s
            span = max(0.4, e_clamped - start_s)
            total_w = sum(max(2, len(c)) for c in clauses)
            t_cur = start_s
            for i_c, clause in enumerate(clauses):
                w = max(2, len(clause))
                t_next = e_clamped if i_c == len(clauses) - 1 else round(t_cur + span * (w / total_w), 3)
                cues.append(
                    SubtitleCue(
                        index=cue_idx,
                        start_sec=round(t_cur, 3),
                        end_sec=round(max(t_cur + 0.25, t_next), 3),
                        text=clause,
                    )
                )
                cue_idx += 1
                t_cur = t_next
        return cues

    cursor = 0.0
    for idx, shot in enumerate(project.shots):
        dur = (
            float(clip_durations[idx])
            if clip_durations and idx < len(clip_durations)
            else float(shot.duration_sec)
        )
        shot_start = cursor
        shot_end = cursor + dur
        cursor = shot_end

        raw_text = (shot.subtitle_text or shot.dialogue or "").strip()
        clauses = split_chinese_clauses(raw_text, max_chars=max_chars_per_line)
        if not clauses:
            continue

        usable_start = shot_start + min(lead_in_sec, dur * 0.1)
        usable_end = max(usable_start + 0.4, shot_end - min(tail_pad_sec, dur * 0.1))
        usable_span = max(0.4, usable_end - usable_start)

        total_weight = sum(max(2, len(c)) for c in clauses)
        t_cur = usable_start
        for i_c, clause in enumerate(clauses):
            w = max(2, len(clause))
            seg_dur = usable_span * (w / total_weight)
            t_next = usable_end if i_c == len(clauses) - 1 else round(t_cur + seg_dur, 3)
            cues.append(
                SubtitleCue(
                    index=cue_idx,
                    start_sec=round(t_cur, 3),
                    end_sec=round(max(t_cur + 0.25, t_next), 3),
                    text=clause,
                    shot_id=shot.shot_id,
                )
            )
            cue_idx += 1
            t_cur = t_next

    return cues


def align_cues_with_asr(
    cues: List[SubtitleCue],
    audio_or_video_path: str | Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
) -> List[SubtitleCue]:
    """Refine cue start/end times using `faster-whisper` word timestamps while preserving exact script text."""
    if not cues:
        return cues
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[Subtitles] faster-whisper not installed; using storyboard proportional timings.")
        return cues

    compute_type = "int8_float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        prompt_text = "，".join(c.text for c in cues)
        segments, _ = model.transcribe(
            str(audio_or_video_path),
            language="zh",
            initial_prompt=prompt_text[:220],
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        words = [
            (float(w.start), float(w.end), w.word.strip())
            for seg in segments
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
    except Exception as exc:
        print(f"[Subtitles] ASR alignment skipped ({exc}); using storyboard proportional timings.")
        return cues

    if not words:
        return cues

    refined: List[SubtitleCue] = []
    for cue in cues:
        local_words = [
            w for w in words
            if w[1] >= cue.start_sec - 0.45 and w[0] <= cue.end_sec + 0.45
        ]
        if local_words:
            new_start = max(cue.start_sec - 0.25, min(cue.end_sec - 0.25, local_words[0][0]))
            new_end = min(cue.end_sec + 0.25, max(new_start + 0.35, local_words[-1][1]))
            refined.append(
                SubtitleCue(
                    index=cue.index,
                    start_sec=round(new_start, 3),
                    end_sec=round(new_end, 3),
                    text=cue.text,
                    shot_id=cue.shot_id,
                )
            )
        else:
            refined.append(cue)

    for i in range(len(refined) - 1):
        if refined[i].end_sec > refined[i + 1].start_sec:
            mid = round((refined[i].end_sec + refined[i + 1].start_sec) / 2.0, 3)
            refined[i].end_sec = max(refined[i].start_sec + 0.2, mid)
            refined[i + 1].start_sec = mid

    return refined


def _format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hh = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    mm = rem // 60_000
    rem %= 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    hh = cs_total // 360_000
    rem = cs_total % 360_000
    mm = rem // 6000
    rem %= 6000
    ss = rem // 100
    cs = rem % 100
    return f"{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}"


def write_srt(cues: Sequence[SubtitleCue], output_srt: str | Path) -> Path:
    """Write cues to a standard UTF-8 `.srt` subtitle file."""
    out = Path(output_srt)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks: List[str] = []
    for idx, c in enumerate(cues, start=1):
        blocks.append(
            f"{idx}\n{_format_srt_time(c.start_sec)} --> {_format_srt_time(c.end_sec)}\n{c.text}\n"
        )
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


def write_ass(
    cues: Sequence[SubtitleCue],
    output_ass: str | Path,
    *,
    width: int = 1344,
    height: int = 768,
    font_name: str = "Noto Sans CJK TC",
    font_size: int = 40,
    margin_v: int = 38,
) -> Path:
    """Write cues to an `.ass` subtitle file with a clean semi-transparent bottom box & outline."""
    out = Path(output_ass)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00141414,&H80000000,-1,0,0,0,100,100,1.2,0,3,2.2,0.0,2,48,48,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.strip()]
    for c in cues:
        safe_txt = c.text.replace("\n", r"\N")
        lines.append(
            f"Dialogue: 0,{_format_ass_time(c.start_sec)},{_format_ass_time(c.end_sec)},Default,,0,0,0,,{safe_txt}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for FFmpeg's `ass='...'` filter on both Linux/Colab and Windows."""
    raw = str(path.resolve()).replace("\\", "/")
    raw = raw.replace(":", r"\:").replace("'", r"\'")
    return raw


def burn_subtitles_into_video(
    input_mp4: str | Path,
    ass_path: str | Path,
    output_mp4: str | Path,
) -> Path:
    """Burn `.ass` subtitles into `input_mp4` while preserving the single audio track (`-map 0:v:0 -map 0:a:0`)."""
    in_p = Path(input_mp4)
    ass_p = Path(ass_path)
    out_p = Path(output_mp4)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    escaped_ass = _escape_ffmpeg_filter_path(ass_p)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(in_p),
        "-vf", f"ass='{escaped_ass}'",
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_p),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError("ffmpeg subtitle burn failed:\n" + " ".join(cmd) + "\n" + r.stderr[-3000:])
    return out_p

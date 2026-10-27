"""Unified Storyboard & Script Loader for AI-Movie (Colab MiniMax H3 + autoip-mv-maker).

Accepts user storyboards in any of three formats:
1. Markdown storyboard (`.md`): Markdown tables (`| 鏡號 | 秒數 | 畫面 | 旁白/台詞 | 字幕 | 參考圖 |`)
   or heading blocks (`### 01_identify ...`).
2. JSON storyboard (`.json`): `scenes.json`, `shotlist.json`, or unified `storyboard.json`.
3. Python MV project (`project.py` or project directory): native `autoip-mv-maker` format.

Solves two core production issues:
- Visual Continuity ("畫面跑掉、不連貫"):
  Locks `<Picture 1>` (scene/camera axis anchor) + `<Picture 2..N>` (character sheets),
  compiles structured `subject_definitions` + `retention_analysis` anti-drift blocks,
  and configures frame-chaining (`anchor` | `hybrid` | `chain_last_frame`) + `tail_guard`.
- Single Audio Track ("只要有一個聲軌就好"):
  Enforces mutually exclusive `audio_mode` (`native_h3_audio` | `clip_voiceover` | `master_audio`)
  both at prompt-generation time (preventing H3 from generating competing speech when external
  voiceover/master audio is used) and at FFmpeg muxing time.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_AUDIO_MODES = ("native_h3_audio", "clip_voiceover", "master_audio")
VALID_CONTINUITY_MODES = ("anchor", "hybrid", "chain_last_frame")
VALID_INFERENCE_MODES = ("reference", "first_frame", "text_only")


@dataclass
class CharacterSpec:
    id: str
    name: str
    description: str
    image_path: str = ""
    screen_position: str = ""  # e.g., "畫面左側", "畫面右側", "畫面中央"


@dataclass
class SceneAnchorSpec:
    id: str
    description: str
    image_path: str = ""
    axis_rule: str = "維持 180 度軸線一致，角色左右站位不可對調，背景陳設與光線色溫保持恆定"


@dataclass
class ShotSpec:
    shot_id: str
    index: int
    title: str = ""
    scene_id: str = "default"
    duration_sec: float = 6.0
    start_sec: float = 0.0
    end_sec: float = 6.0
    visual_prompt: str = ""
    camera_motion: str = "固定鏡位或極緩慢推鏡，禁止劇烈晃動與突然變焦"
    dialogue: str = ""          # Spoken dialogue / narration text
    dialogue_lang: str = "Chinese"
    subtitle_text: str = ""     # Text to display in subtitles (defaults to dialogue if empty)
    characters: List[str] = field(default_factory=list)
    reference_images: List[str] = field(default_factory=list)
    first_frame_image: str = ""
    narration_wav: str = ""
    seed: int = 1001
    continuity_override: str = ""  # Optional per-shot override: "anchor" | "chain_last_frame"
    compiled_h3_prompt: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoryboardProject:
    title: str = "AI_Movie_Project"
    aspect_ratio: str = "16:9"
    width: int = 1280
    height: int = 720
    fps: int = 24
    sample_rate: int = 48000
    audio_mode: str = "native_h3_audio"      # "native_h3_audio" | "clip_voiceover" | "master_audio"
    continuity_mode: str = "hybrid"          # "anchor" | "hybrid" | "chain_last_frame"
    inference_mode: str = "reference"        # "reference" (Ref2VA) | "first_frame" (FL2VA) | "text_only"
    ref_image_size: str = "max"              # "max" (up to 2048px, best identity lock) | "match"
    tail_guard_sec: float = 0.35             # Trim unstable tail frames from raw generation
    cfg: float = 4.0
    steps: int = 8
    master_audio_path: str = ""
    style_prefix: str = (
        "寫實電影級畫面，自然柔和環境光，五官清晰穩定，肢體動作自然流暢，攝影機運動平穩，"
        "嚴格保持人物臉部特徵、髮型、服裝與空間軸線連貫一致。"
    )
    negative_rules: str = (
        "禁止畫面閃爍、禁止人物五官變形或換臉、禁止角色左右位置跳動、禁止多手多指、"
        "禁止背景無故切換、禁止出現任何浮水印或畫面內嵌文字。"
    )
    characters: Dict[str, CharacterSpec] = field(default_factory=dict)
    scenes: Dict[str, SceneAnchorSpec] = field(default_factory=dict)
    shots: List[ShotSpec] = field(default_factory=list)

    def validate(self) -> None:
        if self.audio_mode not in VALID_AUDIO_MODES:
            raise ValueError(f"Invalid audio_mode={self.audio_mode!r}; must be one of {VALID_AUDIO_MODES}")
        if self.continuity_mode not in VALID_CONTINUITY_MODES:
            raise ValueError(f"Invalid continuity_mode={self.continuity_mode!r}; must be one of {VALID_CONTINUITY_MODES}")
        if self.inference_mode not in VALID_INFERENCE_MODES:
            raise ValueError(f"Invalid inference_mode={self.inference_mode!r}; must be one of {VALID_INFERENCE_MODES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "aspect_ratio": self.aspect_ratio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "audio_mode": self.audio_mode,
            "continuity_mode": self.continuity_mode,
            "inference_mode": self.inference_mode,
            "ref_image_size": self.ref_image_size,
            "tail_guard_sec": self.tail_guard_sec,
            "cfg": self.cfg,
            "steps": self.steps,
            "master_audio_path": self.master_audio_path,
            "style_prefix": self.style_prefix,
            "negative_rules": self.negative_rules,
            "characters": {k: asdict(v) for k, v in self.characters.items()},
            "scenes": {k: asdict(v) for k, v in self.scenes.items()},
            "shots": [s.to_dict() for s in self.shots],
        }


def _clean_dialogue_tags(text: str) -> str:
    """Strip any existing `<d>...</d>` wrapper and return clean spoken text."""
    if not text:
        return ""
    cleaned = re.sub(r"<d>\s*(?:\[[A-Za-z\-]+\]\s*)?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"</d>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def compile_shot_prompt(project: StoryboardProject, shot: ShotSpec) -> str:
    """Compile a structured, anti-drift MiniMax H3 prompt with `<Picture k>` bindings and single-audio rules."""
    project.validate()

    # Determine bound reference pictures:
    # <Picture 1> is always the scene anchor (or chained frame from previous shot in same scene)
    # <Picture 2..N> are character reference sheets
    pic_bindings: List[str] = []
    scene_spec = project.scenes.get(shot.scene_id) or next(iter(project.scenes.values()), None)

    pic_idx = 1
    if project.inference_mode == "reference":
        if scene_spec and (scene_spec.description or scene_spec.image_path):
            pic_bindings.append(
                f"- <Picture {pic_idx}>：場景與鏡位基準圖（{scene_spec.description}；{scene_spec.axis_rule}）。"
            )
            pic_idx += 1
        elif shot.reference_images:
            pic_bindings.append(
                f"- <Picture {pic_idx}>：主場景與角色站位基準圖（嚴格維持此構圖軸線、背景與光影色溫）。"
            )
            pic_idx += 1

        for cid in shot.characters:
            ch = project.characters.get(cid)
            if ch:
                pos_note = f"，固定於{ch.screen_position}" if ch.screen_position else ""
                pic_bindings.append(
                    f"- <Picture {pic_idx}>：{ch.name}（{ch.description}{pos_note}，臉部五官、髮型與服裝須與參考圖完全一致）。"
                )
                pic_idx += 1

    # Audio instruction strictly follows project.audio_mode so two voices NEVER overlap
    clean_spoken = _clean_dialogue_tags(shot.dialogue)
    if project.audio_mode == "native_h3_audio":
        if clean_spoken:
            audio_block = (
                f"【單一聲軌規範（原生語音）】僅允許單一清晰人聲唸出以下台詞，禁止背景音樂、禁止重疊第二人聲或雜音：\n"
                f"<d>[{shot.dialogue_lang}] {clean_spoken}</d>"
            )
        else:
            audio_block = (
                "【單一聲軌規範（純環境音）】本鏡頭無台詞對白，人物嘴部保持自然閉合，僅保留極輕微自然的單一環境底噪，禁止人聲與音樂。"
            )
    elif project.audio_mode == "clip_voiceover":
        audio_block = (
            "【單一聲軌規範（外部旁白替換模式）】本鏡頭音軌將於後製由外部單一旁白音檔完全替換。"
            "生成時禁止發出任何說話人聲、禁止背景音樂，人物聚焦於肢體與表情演出，避免嘴型與外部旁白衝突。"
        )
    else:  # master_audio
        audio_block = (
            "【單一聲軌規範（主音軌對齊模式）】最終影片僅使用單一主音軌（Master Audio），"
            "生成片段之原生音軌將以 -an 完全移除，請專注於畫面視覺連貫性與節奏。"
        )

    parts: List[str] = [f"【風格與連貫性總則】{project.style_prefix}"]
    if pic_bindings:
        parts.append("【參考圖綁定 (Subject & Scene Lock)】\n" + "\n".join(pic_bindings))
    if scene_spec:
        parts.append(f"【場景軸線鎖定】{scene_spec.description}。{scene_spec.axis_rule}。")
    parts.append(f"【本鏡動作與構圖（鏡頭 {shot.shot_id}，{shot.duration_sec:.1f} 秒）】{shot.visual_prompt}")
    if shot.camera_motion:
        parts.append(f"【運鏡規範】{shot.camera_motion}")
    parts.append(audio_block)
    parts.append(f"【防跑掉與負面限制 (Retention & Anti-Drift)】{project.negative_rules}")

    prompt = "\n".join(p for p in parts if p.strip())
    shot.compiled_h3_prompt = prompt
    return prompt


def resolve_shot_references(project: StoryboardProject, shot: ShotSpec, base_dir: Optional[Path] = None) -> List[str]:
    """Return the ordered list of reference image paths (`<Picture 1>`, `<Picture 2>`, ...) for a shot."""
    refs: List[str] = []
    if shot.reference_images:
        refs.extend(shot.reference_images)
    else:
        scene_spec = project.scenes.get(shot.scene_id) or next(iter(project.scenes.values()), None)
        if scene_spec and scene_spec.image_path:
            refs.append(scene_spec.image_path)
        for cid in shot.characters:
            ch = project.characters.get(cid)
            if ch and ch.image_path and ch.image_path not in refs:
                refs.append(ch.image_path)

    if base_dir is not None:
        resolved: List[str] = []
        for r in refs:
            p = Path(r)
            if not p.is_absolute() and (base_dir / p).exists():
                resolved.append(str((base_dir / p).resolve()))
            else:
                resolved.append(r)
        return resolved
    return refs


def _parse_duration_cell(raw: str, default: float = 6.0) -> tuple[float, float, float]:
    """Parse cells like '0-10s', '00:00 - 00:10 (10s)', '8s', '6.5' into (start, end, duration)."""
    text = (raw or "").strip()
    if not text:
        return (0.0, default, default)

    # Check range like 00:10-00:20 or 10-20s
    m_mmss = re.search(r"(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[-~～至到]\s*(\d{1,2}):(\d{2}(?:\.\d+)?)", text)
    if m_mmss:
        s = int(m_mmss.group(1)) * 60 + float(m_mmss.group(2))
        e = int(m_mmss.group(3)) * 60 + float(m_mmss.group(4))
        d = max(1.0, e - s)
        return (s, e, d)

    m_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~～至到]\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?", text)
    if m_range:
        s = float(m_range.group(1))
        e = float(m_range.group(2))
        d = max(1.0, e - s)
        return (s, e, d)

    m_single = re.search(r"(\d+(?:\.\d+)?)", text)
    if m_single:
        d = max(1.0, float(m_single.group(1)))
        return (0.0, d, d)

    return (0.0, default, default)


def load_storyboard_from_markdown(
    md_text: str,
    *,
    title: str = "Markdown_Storyboard",
    audio_mode: str = "native_h3_audio",
    continuity_mode: str = "hybrid",
    default_reference_image: str = "",
    base_dir: Optional[Path] = None,
) -> StoryboardProject:
    """Parse a Markdown storyboard table or section list into a `StoryboardProject`."""
    proj = StoryboardProject(
        title=title,
        audio_mode=audio_mode,
        continuity_mode=continuity_mode,
    )
    if default_reference_image:
        proj.scenes["default"] = SceneAnchorSpec(
            id="default",
            description="預設主場景基準圖",
            image_path=default_reference_image,
        )

    lines = [ln.strip() for ln in md_text.splitlines()]
    table_rows: List[List[str]] = []
    headers: List[str] = []

    for ln in lines:
        if not ln.startswith("|") or not ln.endswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.match(r"^[:\-]+$", c.replace(" ", "")) for c in cells if c):
            continue
        if not headers:
            headers = [c.lower() for c in cells]
        else:
            table_rows.append(cells)

    shots: List[ShotSpec] = []
    cursor_sec = 0.0

    if table_rows and headers:
        def _col_idx(keywords: tuple[str, ...], fallback: int = -1) -> int:
            for idx, h in enumerate(headers):
                if any(k in h for k in keywords):
                    return idx
            return fallback

        id_col = _col_idx(("鏡號", "鏡頭", "shot", "id", "編號", "段落"), 0)
        dur_col = _col_idx(("秒數", "時間", "時長", "duration", "time", "sec"), 1)
        scene_col = _col_idx(("場景", "scene"), -1)
        vis_col = _col_idx(("畫面", "視覺", "動作", "描述", "visual", "prompt", "action"), 2)
        cam_col = _col_idx(("運鏡", "鏡位", "camera"), -1)
        dlg_col = _col_idx(("旁白", "台詞", "對白", "語音", "dialogue", "narration", "audio", "voice"), 3)
        sub_col = _col_idx(("字幕", "caption", "subtitle"), -1)
        ref_col = _col_idx(("參考圖", "圖片", "ref", "image"), -1)

        for idx, row in enumerate(table_rows, start=1):
            def _get(c: int) -> str:
                return row[c].strip() if 0 <= c < len(row) else ""

            raw_id = _get(id_col) or f"shot_{idx:02d}"
            shot_id = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", raw_id).strip("_") or f"shot_{idx:02d}"
            s_sec, e_sec, dur = _parse_duration_cell(_get(dur_col), default=6.0)
            if s_sec == 0.0 and idx > 1:
                s_sec = cursor_sec
                e_sec = s_sec + dur
            cursor_sec = e_sec

            visual = _get(vis_col)
            cam = _get(cam_col) or "固定鏡位或平穩緩慢推鏡，維持軸線與構圖穩定"
            dialogue = _get(dlg_col)
            subtitle = _get(sub_col) or _clean_dialogue_tags(dialogue)
            scene_id = _get(scene_col) or "default"
            ref_cell = _get(ref_col)
            refs = [r.strip() for r in re.split(r"[,;，；\s]+", ref_cell) if r.strip()] if ref_cell else []
            if not refs and default_reference_image:
                refs = [default_reference_image]

            shot = ShotSpec(
                shot_id=shot_id,
                index=idx,
                title=raw_id,
                scene_id=scene_id,
                duration_sec=dur,
                start_sec=s_sec,
                end_sec=e_sec,
                visual_prompt=visual or subtitle or f"鏡頭 {shot_id} 穩定敘事畫面",
                camera_motion=cam,
                dialogue=dialogue,
                subtitle_text=subtitle,
                reference_images=refs,
                seed=1000 + idx,
            )
            compile_shot_prompt(proj, shot)
            shots.append(shot)
    else:
        # Fallback: parse heading blocks (`### ...` or `## ...`)
        blocks = re.split(r"(?m)^#{2,3}\s+", md_text)
        idx = 0
        for blk in blocks[1:]:
            blk_lines = [l.strip() for l in blk.splitlines() if l.strip()]
            if not blk_lines:
                continue
            idx += 1
            heading = blk_lines[0]
            s_sec, e_sec, dur = _parse_duration_cell(heading, default=6.0)
            if s_sec == 0.0 and idx > 1:
                s_sec = cursor_sec
                e_sec = s_sec + dur
            cursor_sec = e_sec

            visual_parts: List[str] = []
            dialogue_parts: List[str] = []
            subtitle_parts: List[str] = []
            for ln in blk_lines[1:]:
                if re.match(r"^[-*]?\s*\**(?:旁白|台詞|對白|語音|Narration|Dialogue)\**[：:]", ln, re.I):
                    dialogue_parts.append(re.sub(r"^[-*]?\s*\**[^：:]+\**[：:]\s*", "", ln).strip("「」\"' "))
                elif re.match(r"^[-*]?\s*\**(?:字幕|Subtitle|Caption)\**[：:]", ln, re.I):
                    subtitle_parts.append(re.sub(r"^[-*]?\s*\**[^：:]+\**[：:]\s*", "", ln).strip("「」\"' "))
                else:
                    clean_ln = re.sub(r"^[-*]?\s*\**(?:畫面|視覺|動作|Visual|Prompt)\**[：:]\s*", "", ln, flags=re.I)
                    visual_parts.append(clean_ln)

            dialogue = " ".join(dialogue_parts).strip()
            subtitle = " ".join(subtitle_parts).strip() or _clean_dialogue_tags(dialogue)
            visual = " ".join(visual_parts).strip() or subtitle or heading
            shot_id = f"shot_{idx:02d}"
            refs = [default_reference_image] if default_reference_image else []
            shot = ShotSpec(
                shot_id=shot_id,
                index=idx,
                title=heading,
                duration_sec=dur,
                start_sec=s_sec,
                end_sec=e_sec,
                visual_prompt=visual,
                dialogue=dialogue,
                subtitle_text=subtitle,
                reference_images=refs,
                seed=1000 + idx,
            )
            compile_shot_prompt(proj, shot)
            shots.append(shot)

    proj.shots = shots
    return proj


def load_storyboard_from_json(
    data: Any,
    *,
    title: str = "JSON_Storyboard",
    audio_mode: Optional[str] = None,
    continuity_mode: Optional[str] = None,
    default_reference_image: str = "",
    base_dir: Optional[Path] = None,
) -> StoryboardProject:
    """Parse `scenes.json` (list of dicts), `shotlist.json`, or a full `StoryboardProject` JSON dict."""
    if isinstance(data, list):
        raw_shots = data
        meta: Dict[str, Any] = {}
    elif isinstance(data, dict):
        meta = data
        raw_shots = (
            data.get("shots")
            or data.get("scenes_list")
            or data.get("shotlist")
            or (data.get("scenes") if isinstance(data.get("scenes"), list) else [])
        )
    else:
        raise ValueError("Unsupported JSON storyboard structure")

    proj = StoryboardProject(
        title=meta.get("title", title),
        aspect_ratio=meta.get("aspect_ratio", "16:9"),
        width=int(meta.get("width", 1280)),
        height=int(meta.get("height", 720)),
        fps=int(meta.get("fps", 24)),
        sample_rate=int(meta.get("sample_rate", 48000)),
        audio_mode=audio_mode or meta.get("audio_mode", "native_h3_audio"),
        continuity_mode=continuity_mode or meta.get("continuity_mode", "hybrid"),
        inference_mode=meta.get("inference_mode", "reference"),
        ref_image_size=meta.get("ref_image_size", "max"),
        tail_guard_sec=float(meta.get("tail_guard_sec", 0.35)),
        cfg=float(meta.get("cfg", 4.0)),
        steps=int(meta.get("steps", 8)),
        master_audio_path=meta.get("master_audio_path", ""),
        style_prefix=meta.get("style_prefix", StoryboardProject.style_prefix),
        negative_rules=meta.get("negative_rules", StoryboardProject.negative_rules),
    )

    # Load characters if present
    for cid, cinfo in (meta.get("characters") or {}).items():
        if isinstance(cinfo, dict):
            proj.characters[cid] = CharacterSpec(
                id=cid,
                name=cinfo.get("name", cid),
                description=cinfo.get("description", ""),
                image_path=cinfo.get("image_path", ""),
                screen_position=cinfo.get("screen_position", ""),
            )
        elif isinstance(cinfo, str):
            proj.characters[cid] = CharacterSpec(id=cid, name=cid, description=cinfo)

    # Load scenes if dict
    if isinstance(meta.get("scenes"), dict):
        for sid, sinfo in meta["scenes"].items():
            if isinstance(sinfo, dict):
                proj.scenes[sid] = SceneAnchorSpec(
                    id=sid,
                    description=sinfo.get("description", ""),
                    image_path=sinfo.get("image_path", default_reference_image),
                    axis_rule=sinfo.get("axis_rule", SceneAnchorSpec.axis_rule),
                )
            elif isinstance(sinfo, str):
                proj.scenes[sid] = SceneAnchorSpec(id=sid, description=sinfo, image_path=default_reference_image)

    if default_reference_image and "default" not in proj.scenes:
        proj.scenes["default"] = SceneAnchorSpec(
            id="default",
            description="預設主場景基準圖",
            image_path=default_reference_image,
        )

    cursor_sec = 0.0
    for idx, item in enumerate(raw_shots, start=1):
        shot_id = str(item.get("shot_id") or item.get("id") or f"shot_{idx:02d}")
        dur = float(item.get("duration_sec") or item.get("duration") or item.get("seconds") or 6.0)
        s_sec = float(item.get("start_sec", cursor_sec))
        e_sec = float(item.get("end_sec", s_sec + dur))
        cursor_sec = e_sec

        # Handle prompt file or inline visual prompt
        visual = item.get("visual_prompt") or item.get("prompt") or item.get("visual") or item.get("action") or ""
        prompt_file = item.get("prompt_file")
        if not visual and prompt_file and base_dir and (base_dir / prompt_file).exists():
            visual = (base_dir / prompt_file).read_text(encoding="utf-8").strip()

        dialogue = item.get("dialogue") or item.get("narration") or item.get("spoken_text") or ""
        if not dialogue and visual:
            m_dlg = re.search(r"<d>\s*(?:\[[A-Za-z\-]+\]\s*)?(.*?)</d>", visual, flags=re.S | re.I)
            if m_dlg:
                dialogue = m_dlg.group(1).strip()

        subtitle = item.get("subtitle_text") or item.get("subtitle") or item.get("caption") or _clean_dialogue_tags(dialogue)

        refs: List[str] = []
        if item.get("reference_images"):
            refs = list(item["reference_images"])
        elif item.get("reference_image"):
            refs = [str(item["reference_image"])]
        elif item.get("ref_images"):
            refs = list(item["ref_images"])
        elif default_reference_image:
            refs = [default_reference_image]

        shot = ShotSpec(
            shot_id=shot_id,
            index=int(item.get("index", idx)),
            title=str(item.get("title") or shot_id),
            scene_id=str(item.get("scene_id") or item.get("scene") or "default"),
            duration_sec=dur,
            start_sec=s_sec,
            end_sec=e_sec,
            visual_prompt=visual,
            camera_motion=str(item.get("camera_motion") or item.get("camera") or ShotSpec.camera_motion),
            dialogue=dialogue,
            dialogue_lang=str(item.get("dialogue_lang", "Chinese")),
            subtitle_text=subtitle,
            characters=list(item.get("characters") or []),
            reference_images=refs,
            first_frame_image=str(item.get("first_frame_image") or item.get("first_frame") or ""),
            narration_wav=str(item.get("narration_wav") or ""),
            seed=int(item.get("seed", 1000 + idx)),
            continuity_override=str(item.get("continuity_override", "")),
            notes=str(item.get("notes", "")),
        )
        shot.reference_images = resolve_shot_references(proj, shot, base_dir=base_dir)
        compile_shot_prompt(proj, shot)
        proj.shots.append(shot)

    return proj


def load_storyboard_from_mvkit_project(
    project_path: str | Path,
    *,
    audio_mode: str = "master_audio",
    continuity_mode: str = "hybrid",
) -> StoryboardProject:
    """Adapt an `autoip-mv-maker` `project.py` directory/file into a `StoryboardProject`."""
    from .common import load_project, audio_path
    from .storyboard import project_spec

    proj_mod = load_project(str(project_path))
    spec = project_spec(proj_mod)

    master_wav = ""
    try:
        cand = audio_path(proj_mod, "wav")
        if os.path.exists(cand):
            master_wav = cand
    except Exception:
        pass

    sb = StoryboardProject(
        title=getattr(proj_mod, "TITLE", Path(str(project_path)).stem),
        aspect_ratio=getattr(proj_mod, "ASPECT_RATIO", "16:9"),
        audio_mode=audio_mode,
        continuity_mode=continuity_mode,
        master_audio_path=master_wav,
        style_prefix=getattr(proj_mod, "STYLE", StoryboardProject.style_prefix),
        tail_guard_sec=float(getattr(proj_mod, "TAIL_GUARD", 0.35)),
    )

    for cid, cdesc in (spec.get("characters") or {}).items():
        ref_file = (getattr(proj_mod, "REFS", {}) or {}).get(cid, f"{cid}.png")
        ref_full = os.path.join(proj_mod.PROJECT_DIR, "refs", ref_file)
        sb.characters[cid] = CharacterSpec(
            id=cid,
            name=cid,
            description=cdesc,
            image_path=ref_full if os.path.exists(ref_full) else ref_file,
        )

    for sid, sdesc in (spec.get("scenes") or {}).items():
        ref_file = (getattr(proj_mod, "REFS", {}) or {}).get(sid, f"{sid}.png")
        ref_full = os.path.join(proj_mod.PROJECT_DIR, "refs", ref_file)
        sb.scenes[sid] = SceneAnchorSpec(
            id=sid,
            description=sdesc,
            image_path=ref_full if os.path.exists(ref_full) else ref_file,
        )

    idx = 0
    for sec in spec.get("sections", []):
        sec_name = sec.get("name", "section")
        scene_id = sec.get("scene", "default")
        chars = list(sec.get("characters") or [])
        lines = sec.get("lines") or []
        sec_start = float(sec.get("start", 0.0))
        sec_end = float(sec.get("end", sec_start + 6.0))
        sec_dur = max(2.0, sec_end - sec_start)

        if lines:
            for l_idx, ln in enumerate(lines):
                idx += 1
                l_start = float(ln.get("t", sec_start))
                l_end = float(lines[l_idx + 1]["t"]) if l_idx + 1 < len(lines) else sec_end
                l_dur = max(2.0, round(l_end - l_start, 2))
                shot = ShotSpec(
                    shot_id=f"{sec_name}_{l_idx:02d}",
                    index=idx,
                    title=f"{sec_name} #{l_idx + 1}",
                    scene_id=scene_id,
                    duration_sec=l_dur,
                    start_sec=l_start,
                    end_sec=l_end,
                    visual_prompt=f"{sec.get('Scene_Description', '')}；角色動作：{ln.get('action', '')}",
                    dialogue=ln.get("lyric", ""),
                    subtitle_text=ln.get("lyric", ""),
                    characters=chars,
                    seed=1000 + idx,
                )
                shot.reference_images = resolve_shot_references(sb, shot)
                compile_shot_prompt(sb, shot)
                sb.shots.append(shot)
        else:
            idx += 1
            shot = ShotSpec(
                shot_id=f"{sec_name}_00",
                index=idx,
                title=sec_name,
                scene_id=scene_id,
                duration_sec=sec_dur,
                start_sec=sec_start,
                end_sec=sec_end,
                visual_prompt=f"{sec.get('Scene_Description', '')}；{sec.get('action', '')}",
                characters=chars,
                seed=1000 + idx,
            )
            shot.reference_images = resolve_shot_references(sb, shot)
            compile_shot_prompt(sb, shot)
            sb.shots.append(shot)

    return sb


def load_storyboard(
    source: str | Path,
    *,
    audio_mode: Optional[str] = None,
    continuity_mode: Optional[str] = None,
    default_reference_image: str = "",
) -> StoryboardProject:
    """Auto-detect and load a storyboard from `.md`, `.json`, or `project.py` (or directory)."""
    p = Path(source)
    if p.is_dir() or p.name.endswith(".py"):
        return load_storyboard_from_mvkit_project(
            p,
            audio_mode=audio_mode or "master_audio",
            continuity_mode=continuity_mode or "hybrid",
        )
    if not p.exists():
        raise FileNotFoundError(f"Storyboard source not found: {p}")

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        return load_storyboard_from_json(
            raw,
            title=p.stem,
            audio_mode=audio_mode,
            continuity_mode=continuity_mode,
            default_reference_image=default_reference_image,
            base_dir=p.parent,
        )

    md_text = p.read_text(encoding="utf-8")
    return load_storyboard_from_markdown(
        md_text,
        title=p.stem,
        audio_mode=audio_mode or "native_h3_audio",
        continuity_mode=continuity_mode or "hybrid",
        default_reference_image=default_reference_image,
        base_dir=p.parent,
    )

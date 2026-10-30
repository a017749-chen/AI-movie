"""Storyboard loader and MiniMax H3 prompt builder.

Input formats (all become one `StoryboardProject`):
  * Markdown table (`.md`)       | 鏡號 | 秒數 | 場景 | 角色 | 畫面 | 運鏡 | 旁白 | 字幕 | 參考圖 |
  * JSON (`.json`)               storyboard_template.json, scenes.json, shotlist.json
  * autoip-mv-maker `project.py` CHARS / BROLL / BAND / LS_SCENES / SECTIONS

Each storyboard row is generated as ONE H3 clip (5-15 s). Continuity between clips comes from
(1) a fixed picture plan (`<Picture 1>` = scene/start frame, `<Picture 2..>` = character sheets, Ref2VA only)
and (2) optional last-frame chaining done by `h3_colab_runner`.

Prompts follow the format used by the upstream MiniMax_H3_Turbo_Colab notebook and the ISBAR prompts that
worked in practice:

    For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
    integrated_multimodal_description: [Shot 1] ...
    overall_soundscape: ...
    non_diegetic_music: N/A

Audio policy (`audio_mode`) decides what the prompt asks H3 to voice, so the model never produces a voice that
post-production will later stack a second voice on top of:
  * native_h3_audio : H3 speaks the line in `<d>[Chinese] ...</d>`; that audio is the only track kept.
  * clip_voiceover  : H3 is told nobody speaks; its audio is replaced by one WAV per shot.
  * master_audio    : H3 is told nobody speaks; all clip audio is dropped and one master track is laid under.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_AUDIO_MODES = ("native_h3_audio", "clip_voiceover", "master_audio")
VALID_CONTINUITY_MODES = ("hybrid", "chain_last_frame", "anchor")
VALID_INFERENCE_MODES = ("reference", "first_frame")
# per-segment generation modes -> the H3 model they need
GEN_MODE_ENGINE = {
    "reference": "reference",            # Ref2VA: scene + character pictures
    "reference_keyframes": "reference",  # Ref2VA + MiniMaxH3AddGuide pinning a first and/or last keyframe
    "first_frame": "first_frame",        # FL2VA with a first frame
    "first_last": "first_frame",         # FL2VA with first and last frame
}
CHARACTER_TIERS = ("main", "secondary", "npc")

H3_FPS = 24
H3_MAX_AREA = 1344 * 768
H3_MIN_SECONDS = 4.0
H3_MAX_SECONDS = 15.0
H3_STABLE_MIN_SECONDS = 5.0          # ~124 frames, lower edge of the trained range
MAX_REFERENCE_IMAGES = 9

ASPECT_PRESETS = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (992, 992),
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def h3_frames(seconds: float) -> int:
    """Frame count ComfyUI's H3 nodes accept for a requested duration (24 fps, aligned to 17k+5)."""
    requested = max(5, round(seconds * H3_FPS))
    return requested + (5 - requested % 17) % 17


@dataclass
class CharacterSpec:
    id: str
    name: str
    description: str = ""
    image_path: str = ""
    screen_position: str = ""        # e.g. "left side of frame"
    aliases: List[str] = field(default_factory=list)   # names the script uses for this person ("考生", "林小姐")
    voice: str = ""                  # native_h3_audio: the same voice description in every clip
    lead: bool = False               # speaks the "（口述）" / "（對病人）" lines of the script
    tier: str = ""                   # main | secondary | npc: decides who keeps a picture when slots run out
    extra_images: List[str] = field(default_factory=list)   # turnaround sheet, face close-up (main characters)


@dataclass
class SceneAnchorSpec:
    id: str
    description: str = ""
    image_path: str = ""
    name: str = ""
    aliases: List[str] = field(default_factory=list)   # place names the script uses ("病床旁", "治療台")
    cast: List[str] = field(default_factory=list)      # character ids always present in this scene


@dataclass
class BeatSpec:
    """One 分鏡 inside a segment: H3 renders all beats of a segment as cuts inside ONE clip."""
    beat_id: str
    duration_sec: float = 3.0
    framing: str = ""                 # 遠景 | 全景 | 中景 | 近景 | 特寫
    lighting: str = ""                # 色調打光
    visual: str = ""                  # 畫面描述
    dialogue: str = ""                # 旁白對白 (spoken words only)
    speaker: str = ""                 # character id after alignment, "" = narrator
    sound: str = ""                   # 音效配樂
    camera: str = ""                  # 運鏡
    missing_fields: List[str] = field(default_factory=list)   # modules the script did not write


@dataclass
class ShotSpec:
    shot_id: str
    index: int
    title: str = ""
    scene_id: str = "default"
    duration_sec: float = 6.0         # length kept in the final cut
    visual_prompt: str = ""
    camera_motion: str = ""
    dialogue: str = ""                # spoken text (Traditional Chinese)
    speaker: str = ""                 # empty = the project narrator (off screen)
    subtitle_text: str = ""           # defaults to dialogue
    characters: List[str] = field(default_factory=list)
    reference_images: List[str] = field(default_factory=list)   # explicit picture list overrides the plan
    narration_wav: str = ""           # clip_voiceover mode
    seed: int = 0
    continuity_override: str = ""     # "anchor" | "chain_last_frame" for this shot only
    notes: str = ""
    beats: List[BeatSpec] = field(default_factory=list)   # segment = several 分鏡 cut inside one clip
    gen_mode: str = ""                # "" = project.inference_mode; see GEN_MODE_ENGINE
    first_frame: str = ""             # keyframe picture pinned at frame 0 (融圖 key frame)
    last_frame: str = ""              # keyframe picture pinned at the last frame

    @property
    def start_framing(self) -> str:
        return self.beats[0].framing if self.beats else ""

    @property
    def end_framing(self) -> str:
        return self.beats[-1].framing if self.beats else ""


@dataclass
class PictureRef:
    path: str
    role: str                         # "scene" | "character" | "chain" | "extra" | "keyframe"
    label: str                        # sentence used in the prompt
    char_id: str = ""


@dataclass
class StoryboardProject:
    title: str = "AI_Movie"
    aspect_ratio: str = "16:9"
    width: int = 1344
    height: int = 768
    sample_rate: int = 48000
    audio_mode: str = "native_h3_audio"
    continuity_mode: str = "hybrid"
    inference_mode: str = "reference"
    ref_image_size: str = "max"
    tail_guard_sec: float = 0.35
    base_seed: int = 768914106098754
    master_audio_path: str = ""
    preset: str = ""                  # osce_demo | teaching | recruitment (see presets.py)
    style_prefix: str = (
        "Photorealistic cinematic film, natural soft light, stable faces, natural body movement, "
        "restrained camera motion."
    )
    continuity_rules: str = (
        "Keep every person's face, hairstyle, clothing and props, the room layout, screen positions and "
        "lighting exactly consistent with the reference pictures."
    )
    negative_rules: str = (
        "No flicker, no face changes, no extra people, anatomically correct hands, "
        "no captions, no logos, no generated text or watermarks."
    )
    narrator: str = "A calm female off-screen narrator (S1) speaks in clear Taiwan Mandarin at a measured pace"
    overall_soundscape: str = "Quiet indoor room tone with soft natural movement sounds."
    characters: Dict[str, CharacterSpec] = field(default_factory=dict)
    scenes: Dict[str, SceneAnchorSpec] = field(default_factory=dict)
    shots: List[ShotSpec] = field(default_factory=list)
    subtitle_cues: List[Tuple[float, float, str]] = field(default_factory=list)  # absolute timings (MV lyrics)

    # ------------------------------------------------------------------ settings
    def set_aspect(self, aspect: str) -> None:
        if aspect not in ASPECT_PRESETS:
            raise ValueError(f"aspect_ratio must be one of {sorted(ASPECT_PRESETS)}")
        self.aspect_ratio = aspect
        self.width, self.height = ASPECT_PRESETS[aspect]

    def generation_seconds(self, shot: ShotSpec) -> float:
        """Requested H3 length: the kept length plus the tail that gets trimmed, never below 5 s."""
        return min(H3_MAX_SECONDS, max(H3_STABLE_MIN_SECONDS, shot.duration_sec + self.tail_guard_sec))

    def validate(self) -> None:
        errors: List[str] = []
        if self.audio_mode not in VALID_AUDIO_MODES:
            errors.append(f"audio_mode={self.audio_mode!r} must be one of {VALID_AUDIO_MODES}")
        if self.continuity_mode not in VALID_CONTINUITY_MODES:
            errors.append(f"continuity_mode={self.continuity_mode!r} must be one of {VALID_CONTINUITY_MODES}")
        if self.inference_mode not in VALID_INFERENCE_MODES:
            errors.append(f"inference_mode={self.inference_mode!r} must be one of {VALID_INFERENCE_MODES}")
        if self.ref_image_size not in ("max", "match"):
            errors.append("ref_image_size must be 'max' or 'match'")
        if self.width % 32 or self.height % 32 or self.width * self.height > H3_MAX_AREA:
            errors.append(f"{self.width}x{self.height}: width/height must be multiples of 32, area <= 1344x768")
        if not 0 <= self.tail_guard_sec <= 1.0:
            errors.append("tail_guard_sec should be between 0 and 1 second")
        if not self.shots:
            errors.append("storyboard has no shots")
        longest = H3_MAX_SECONDS - self.tail_guard_sec
        seen = set()
        for s in self.shots:
            if s.shot_id in seen:
                errors.append(f"duplicate shot id {s.shot_id!r}")
            seen.add(s.shot_id)
            if not s.visual_prompt.strip():
                errors.append(f"shot {s.shot_id}: empty visual description")
            if s.duration_sec <= 0:
                errors.append(f"shot {s.shot_id}: duration must be > 0")
            if s.duration_sec > longest + 1e-6:
                errors.append(
                    f"shot {s.shot_id}: {s.duration_sec:.2f}s is longer than one H3 clip allows "
                    f"({longest:.2f}s after the {self.tail_guard_sec}s tail guard) - split it into two rows")
            if s.continuity_override and s.continuity_override not in ("anchor", "chain_last_frame"):
                errors.append(f"shot {s.shot_id}: continuity_override must be 'anchor' or 'chain_last_frame'")
            if s.gen_mode and s.gen_mode not in GEN_MODE_ENGINE:
                errors.append(f"shot {s.shot_id}: gen_mode must be one of {sorted(GEN_MODE_ENGINE)}")
            if s.gen_mode == "first_last" and not s.last_frame:
                errors.append(f"shot {s.shot_id}: 首尾幀模式需要尾幀圖 (last_frame)")
            if s.beats and abs(sum(b.duration_sec for b in s.beats) - s.duration_sec) > 0.05:
                errors.append(f"shot {s.shot_id}: beats add up to {sum(b.duration_sec for b in s.beats):.2f}s "
                              f"but the segment keeps {s.duration_sec:.2f}s")
        for c in self.characters.values():
            if c.tier and c.tier not in CHARACTER_TIERS:
                errors.append(f"character {c.id}: tier must be one of {CHARACTER_TIERS}")
        if errors:
            raise ValueError("Storyboard problems:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------- helpers
def clean_dialogue(text: str) -> str:
    """Strip `<d>[Lang] ... </d>` wrappers and quotes, keep the spoken words."""
    if not text:
        return ""
    t = re.sub(r"<d>\s*(?:\[[A-Za-z\-]+\]\s*)?", "", text, flags=re.I)
    t = re.sub(r"</d>", "", t, flags=re.I)
    return t.strip().strip("「」\"'“” ").strip()


# kept for backwards compatibility with earlier imports
_clean_dialogue_tags = clean_dialogue


def _resolve_path(p: str, base_dirs: List[Path]) -> str:
    if not p:
        return ""
    cand = Path(p)
    if cand.is_absolute():
        return str(cand)
    for b in base_dirs:
        if b and (b / cand).exists():
            return str((b / cand).resolve())
    return p


FRAMING_EN = {
    "大特寫": "extreme close-up", "特寫": "close-up", "近景": "medium close-up", "中近景": "medium close-up",
    "中景": "medium shot", "全景": "full shot", "遠景": "extreme wide establishing shot",
    "大遠景": "extreme wide establishing shot",
}
_TIER_RANK = {"main": 0, "secondary": 1, "": 1, "npc": 2}


def framing_en(framing: str) -> str:
    f = (framing or "").strip()
    for zh in sorted(FRAMING_EN, key=len, reverse=True):
        if zh in f:
            return FRAMING_EN[zh]
    return f


def shot_gen_mode(project: StoryboardProject, shot: ShotSpec) -> str:
    if shot.gen_mode:
        return shot.gen_mode
    return "first_frame" if project.inference_mode == "first_frame" else "reference"


def shot_engine(project: StoryboardProject, shot: ShotSpec) -> str:
    """Which H3 model a shot needs: 'reference' (Ref2VA) or 'first_frame' (FL2VA)."""
    return GEN_MODE_ENGINE[shot_gen_mode(project, shot)]


def shot_picture_plan(project: StoryboardProject, shot: ShotSpec,
                      base_dirs: Optional[List[Path]] = None) -> List[PictureRef]:
    """Ordered pictures for a shot: <Picture 1> scene/start frame, then character sheets (Ref2VA only).

    With more characters than picture slots (9 including the scene), main characters keep their pictures
    first, then secondary, then NPCs (who fall back to words); main characters' extra views (turnaround,
    face close-up) only take slots that are left. Missing files are dropped so the prompt never mentions
    a picture the graph does not receive.
    """
    base_dirs = [Path(b) for b in (base_dirs or []) if b]
    plan: List[PictureRef] = []
    scene = project.scenes.get(shot.scene_id) or project.scenes.get("default")

    def exists(p: str) -> bool:
        return bool(p) and Path(p).is_file()

    if shot.reference_images:
        char_by_file = {Path(c.image_path).name: c for c in project.characters.values() if c.image_path}
        for raw in shot.reference_images:
            p = _resolve_path(raw, base_dirs)
            if not exists(p):
                continue
            ch = char_by_file.get(Path(p).name)
            if ch:
                plan.append(PictureRef(p, "character", _character_label(ch), ch.id))
            elif not plan:
                plan.append(PictureRef(p, "scene", "the scene, people and composition"))
            else:
                plan.append(PictureRef(p, "extra", "an additional visual reference"))
    else:
        if scene and exists(_resolve_path(scene.image_path, base_dirs)):
            desc = scene.description or "the scene"
            plan.append(PictureRef(_resolve_path(scene.image_path, base_dirs), "scene",
                                   f"the scene and composition ({desc})"))
        slots = MAX_REFERENCE_IMAGES - len(plan)
        order = {cid: i for i, cid in enumerate(shot.characters)}
        cast = [project.characters[c] for c in shot.characters if c in project.characters]
        with_pic = [ch for ch in cast if exists(_resolve_path(ch.image_path, base_dirs))]
        chosen = sorted(with_pic, key=lambda ch: (_TIER_RANK.get(ch.tier, 1), order[ch.id]))[:slots]
        chosen.sort(key=lambda ch: order[ch.id])
        for ch in chosen:
            plan.append(PictureRef(_resolve_path(ch.image_path, base_dirs), "character", _character_label(ch), ch.id))
        for ch in chosen:
            if ch.tier != "main":
                continue
            for raw in ch.extra_images:
                p = _resolve_path(raw, base_dirs)
                if exists(p) and len(plan) < MAX_REFERENCE_IMAGES:
                    plan.append(PictureRef(p, "extra", f"another view of {ch.name}", ch.id))

    if shot_engine(project, shot) == "first_frame":
        plan = plan[:1]
    return plan[:MAX_REFERENCE_IMAGES]


def _character_label(ch: CharacterSpec) -> str:
    bits = [ch.name]
    if ch.description:
        bits.append(ch.description)
    if ch.screen_position:
        bits.append(f"always in the same place ({ch.screen_position})")
    return ", ".join(bits)


def _picture_of(pictures: List[PictureRef], char_id: str, label: str = "") -> int:
    for n, pic in enumerate(pictures, start=1):
        if n > 1 and pic.role == "character" and (pic.char_id == char_id or (label and pic.label == label)):
            return n
    return 0


def _speaker_phrase(project: StoryboardProject, speaker: str, pictures: List[PictureRef]) -> str:
    """'The person in <Picture 2> (examinee), voice: a calm young male voice,' for a known character.

    Tying the line to the speaker's picture keeps the lips moving on the right face, and repeating the
    same voice description in every clip keeps the voice from changing between clips.
    """
    ch = project.characters.get(speaker)
    if ch is None:
        return speaker
    n = _picture_of(pictures, ch.id, _character_label(ch))
    who = f"The person in <Picture {n}> ({ch.name})" if n else ch.name
    return f"{who}, voice: {ch.voice}," if ch.voice else who


def _music_from(sound: str) -> str:
    """'環境音：病房；配樂：輕柔鋼琴' -> '輕柔鋼琴' ('無' / 'none' -> '')."""
    m = re.search(r"(?:配樂|音樂|music)\s*[：:]?\s*([^；;。\n]+)", sound or "", flags=re.I)
    if not m:
        return ""
    music = m.group(1).strip()
    return "" if re.fullmatch(r"(無|none|n/?a|沒有|不需要)", music, flags=re.I) else music


def _ambient(sound: str) -> str:
    """Sound effects only; the 配樂 part goes to non_diegetic_music."""
    s = re.sub(r"[；;，,]?\s*(?:配樂|音樂|music)\s*[：:]?[^；;。\n]*", "", sound or "", flags=re.I)
    return s.strip(" ；;，,。.")


def _speech(project: StoryboardProject, speaker: str, spoken: str, start: float, finish: float,
            pictures: List[PictureRef]) -> str:
    if speaker:
        return (f"{_speaker_phrase(project, speaker, pictures)} says, starting at {start:.1f} seconds and "
                f"finishing by {finish:.1f} seconds: <d>[Chinese] {spoken}</d> Nobody else speaks.")
    return (f"{project.narrator}, starting at {start:.1f} seconds and finishing by {finish:.1f} seconds: "
            f"<d>[Chinese] {spoken}</d> The on-screen people do not speak; their lips stay relaxed.")


def _voices_line(project: StoryboardProject, speakers: List[str]) -> str:
    named = [project.characters[s].name for s in dict.fromkeys(speakers) if s in project.characters]
    if not speakers:
        return " No voices."
    if len(named) == 1 and len(set(speakers)) == 1:
        return f" Only {named[0]} speaks, with the same voice as in every other clip. No other voices."
    if named:
        who = ", ".join(named) + (" and the narrator" if "" in speakers else "")
        return f" Only {who} speak, each with the same voice as in every other clip. No other voices."
    return " The same single narrator voice in every clip. No other voices."


def build_h3_prompt(project: StoryboardProject, shot: ShotSpec, pictures: List[PictureRef], *,
                    opening: Optional[bool] = None, end_picture: int = 0) -> str:
    """Compose the full H3 prompt for one clip. `pictures` must be the exact list fed to the graph.

    opening      True: <Picture 1> is the clip's first frame ("fully referenced at 0.00 s"); False: it is only
                 a reference for the place (the first frame is pinned by a keyframe, or the clip opens on a
                 closer framing than the scene picture). None decides from the first beat's framing.
    end_picture  FL2VA first+last mode: the picture number of the last frame.

    A segment (shot with beats) is written as consecutive [Shot k] cuts, each with its own time window,
    framing, lighting, action, line and sound - all generated as one clip. The fixed assets (setting,
    people) come first so every clip starts from the same description.
    """
    seconds = project.generation_seconds(shot)
    lines: List[str] = []
    if opening is None:
        first = framing_en(shot.start_framing)
        opening = (not first or "wide" in first or "full shot" in first
                   or bool(pictures and pictures[0].role == "chain"))

    if pictures:
        if opening:
            lines.append("For the target video, at 0.00 seconds into the target video, "
                         "<Picture 1> (from [Shot 1]) is fully referenced.")
        else:
            lines.append("<Picture 1> shows the setting (room layout, furniture and lighting); use it as a "
                         "reference for the place, not as the opening frame.")
        if end_picture:
            last_beat = len(shot.beats) or 1
            lines.append(f"At {seconds - 0.05:.2f} seconds into the target video, <Picture {end_picture}> "
                         f"(from [Shot {last_beat}]) is fully referenced.")
        refs = []
        for n, pic in enumerate(pictures[1:], start=2):
            if n == end_picture:
                continue
            same = ""
            if pic.role == "extra" and pic.char_id:
                k = _picture_of(pictures, pic.char_id)
                same = f" (the same person as <Picture {k}>)" if k else ""
            refs.append(f"<Picture {n}> shows {pic.label}{same}; keep this person's face, hair and clothing identical.")
        if refs:
            lines.append(" ".join(refs))
        lines.append("")

    body: List[str] = []
    scene = project.scenes.get(shot.scene_id) or project.scenes.get("default")
    if scene and scene.description:
        body.append(f"Setting: {scene.description}.")
    pictured = {p.char_id for p in pictures if p.char_id}
    for cid in shot.characters:
        ch = project.characters.get(cid)
        if ch and cid not in pictured:            # no picture of their own: describe them in words
            body.append(f"{_character_label(ch)}.")
    body.append(project.style_prefix.strip())

    native = project.audio_mode == "native_h3_audio"
    speakers: List[str] = []
    action: List[str] = []
    if shot.beats:
        t = 0.0
        for k, b in enumerate(shot.beats, start=1):
            head = f"[Shot {k}] {t:.1f}-{t + b.duration_sec:.1f} seconds"
            if b.framing:
                head += f", {framing_en(b.framing)}"
            parts = [head + ":"]
            if b.lighting:
                parts.append(f"Lighting and colour: {b.lighting.strip().rstrip('。.')}.")
            if b.visual:
                parts.append(b.visual.strip().rstrip("。.") + ".")
            if b.camera:
                parts.append(f"Camera: {b.camera.strip().rstrip('。.')}.")
            spoken = clean_dialogue(b.dialogue)
            if native and spoken:
                finish = t + b.duration_sec - 0.3
                if k == len(shot.beats):         # leave the silent last second
                    finish = min(finish, seconds - 1.0)
                parts.append(_speech(project, b.speaker, spoken, t + 0.3, max(t + 0.8, finish), pictures))
                speakers.append(b.speaker)
            elif native:
                parts.append("No one speaks in this shot.")
            ambient = _ambient(b.sound)
            if native and ambient:
                parts.append(f"Sound: {ambient}.")
            action.append(" ".join(parts))
            t += b.duration_sec
        if not native:
            action.append("Nobody speaks: no dialogue, no narration, no singing; lips stay relaxed.")
        elif speakers:
            action.append("Hold the final pose silently for the last second.")
    else:
        action.append(shot.visual_prompt.strip().rstrip("。.") + ".")
        if shot.camera_motion:
            action.append(f"Camera: {shot.camera_motion.strip().rstrip('。.')}.")
        spoken = clean_dialogue(shot.dialogue)
        if native and spoken:
            action.append(_speech(project, shot.speaker, spoken, 0.4, max(1.5, seconds - 1.0), pictures))
            action.append("Hold the final pose silently for the last second.")
            speakers.append(shot.speaker)
        else:
            action.append("Nobody speaks: no dialogue, no narration, no singing; lips stay relaxed.")

    tail: List[str] = [project.continuity_rules] if pictures else []
    tail.append(project.negative_rules)
    lines.append("integrated_multimodal_description: " + ("" if shot.beats else "[Shot 1] ")
                 + " ".join(x for x in body + action + tail if x))
    lines.append("")
    if native:
        sound = project.overall_soundscape.strip() + _voices_line(project, speakers)
    else:
        sound = "Quiet room tone only. No speech, no voices, no music."
    lines.append("overall_soundscape: " + sound)
    lines.append("")
    music = ""
    if native and shot.beats:
        music = next((m for m in (_music_from(b.sound) for b in shot.beats) if m), "")
    lines.append("non_diegetic_music: " + (music or "N/A"))
    return "\n".join(lines)


def check_picture_tags(prompt: str, picture_count: int) -> None:
    used = {int(m.group(1)) for m in re.finditer(r"<Picture\s+(\d+)>", prompt)}
    bad = sorted(n for n in used if n < 1 or n > picture_count)
    if bad:
        raise ValueError(f"prompt mentions {['<Picture %d>' % n for n in bad]} but only {picture_count} "
                         f"picture(s) are attached")


def preview_shot(project: StoryboardProject, shot: ShotSpec, base_dirs: Optional[List[Path]] = None) -> str:
    plan = shot_picture_plan(project, shot, base_dirs)
    prompt = build_h3_prompt(project, shot, plan)
    check_picture_tags(prompt, len(plan))
    return prompt


# ---------------------------------------------------------------------- parsing
def _parse_time_cell(raw: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """'0-8s', '00:08-00:18', '8s', '8' -> (start, end, duration); unknown parts are None."""
    t = (raw or "").strip()
    if not t:
        return None, None, None
    m = re.search(r"(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[-~～至到]\s*(\d{1,2}):(\d{2}(?:\.\d+)?)", t)
    if m:
        s = int(m.group(1)) * 60 + float(m.group(2))
        e = int(m.group(3)) * 60 + float(m.group(4))
        return s, e, e - s
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~～至到]\s*(\d+(?:\.\d+)?)", t)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        return s, e, e - s
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        return None, None, float(m.group(1))
    return None, None, None


def _split_list(cell: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;，；、\s]+", cell or "") if x.strip()]


def _new_project(title: str, audio_mode: Optional[str], continuity_mode: Optional[str],
                 default_reference_image: str) -> StoryboardProject:
    p = StoryboardProject(title=title)
    if audio_mode:
        p.audio_mode = audio_mode
    if continuity_mode:
        p.continuity_mode = continuity_mode
    if default_reference_image:
        p.scenes["default"] = SceneAnchorSpec("default", "", default_reference_image)
    return p


def _assign_seeds(p: StoryboardProject) -> None:
    for s in p.shots:
        if not s.seed:
            s.seed = (p.base_seed + s.index * 1009) % (2 ** 63)


def load_storyboard_from_markdown(md_text: str, *, title: str = "AI_Movie",
                                  audio_mode: Optional[str] = None, continuity_mode: Optional[str] = None,
                                  default_reference_image: str = "", **_: Any) -> StoryboardProject:
    """Parse the first Markdown table that has a visual/description column."""
    p = _new_project(title, audio_mode, continuity_mode, default_reference_image)
    headers: List[str] = []
    rows: List[List[str]] = []
    for ln in md_text.splitlines():
        ln = ln.strip()
        if not (ln.startswith("|") and ln.endswith("|")):
            if headers and rows:
                break
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
            continue
        if not headers:
            headers = [c.lower() for c in cells]
        else:
            rows.append(cells)
    if not headers:
        raise ValueError("no Markdown table found; expected | 鏡號 | 秒數 | 畫面 | 旁白 | ... |")

    def col(*keys: str) -> int:
        for i, h in enumerate(headers):
            if any(k in h for k in keys):
                return i
        return -1

    c_id = col("鏡號", "鏡頭", "shot", "編號", "id")
    c_time = col("秒數", "時間", "時長", "duration", "time", "sec")
    c_scene = col("場景", "scene")
    c_char = col("角色", "人物", "character", "cast")
    c_vis = col("畫面", "視覺", "動作", "描述", "visual", "prompt", "action")
    c_cam = col("運鏡", "鏡位", "camera")
    c_dlg = col("旁白", "台詞", "對白", "語音", "dialogue", "narration", "voice")
    c_spk = col("說話者", "講者", "speaker")
    c_sub = col("字幕", "subtitle", "caption")
    c_ref = col("參考圖", "圖片", "ref", "image")
    c_wav = col("音檔", "wav")
    if c_vis < 0:
        raise ValueError(f"Markdown table needs a 畫面/visual column; found headers {headers}")

    for n, row in enumerate(rows, start=1):
        get = (lambda c: row[c].strip() if 0 <= c < len(row) else "")
        sid = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", get(c_id)).strip("_") or f"shot_{n:02d}"
        _, _, dur = _parse_time_cell(get(c_time))
        dialogue = clean_dialogue(get(c_dlg))
        p.shots.append(ShotSpec(
            shot_id=sid, index=n, title=get(c_id) or sid,
            scene_id=get(c_scene) or "default",
            duration_sec=float(dur) if dur else 8.0,
            visual_prompt=get(c_vis),
            camera_motion=get(c_cam),
            dialogue=dialogue,
            speaker=get(c_spk),
            subtitle_text=get(c_sub) or dialogue,
            characters=_split_list(get(c_char)),
            reference_images=_split_list(get(c_ref)),
            narration_wav=get(c_wav),
        ))
    _assign_seeds(p)
    return p


def load_storyboard_from_json(data: Any, *, title: str = "AI_Movie", audio_mode: Optional[str] = None,
                              continuity_mode: Optional[str] = None, default_reference_image: str = "",
                              base_dir: Optional[Path] = None, **_: Any) -> StoryboardProject:
    """Accept a full project dict, or a bare list of shots (scenes.json / shotlist.json)."""
    meta: Dict[str, Any] = data if isinstance(data, dict) else {}
    raw_shots = data if isinstance(data, list) else (
        meta.get("shots") or meta.get("shotlist") or (meta.get("scenes") if isinstance(meta.get("scenes"), list) else []))
    p = _new_project(meta.get("title", title), audio_mode or meta.get("audio_mode"),
                     continuity_mode or meta.get("continuity_mode"), default_reference_image)
    if meta.get("aspect_ratio"):
        p.set_aspect(meta["aspect_ratio"])
    for key in ("inference_mode", "ref_image_size", "master_audio_path", "style_prefix", "continuity_rules",
                "negative_rules", "narrator", "overall_soundscape"):
        if meta.get(key):
            setattr(p, key, meta[key])
    for key, cast in (("tail_guard_sec", float), ("base_seed", int), ("sample_rate", int)):
        if key in meta:
            setattr(p, key, cast(meta[key]))

    if meta.get("preset"):
        p.preset = str(meta["preset"])
    for cid, c in (meta.get("characters") or {}).items():
        c = c if isinstance(c, dict) else {"description": str(c)}
        p.characters[cid] = CharacterSpec(cid, c.get("name", cid), c.get("description", ""),
                                          c.get("image_path", ""), c.get("screen_position", ""),
                                          list(c.get("aliases") or []), c.get("voice", ""),
                                          bool(c.get("lead", False)), str(c.get("tier", "")),
                                          list(c.get("extra_images") or []))
    if isinstance(meta.get("scenes"), dict):
        for sid, s in meta["scenes"].items():
            s = s if isinstance(s, dict) else {"description": str(s)}
            p.scenes[sid] = SceneAnchorSpec(sid, s.get("description", ""),
                                            s.get("image_path", "") or default_reference_image,
                                            s.get("name", ""), list(s.get("aliases") or []),
                                            list(s.get("cast") or []))
    for cue in meta.get("subtitle_cues") or []:
        p.subtitle_cues.append((float(cue[0]), float(cue[1]), str(cue[2])))

    for n, it in enumerate(raw_shots, start=1):
        visual = it.get("visual_prompt") or it.get("prompt") or it.get("visual") or it.get("description") or ""
        if not visual and it.get("prompt_file") and base_dir and (Path(base_dir) / it["prompt_file"]).is_file():
            visual = (Path(base_dir) / it["prompt_file"]).read_text(encoding="utf-8").strip()
        dialogue = it.get("dialogue") or it.get("narration") or ""
        if not dialogue:
            m = re.search(r"<d>\s*(?:\[[A-Za-z\-]+\]\s*)?(.*?)</d>", visual, flags=re.S | re.I)
            if m:
                dialogue = m.group(1)
        dialogue = clean_dialogue(dialogue)
        refs = it.get("reference_images") or ([it["reference_image"]] if it.get("reference_image") else [])
        p.shots.append(ShotSpec(
            shot_id=str(it.get("shot_id") or it.get("id") or f"shot_{n:02d}"),
            index=int(it.get("index", n)),
            title=str(it.get("title", "")),
            scene_id=str(it.get("scene_id") or it.get("scene") or "default"),
            duration_sec=float(it.get("duration_sec") or it.get("duration") or it.get("seconds") or 8.0),
            visual_prompt=visual,
            camera_motion=str(it.get("camera_motion") or it.get("camera") or ""),
            dialogue=dialogue,
            speaker=str(it.get("speaker", "")),
            subtitle_text=str(it.get("subtitle_text") or it.get("subtitle") or dialogue),
            characters=list(it.get("characters") or []),
            reference_images=list(refs),
            narration_wav=str(it.get("narration_wav", "")),
            seed=int(it.get("seed", 0)),
            continuity_override=str(it.get("continuity_override", "")),
            notes=str(it.get("notes", "")),
            beats=[_beat_from_dict(b) for b in it.get("beats") or []],
            gen_mode=str(it.get("gen_mode", "")),
            first_frame=str(it.get("first_frame", "")),
            last_frame=str(it.get("last_frame", "")),
        ))
    _assign_seeds(p)
    return p


def _beat_from_dict(b: Dict[str, Any]) -> BeatSpec:
    return BeatSpec(beat_id=str(b.get("beat_id", "")), duration_sec=float(b.get("duration_sec", 3.0)),
                    framing=str(b.get("framing", "")), lighting=str(b.get("lighting", "")),
                    visual=str(b.get("visual", "")), dialogue=str(b.get("dialogue", "")),
                    speaker=str(b.get("speaker", "")), sound=str(b.get("sound", "")),
                    camera=str(b.get("camera", "")), missing_fields=list(b.get("missing_fields") or []))


def save_project_json(project: StoryboardProject, path: str | Path) -> Path:
    """Snapshot the whole project (bible, presets, seeds) so a later Colab session reloads exactly the same plan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_storyboard_from_mvkit_project(project_path: str | Path, *, audio_mode: Optional[str] = None,
                                       continuity_mode: Optional[str] = None, **_: Any) -> StoryboardProject:
    """Adapt an autoip-mv-maker project.py: every shot id listed in SECTIONS becomes one H3 clip.

    The song is the single master track; lyric lines become absolute-timed subtitles.
    """
    path = Path(project_path)
    pdir = path if path.is_dir() else path.parent
    spec = importlib.util.spec_from_file_location("mv_project", pdir / "project.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no project.py in {pdir}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    p = StoryboardProject(title=getattr(m, "NAME", pdir.name))
    p.audio_mode = audio_mode or "master_audio"
    p.continuity_mode = continuity_mode or "anchor"
    song = getattr(m, "SONG", "")
    p.master_audio_path = str(pdir / song) if song else ""
    if getattr(m, "STYLE", ""):
        p.style_prefix = m.STYLE
    prefix = getattr(m, "MOTION_PREFIX", "")

    for cid, c in getattr(m, "CHARS", {}).items():
        p.characters[cid] = CharacterSpec(cid, c.get("name", cid), c.get("desc", ""),
                                          str(pdir / c["file"]) if c.get("file") else "")

    catalog: Dict[str, Tuple[str, List[str]]] = {}
    ls_scene = {s[0]: s for s in getattr(m, "LS_SCENES", [])}
    for sid, _cap, prompt, motion in getattr(m, "BROLL", []):
        catalog[sid] = (f"{prompt}. {motion}", [])
    for sid, chars, _cap, prompt, motion in getattr(m, "BAND", []):
        chars = chars if isinstance(chars, list) else [chars]
        catalog[sid] = (f"{prompt}. {motion}".strip(". "), chars)
    for sid, scene_id, _a0, _dur, _voice in getattr(m, "LIPSYNC", []):
        sc = ls_scene.get(scene_id)
        if sc:
            catalog[sid] = (f"{sc[2]}. {sc[3]}", [sc[1]])

    n = 0
    longest = H3_MAX_SECONDS - p.tail_guard_sec
    for name, (a, b), _typo, lines, shot_ids in getattr(m, "SECTIONS", []):
        ids = [s for s in shot_ids if s in catalog] or []
        if not ids:
            continue
        per = (b - a) / len(ids)
        pieces = max(1, math.ceil(per / longest))
        for sid in ids:
            visual, chars = catalog[sid]
            for k in range(pieces):
                n += 1
                p.shots.append(ShotSpec(
                    shot_id=f"{sid}_{k + 1}" if pieces > 1 else sid, index=n, title=name,
                    scene_id=sid, duration_sec=round(per / pieces, 3),
                    visual_prompt=f"{prefix}{visual}", characters=list(chars)))
        for i, (t, text, _voice) in enumerate(lines):
            end = lines[i + 1][0] if i + 1 < len(lines) else b
            p.subtitle_cues.append((float(t), float(end) - 0.1, text))
    _assign_seeds(p)
    return p


def load_storyboard(source: str | Path, *, audio_mode: Optional[str] = None,
                    continuity_mode: Optional[str] = None, default_reference_image: str = "") -> StoryboardProject:
    """Auto-detect `.md`, `.json`, `project.py` or a project folder."""
    src = Path(source)
    if src.is_dir() or src.name == "project.py":
        return load_storyboard_from_mvkit_project(src, audio_mode=audio_mode, continuity_mode=continuity_mode)
    if not src.is_file():
        raise FileNotFoundError(f"storyboard not found: {src}")
    if src.suffix.lower() == ".json":
        return load_storyboard_from_json(json.loads(src.read_text(encoding="utf-8")), title=src.stem,
                                         audio_mode=audio_mode, continuity_mode=continuity_mode,
                                         default_reference_image=default_reference_image, base_dir=src.parent)
    return load_storyboard_from_markdown(src.read_text(encoding="utf-8"), title=src.stem, audio_mode=audio_mode,
                                         continuity_mode=continuity_mode,
                                         default_reference_image=default_reference_image)

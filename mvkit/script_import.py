"""Upload a script (teaching / recruitment / OSCE demo) and turn it into a storyboard.

Segment scripts (`段落 1（0–10 秒）` + `分鏡 1-1｜全景｜3 秒` with the four modules) are handled by
`segment_script.py`; the older free-form layout below still works.

Older free-form layout: upload a free-form script (teaching / recruitment / OSCE demo) and turn it into a storyboard.

Accepted files:
  * `.txt` / `.md` written as numbered shots, e.g.

        第一幕：準備與溝通
        分鏡 1：準備用物與手部衛生
            場景： 病床旁的操作台
            畫面： 考生執行洗手，治療盤上備有鼻胃管、潤滑劑……
            對話/口白： （口述）「操作前，我會先洗手，並準備好所有需要的物品。」
            動作與評分重點： 執行技術前須洗手。

    (`Shot 1:` / `鏡頭 1` / `#1` headings and English field names work too)
  * `.docx` (Word) with the same layout - read without extra packages
  * Google AI Studio exports (`{"veoPrompt": {"prompt": "..."}}`)
  * `.md` storyboard tables, storyboard `.json` and `project.py` go to `script_loader.load_storyboard` unchanged

Durations are estimated from the spoken line (Taiwan Mandarin ~4 characters per second plus breathing room)
unless the script gives `秒數` / `時間`. A line too long for one H3 clip is split at sentence ends into
`03a`, `03b`, ... which share the picture description and chain from each other.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

from .script_loader import (
    H3_MAX_SECONDS,
    ShotSpec,
    StoryboardProject,
    _assign_seeds,
    _new_project,
    _parse_time_cell,
    clean_dialogue,
    load_storyboard,
)

SPEECH_CHARS_PER_SEC = 4.0           # measured Taiwan Mandarin narration
SPEECH_PAD_SEC = 1.6                 # 0.4 s lead-in + ~1 s silent hold at the end (see build_h3_prompt)
MIN_SHOT_SEC = 5.0
SCRIPT_EXTS = (".txt", ".md", ".markdown", ".docx", ".json")

_SHOT_HEAD = re.compile(
    r"^\s*(?:#+\s*)?(?:分鏡|鏡頭|鏡號|镜头|分镜|Shot|SHOT|shot|Cut|CUT)\s*[#＃]?\s*([0-9０-９]+[A-Za-z]?)\s*[：:．.、\-—]?\s*(.*)$")
_ACT_HEAD = re.compile(r"^\s*(?:#+\s*)?(第[一二三四五六七八九十百0-9]+[幕場章部分]|Act\s*\d+|ACT\s*\d+)\s*[：:．.、\-—]?\s*(.*)$")

# field label -> ShotDraft attribute (longest labels first so "動作與評分重點" beats "動作")
_FIELD_LABELS = [
    ("動作與評分重點", "scoring"), ("評分重點", "scoring"), ("評分項目", "scoring"), ("重點", "scoring"),
    ("scoring", "scoring"), ("checklist", "scoring"),
    ("對話/口白", "dialogue"), ("對話／口白", "dialogue"), ("對白", "dialogue"), ("對話", "dialogue"),
    ("口白", "dialogue"), ("台詞", "dialogue"), ("旁白", "dialogue"), ("dialogue", "dialogue"),
    ("narration", "dialogue"), ("voice", "dialogue"),
    ("畫面描述", "visual"), ("畫面", "visual"), ("動作", "visual"), ("visual", "visual"), ("action", "visual"),
    ("場景", "scene"), ("地點", "scene"), ("scene", "scene"), ("location", "scene"),
    ("運鏡", "camera"), ("鏡位", "camera"), ("camera", "camera"),
    ("角色", "characters"), ("人物", "characters"), ("出場", "characters"), ("characters", "characters"),
    ("cast", "characters"),
    ("說話者", "speaker"), ("講者", "speaker"), ("speaker", "speaker"),
    ("秒數", "time"), ("時間", "time"), ("時長", "time"), ("duration", "time"),
    ("字幕", "subtitle"), ("subtitle", "subtitle"),
    ("參考圖", "refs"), ("reference", "refs"),
]
_FIELD_RE = re.compile(
    r"^\s*[-*•·]?\s*(" + "|".join(re.escape(k) for k, _ in _FIELD_LABELS) + r")\s*[：:]\s*(.*)$", re.I)
_FIELD_MAP = {k.lower(): v for k, v in _FIELD_LABELS}

# （口述）「……」 / （對病人）「……」 / 考生：「……」
_QUOTE_RE = re.compile(r"[「『“\"]([^」』”\"]+)[」』”\"]")
_CUE_RE = re.compile(r"[（(]\s*([^）)]{1,12})\s*[）)]")


@dataclass
class ShotDraft:
    number: str
    title: str = ""
    act: str = ""
    scene: str = ""
    visual: str = ""
    dialogue: str = ""
    speaker_cue: str = ""             # "口述", "對病人", "病人" ... resolved by alignment.py
    scoring: str = ""
    camera: str = ""
    characters: str = ""
    speaker: str = ""
    time: str = ""
    subtitle: str = ""
    refs: str = ""
    extra: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------- reading files
def read_docx_text(path: str | Path) -> str:
    """Paragraph text of a .docx (tables flattened row by row as `| a | b |`), no python-docx needed."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    body = root.find("w:body", ns)
    lines: List[str] = []
    for el in list(body) if body is not None else []:
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append("".join(t.text or "" for t in el.iter(f"{{{ns['w']}}}t")))
        elif tag == "tbl":
            for tr in el.iter(f"{{{ns['w']}}}tr"):
                cells = ["".join(t.text or "" for t in tc.iter(f"{{{ns['w']}}}t")).strip()
                         for tc in tr.iter(f"{{{ns['w']}}}tc")]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_script_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_text(path)
    raw = path.read_text(encoding="utf-8-sig")
    if suffix == ".json" or raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        prompt = (data.get("veoPrompt") or {}).get("prompt") if isinstance(data, dict) else None
        if prompt:
            return prompt
    return raw


def is_storyboard_table(text: str) -> bool:
    """True for the original Markdown storyboard table (handled by script_loader)."""
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("|") and ln.endswith("|") and re.search(r"畫面|visual|prompt|描述", ln, re.I):
            return True
    return False


# ---------------------------------------------------------------------- parsing
def _normalise_digits(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def parse_freeform_script(text: str) -> List[ShotDraft]:
    """Split a numbered script into shot drafts; lines without a label continue the previous field."""
    drafts: List[ShotDraft] = []
    act = ""
    cur: Optional[ShotDraft] = None
    last_field = ""
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip().replace("　", " ")
        if not line:
            continue
        m = _SHOT_HEAD.match(line)
        if m and not _FIELD_RE.match(line):
            cur = ShotDraft(number=_normalise_digits(m.group(1)), title=m.group(2).strip(), act=act)
            drafts.append(cur)
            last_field = ""
            continue
        m = _ACT_HEAD.match(line)
        if m:
            act = (m.group(1) + (" " + m.group(2) if m.group(2) else "")).strip()
            continue
        if cur is None:
            continue                  # title / preamble before the first shot
        m = _FIELD_RE.match(line)
        if m:
            last_field = _FIELD_MAP[m.group(1).lower()]
            value = m.group(2).strip()
            setattr(cur, last_field, (getattr(cur, last_field) + " " + value).strip())
        elif last_field:
            setattr(cur, last_field, (getattr(cur, last_field) + " " + line).strip())
        else:
            cur.extra.append(line)
    for d in drafts:
        d.dialogue, d.speaker_cue = split_dialogue(d.dialogue)
        d.visual = _tidy(d.visual or " ".join(d.extra))
        d.scene, d.scoring, d.title = _tidy(d.scene), _tidy(d.scoring), _tidy(d.title)
    return drafts


def _tidy(s: str) -> str:
    s = re.sub(r"\s+([，。、；：！？）」])", r"\1", s or "")     # "洗手 。" -> "洗手。"
    return re.sub(r"\s{2,}", " ", s).strip()


def split_dialogue(raw: str) -> tuple:
    """'（對病人）「林小姐您好……」' -> ('林小姐您好……', '對病人'). Unquoted text is kept as the line."""
    raw = (raw or "").strip()
    if not raw:
        return "", ""
    cue_m = _CUE_RE.search(raw)
    cue = cue_m.group(1).strip() if cue_m else ""
    quotes = _QUOTE_RE.findall(raw)
    if quotes:
        line = "".join(q.strip() for q in quotes)
    else:
        line = _CUE_RE.sub("", raw)
        m = re.match(r"^\s*([^：:「]{1,8})[：:]\s*(.+)$", line)    # 考生：好，我們開始
        if m:
            cue = cue or m.group(1).strip()
            line = m.group(2)
    return _tidy(clean_dialogue(line)), cue


def estimate_seconds(dialogue: str, *, minimum: float = MIN_SHOT_SEC) -> float:
    spoken = len(re.sub(r"[\s，。、；：！？,.!?「」『』（）()…—\-]", "", dialogue or ""))
    sec = spoken / SPEECH_CHARS_PER_SEC + SPEECH_PAD_SEC if spoken else minimum + 1.0
    return round(max(minimum, sec) * 2) / 2


def split_long_line(dialogue: str, max_sec: float) -> List[str]:
    """Cut a line at sentence ends so every piece fits one clip."""
    pieces = [p for p in re.split(r"(?<=[。！？!?；;])", dialogue) if p.strip()]
    out: List[str] = []
    buf = ""
    for p in pieces:
        if buf and estimate_seconds(buf + p) > max_sec:
            out.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        out.append(buf)
    final: List[str] = []
    for chunk in out:                # one sentence still too long: cut at commas
        while estimate_seconds(chunk) > max_sec:
            limit = int((max_sec - SPEECH_PAD_SEC) * SPEECH_CHARS_PER_SEC)
            cut = max((chunk.rfind(c, 0, limit) for c in "，,、"), default=-1)
            cut = cut + 1 if cut > 0 else limit
            final.append(chunk[:cut])
            chunk = chunk[cut:]
        if chunk.strip():
            final.append(chunk)
    return [c.strip() for c in final]


def drafts_to_project(drafts: List[ShotDraft], *, title: str = "AI_Movie", audio_mode: Optional[str] = None,
                      continuity_mode: Optional[str] = None, default_reference_image: str = "",
                      tail_guard_sec: float = 0.35) -> StoryboardProject:
    if not drafts:
        raise ValueError("no shots found. Number each shot like `分鏡 1：標題` / `Shot 1:` and label the lines "
                         "`場景：` `畫面：` `對話/口白：`")
    p = _new_project(title, audio_mode, continuity_mode, default_reference_image)
    p.tail_guard_sec = tail_guard_sec
    longest = H3_MAX_SECONDS - tail_guard_sec
    n = 0
    for d in drafts:
        _, _, given = _parse_time_cell(d.time)
        lines = [d.dialogue]
        if d.dialogue and (given or estimate_seconds(d.dialogue)) > longest:
            lines = split_long_line(d.dialogue, longest)
        base_id = re.sub(r"[^0-9A-Za-z_\-一-鿿]+", "_", f"{int(d.number) if d.number.isdigit() else d.number}"
                         ).strip("_")
        base_id = base_id.zfill(2) if base_id.isdigit() else base_id
        for k, line in enumerate(lines):
            n += 1
            sid = base_id if len(lines) == 1 else f"{base_id}{chr(ord('a') + k)}"
            dur = given if (given and len(lines) == 1) else estimate_seconds(line)
            visual = d.visual if k == 0 else f"Continuing the same action: {d.visual}"
            notes = "；".join(x for x in (d.act, d.scoring) if x)
            p.shots.append(ShotSpec(
                shot_id=sid, index=n, title=d.title or sid,
                scene_id=d.scene or "default",
                duration_sec=min(longest, float(dur)),
                visual_prompt=visual,
                camera_motion=d.camera,
                dialogue=line,
                speaker=d.speaker or d.speaker_cue,
                subtitle_text=(d.subtitle if len(lines) == 1 else "") or line,
                characters=[c for c in re.split(r"[,;，；、\s]+", d.characters) if c],
                reference_images=[r for r in re.split(r"[,;，；、\s]+", d.refs) if r],
                notes=notes,
                continuity_override="chain_last_frame" if k > 0 else "",
            ))
    _assign_seeds(p)
    return p


def import_script(source: str | Path, *, title: Optional[str] = None, audio_mode: Optional[str] = None,
                  continuity_mode: Optional[str] = None, default_reference_image: str = "") -> StoryboardProject:
    """One entry point for every uploaded script: free-form text/docx, storyboard table/json, or project.py."""
    src = Path(source)
    if src.is_dir() or src.name == "project.py":
        return load_storyboard(src, audio_mode=audio_mode, continuity_mode=continuity_mode)
    text = read_script_text(src)
    looks_like_storyboard_json = False
    if src.suffix.lower() == ".json":
        try:
            data = json.loads(src.read_text(encoding="utf-8-sig"))
            looks_like_storyboard_json = isinstance(data, list) or (
                isinstance(data, dict) and any(k in data for k in ("shots", "shotlist", "scenes")))
        except json.JSONDecodeError:
            pass
    if looks_like_storyboard_json or (src.suffix.lower() in (".md", ".markdown") and is_storyboard_table(text)):
        project = load_storyboard(src, audio_mode=audio_mode, continuity_mode=continuity_mode,
                                  default_reference_image=default_reference_image)
    else:
        project = import_script_text(text, title=title or src.stem, audio_mode=audio_mode,
                                     continuity_mode=continuity_mode,
                                     default_reference_image=default_reference_image)
    if title:
        project.title = title
    return project


def import_script_text(text: str, *, title: str = "AI_Movie", audio_mode: Optional[str] = None,
                       continuity_mode: Optional[str] = None, default_reference_image: str = "") -> StoryboardProject:
    """Pasted text: segment scripts (段落 1 …) -> segment parser, a storyboard table -> the table loader,
    anything else -> the free-form shot parser."""
    from .segment_script import is_segment_script, parse_segment_script, segments_to_project
    if is_segment_script(text):
        return segments_to_project(parse_segment_script(text), title=title, audio_mode=audio_mode,
                                   continuity_mode=continuity_mode, default_reference_image=default_reference_image)
    if is_storyboard_table(text) and not any(_SHOT_HEAD.match(ln) for ln in text.splitlines()):
        from .script_loader import load_storyboard_from_markdown
        return load_storyboard_from_markdown(text, title=title, audio_mode=audio_mode,
                                             continuity_mode=continuity_mode,
                                             default_reference_image=default_reference_image)
    return drafts_to_project(parse_freeform_script(text), title=title, audio_mode=audio_mode,
                             continuity_mode=continuity_mode, default_reference_image=default_reference_image)


def drafts_summary(drafts: List[ShotDraft]) -> List[Dict[str, str]]:
    return [{"number": d.number, "title": d.title, "scene": d.scene, "speaker_cue": d.speaker_cue,
             "dialogue": d.dialogue, "scoring": d.scoring} for d in drafts]

"""Segment scripts: 10 s / 15 s segments, each holding several 分鏡 that H3 renders as cuts inside ONE clip.

Written as bullet text (no tables), every 分鏡 with the four modules and a 景別:

    段落 1（0–10 秒｜參考圖）
    場景：病床旁
    分鏡 1-1｜全景｜3 秒
    - 色調打光：冷白日光燈，均勻無陰影
    - 畫面描述：考生走進技能教室，病人坐在床邊
    - 旁白對白：（對病人）「王先生您好，我是今天的醫師。」
    - 音效配樂：室內環境音；配樂：無
    分鏡 1-2｜中景｜3.5 秒
    ...

Segment lines (before its first 分鏡): 場景, 角色, 模式, 首幀圖, 尾幀圖, 色調打光 (default for its 分鏡).
Mode words in the header or `模式：` - 參考圖 (default) | 參考圖+首幀 / 首幀鎖定 | 首幀 | 首尾幀.

Rules (`check_segments`): `10s_3` = 10 s segments with at least 3 分鏡, `15s_6` = 15 s with at least 6;
a 15 s segment keeps 14.65 s because H3 clips end at 15 s and the last 0.35 s is trimmed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .script_loader import (
    H3_MAX_SECONDS,
    BeatSpec,
    ShotSpec,
    StoryboardProject,
    _assign_seeds,
    _new_project,
    clean_dialogue,
)

SEGMENT_RULES: Dict[str, Tuple[float, int]] = {"10s_3": (10.0, 3), "15s_6": (15.0, 6)}
FRAMINGS = ("大遠景", "大特寫", "中近景", "遠景", "全景", "中景", "近景", "特寫")
REQUIRED = {"framing": "景別", "lighting": "色調打光", "visual": "畫面描述", "dialogue": "旁白對白",
            "sound": "音效配樂"}

_SEG_HEAD = re.compile(r"^\s*(?:#+\s*)?(?:段落|Segment|SEGMENT|segment)\s*([0-9０-９]+)\s*(.*)$")
_BEAT_HEAD = re.compile(
    r"^\s*(?:#+\s*)?[-*•・]?\s*(?:分鏡|鏡頭|Shot|SHOT|shot)\s*([0-9０-９]+(?:\s*[-－–._]\s*[0-9０-９]+)?)\s*(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*•・]|\d+[.)、])\s*")

_LABELS = [
    # beat modules
    ("背景音效與配樂", "sound"), ("音效與配樂", "sound"), ("音效配樂", "sound"), ("音效", "sound_fx"),
    ("配樂", "music"), ("sound", "sound"), ("music", "music"),
    ("畫面色調打光", "lighting"), ("色調與打光", "lighting"), ("色調打光", "lighting"), ("畫面色調", "lighting"),
    ("色調", "lighting"), ("打光", "lighting"), ("光線", "lighting"), ("lighting", "lighting"),
    ("分鏡畫面描述", "visual"), ("畫面描述", "visual"), ("視頻提示詞", "visual"), ("影片提示詞", "visual"),
    ("提示詞", "visual"), ("畫面", "visual"), ("visual", "visual"), ("prompt", "visual"),
    ("旁白對白", "dialogue"), ("旁白/對白", "dialogue"), ("旁白／對白", "dialogue"), ("對白旁白", "dialogue"),
    ("旁白", "dialogue"), ("對白", "dialogue"), ("台詞", "dialogue"), ("對話", "dialogue"), ("口白", "dialogue"),
    ("dialogue", "dialogue"),
    ("景別", "framing"), ("framing", "framing"), ("運鏡", "camera"), ("camera", "camera"),
    ("秒數", "seconds"), ("時長", "seconds"),
    # segment lines
    ("場景", "scene"), ("地點", "scene"), ("scene", "scene"), ("角色", "characters"), ("人物", "characters"),
    ("生成模式", "mode"), ("模式", "mode"), ("mode", "mode"),
    ("首幀圖", "first_frame"), ("首幀", "first_frame"), ("first_frame", "first_frame"),
    ("尾幀圖", "last_frame"), ("尾幀", "last_frame"), ("last_frame", "last_frame"),
    ("標題", "title"),
]
_LABEL_RE = re.compile(r"^(" + "|".join(re.escape(k) for k, _ in _LABELS) + r")\s*[：:]\s*(.*)$", re.I)
_LABEL_MAP = {k.lower(): v for k, v in _LABELS}


def _digits(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def is_segment_script(text: str) -> bool:
    return any(_SEG_HEAD.match(ln) for ln in (text or "").splitlines())


def parse_mode(text: str) -> str:
    t = (text or "").lower()
    if "首尾幀" in t or "first_last" in t or "first-last" in t:
        return "first_last"
    if "首幀鎖定" in t or ("參考" in t and "首幀" in t) or "reference_keyframes" in t or "鎖定" in t:
        return "reference_keyframes"
    if "首幀" in t or "first_frame" in t:
        return "first_frame"
    if "參考" in t or "reference" in t:
        return "reference"
    return ""


def _seconds(text: str) -> Tuple[Optional[float], Optional[float]]:
    """('0–10 秒') -> (None, 10.0) duration; ('3 秒') -> (None, 3.0)."""
    t = _digits(text or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*[-–—~～至到]\s*(\d+(?:\.\d+)?)\s*(?:秒|s|sec)", t)
    if m:
        return float(m.group(1)), float(m.group(2)) - float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b|sec)", t)
    if m:
        return None, float(m.group(1))
    return None, None


def _framing(text: str) -> str:
    for f in FRAMINGS:
        if f in (text or ""):
            return f
    m = re.search(r"(extreme close-up|close-up|medium close-up|medium shot|full shot|wide shot)", text or "", re.I)
    return m.group(1) if m else ""


class _Draft:
    def __init__(self, number: str, header: str):
        self.number = number
        self.header = header
        self.fields: Dict[str, str] = {}
        self.beats: List[Dict[str, str]] = []


def parse_segment_script(text: str) -> List[_Draft]:
    segs: List[_Draft] = []
    cur: Optional[_Draft] = None
    target: Optional[Dict[str, str]] = None
    last = ""
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip().replace("　", " ")
        if not line:
            continue
        m = _SEG_HEAD.match(line)
        if m:
            cur = _Draft(_digits(m.group(1)), m.group(2).strip())
            segs.append(cur)
            target, last = cur.fields, ""
            continue
        if cur is None:
            continue
        m = _BEAT_HEAD.match(line)
        if m and not _LABEL_RE.match(_BULLET.sub("", line)):
            beat = {"_id": re.sub(r"\s+", "", _digits(m.group(1))), "_header": m.group(2)}
            cur.beats.append(beat)
            target, last = beat, ""
            continue
        body = _BULLET.sub("", line)
        m = _LABEL_RE.match(body)
        if m:
            last = _LABEL_MAP[m.group(1).lower()]
            target[last] = (target.get(last, "") + " " + m.group(2).strip()).strip()
        elif last:
            target[last] = (target.get(last, "") + " " + body).strip()
    return segs


def _beat_seconds(seg_fields: Dict[str, str], beats: List[Dict[str, str]], seg_dur: Optional[float]) -> List[float]:
    given = []
    for b in beats:
        _, d = _seconds(b.get("seconds", "") or b.get("_header", ""))
        given.append(d)
    if seg_dur is None:
        out = []
        for g, b in zip(given, beats):
            if g:
                out.append(g)
            else:
                chars = len(re.sub(r"[\s，。、；：！？,.!?「」（）()]", "", clean_dialogue(b.get("dialogue", ""))))
                out.append(round(max(2.5, chars / 4.0 + 0.6), 1))
        return out
    known = sum(g for g in given if g)
    unknown = [i for i, g in enumerate(given) if not g]
    rest = max(0.0, seg_dur - known)
    out = [g if g else (rest / len(unknown) if unknown else 0.0) for g in given]
    total = sum(out)
    if total > 0 and abs(total - seg_dur) > 0.05:
        out = [x * seg_dur / total for x in out]
    return out


_SAME = re.compile(r"^\s*(同上一鏡|同前一鏡|同上|同前|same as above)\s*[，,、；;:：]?\s*", re.I)


def _inherit(beat: Dict[str, str], prev: Dict[str, str]) -> Dict[str, str]:
    """`色調打光：同上，臉部受光均勻` -> the previous 分鏡's lighting + '，臉部受光均勻'."""
    out = dict(beat)
    for key in ("lighting", "sound", "visual", "camera", "framing"):
        v = out.get(key, "")
        m = _SAME.match(v)
        if m:
            rest = v[m.end():].strip()
            out[key] = "，".join(x for x in (prev.get(key, ""), rest) if x)
    return out


def segments_to_project(segs: List[_Draft], *, title: str = "AI_Movie", audio_mode: Optional[str] = None,
                        continuity_mode: Optional[str] = None, default_reference_image: str = "",
                        tail_guard_sec: float = 0.35) -> StoryboardProject:
    from .script_import import split_dialogue

    if not segs:
        raise ValueError("no segments found: start each one with `段落 1（0–10 秒）`")
    p = _new_project(title, audio_mode, continuity_mode, default_reference_image)
    p.tail_guard_sec = tail_guard_sec
    longest = H3_MAX_SECONDS - tail_guard_sec
    for n, sd in enumerate(segs, start=1):
        f = sd.fields
        if not sd.beats:          # a segment written without 分鏡 lines is one beat
            sd.beats = [{"_id": f"{sd.number}-1", "_header": sd.header, **{k: v for k, v in f.items()
                                                                          if k in REQUIRED or k in ("camera",)}}]
        _, seg_dur = _seconds(sd.header)
        secs = _beat_seconds(f, sd.beats, seg_dur)
        total = sum(secs)
        if total > longest:       # 15 s segments keep 14.65 s
            secs = [x * longest / total for x in secs]
        secs = [round(x, 2) for x in secs]
        secs[-1] = round(sum(secs) - sum(secs[:-1]), 2)
        beats: List[BeatSpec] = []
        prev: Dict[str, str] = {}
        for k, (b, sec) in enumerate(zip(sd.beats, secs), start=1):
            b = _inherit(b, prev)
            prev = b
            sound = b.get("sound", "")
            extra = [f"音效：{b['sound_fx']}" if b.get("sound_fx") else "", f"配樂：{b['music']}" if b.get("music") else ""]
            sound = "；".join(x for x in [sound] + extra if x)
            dialogue_raw = b.get("dialogue", "")
            spoken, cue = split_dialogue(dialogue_raw)
            if re.fullmatch(r"\s*(無|none|n/?a|—|-)\s*", spoken or "", flags=re.I):
                spoken = ""
            framing = b.get("framing", "") or _framing(b.get("_header", ""))
            lighting = b.get("lighting", "") or f.get("lighting", "")
            missing = [zh for key, zh in REQUIRED.items()
                       if not {"framing": framing, "lighting": lighting, "visual": b.get("visual", ""),
                               "dialogue": "dialogue" in b, "sound": sound}[key]]
            beats.append(BeatSpec(beat_id=b.get("_id") or f"{sd.number}-{k}", duration_sec=sec, framing=framing,
                                  lighting=lighting, visual=b.get("visual", ""), dialogue=spoken, speaker=cue,
                                  sound=sound, camera=b.get("camera", ""), missing_fields=missing))
        mode = parse_mode(f.get("mode", "") or sd.header)
        if not mode and (f.get("first_frame") or f.get("last_frame")):
            mode = "first_last" if f.get("last_frame") and "首幀" not in sd.header else "reference_keyframes"
        heading = re.sub(r"[（(].*?[）)]", "", sd.header).strip(" ：:｜|")
        sid = sd.number.zfill(2)
        p.shots.append(ShotSpec(
            shot_id=sid, index=n, title=f.get("title") or heading or f"段落 {sd.number}",
            scene_id=f.get("scene") or "default",
            duration_sec=round(sum(b.duration_sec for b in beats), 2),
            visual_prompt=" ".join(b.visual for b in beats if b.visual) or heading or f"segment {sid}",
            dialogue="".join(b.dialogue for b in beats),
            subtitle_text="",
            characters=[c for c in re.split(r"[,;，；、\s]+", f.get("characters", "")) if c],
            notes=heading,
            beats=beats,
            gen_mode=mode,
            first_frame=f.get("first_frame", "").strip(),
            last_frame=f.get("last_frame", "").strip(),
        ))
    _assign_seeds(p)
    return p


def check_segments(project: StoryboardProject, rule: str = "10s_3") -> List[str]:
    """Problems against the script spec: segment length and 分鏡 count, the four modules + 景別 per 分鏡,
    lines too long for their 分鏡, missing keyframes."""
    issues: List[str] = []
    target, min_beats = SEGMENT_RULES.get(rule, (None, 0))
    longest = H3_MAX_SECONDS - project.tail_guard_sec
    for i, s in enumerate(project.shots):
        if not s.beats:
            continue
        if target:
            want = min(target, longest)
            last = i == len(project.shots) - 1
            if abs(s.duration_sec - want) > 0.5 and not (last and s.duration_sec < want):
                issues.append(f"段落 {s.shot_id}: {s.duration_sec:.1f} 秒，規格是 {target:g} 秒一段")
            if len(s.beats) < min_beats:
                issues.append(f"段落 {s.shot_id}: 只有 {len(s.beats)} 個分鏡，規格至少 {min_beats} 個")
        for b in s.beats:
            if b.missing_fields:
                issues.append(f"分鏡 {b.beat_id}: 缺 {'、'.join(b.missing_fields)}")
            chars = len(re.sub(r"[\s，。、；：！？,.!?「」（）()]", "", b.dialogue))
            if chars and chars / 4.0 > b.duration_sec - 0.5:
                issues.append(f"分鏡 {b.beat_id}: 台詞 {chars} 字約需 {chars / 4.0 + 0.5:.1f} 秒，"
                              f"超過分鏡長度 {b.duration_sec:.1f} 秒")
        if (s.gen_mode == "first_last") and not s.last_frame:
            issues.append(f"段落 {s.shot_id}: 首尾幀模式要指定「尾幀圖」")
    return issues

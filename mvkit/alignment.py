"""Character and scene bible + alignment of every shot to it (角色對齊、場景設定).

The bible is written once per video and reused by every shot, so the same person always gets the same
picture, the same wording and the same voice, and the same room always gets the same anchor picture:

    ## 角色
    | 代號 | 名稱 | 別名 | 外觀 | 參考圖 | 位置 | 聲音 | 主講 |
    | examinee | 考生 | 考生, 醫師 | 25 歲台灣男性醫學生，短黑髮，白色短袍 | char_examinee.png | 畫面左側 | calm young male voice | 是 |
    | patient  | 病人 | 病人, 林小姐 | 60 歲台灣女性，灰短髮，淺藍病人服 | char_patient.png | 病床上 | soft elderly female voice | |

    ## 場景
    | 代號 | 名稱 | 別名 | 描述 | 參考圖 | 角色 |
    | bedside | 病床旁 | 病床, 操作台, 治療台 | OSCE 技能教室，病床在右，治療台在左 | scene_bedside.png | examinee, patient |

A JSON file with the same `characters` / `scenes` keys as the storyboard JSON works too.

`align_project` then, for every shot:
  * maps the script's place text (`病床旁的操作台`) to a bible scene; framing words (`特寫`, `close-up`)
    stay in the same room and move to the camera line instead of creating a new location;
  * maps names, aliases and the scene's cast to character ids, always in bible order, so `<Picture 2>` is
    the same person in every clip;
  * resolves who speaks: `（口述）` / `（對病人）` -> the lead character, `（病人）` -> the patient,
    `（旁白）` -> the off-screen narrator.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .script_loader import (
    MAX_REFERENCE_IMAGES,
    CharacterSpec,
    SceneAnchorSpec,
    StoryboardProject,
    _resolve_path,
)

FRAMING_WORDS = ("特寫", "近景", "大特寫", "中景", "遠景", "全景", "俯拍", "仰拍", "close-up", "closeup",
                 "close up", "extreme close", "wide shot", "medium shot", "overhead", "insert")
NARRATOR_CUES = ("旁白", "畫外音", "os", "o.s.", "v.o.", "vo", "narrator", "narration", "字卡")
LEAD_CUES = ("口述", "自述", "說明", "解說", "唸出", "念出")
_YES = ("是", "y", "yes", "true", "1", "v", "✓", "主講", "lead")


@dataclass
class Bible:
    characters: Dict[str, CharacterSpec] = field(default_factory=dict)
    scenes: Dict[str, SceneAnchorSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------- bible files
def _split(cell: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;，；、/／]+", cell or "") if x.strip()]


def _md_tables(text: str) -> List[tuple]:
    """[(heading, headers, rows)] for every Markdown table, with the nearest heading above it."""
    tables: List[tuple] = []
    heading, headers, rows = "", [], []

    def flush():
        if headers:
            tables.append((heading, headers, rows))

    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
                continue
            if not headers:
                headers = [c.lower() for c in cells]
            else:
                rows.append(cells)
            continue
        if headers:
            flush()
            headers, rows = [], []
        if s.startswith("#") or re.match(r"^(角色|人物|場景|地點|characters?|scenes?)\s*[:：]?$", s, re.I):
            heading = s.lstrip("#").strip().lower()
    flush()
    return tables


def _col(headers: List[str], *keys: str) -> int:
    for i, h in enumerate(headers):
        if any(k in h for k in keys):
            return i
    return -1


def parse_bible_markdown(text: str) -> Bible:
    bible = Bible()
    for heading, headers, rows in _md_tables(text):
        is_scene = any(k in heading for k in ("場景", "地點", "scene", "location"))
        is_char = any(k in heading for k in ("角色", "人物", "character", "cast"))
        if not (is_scene or is_char):
            is_char = _col(headers, "聲音", "外觀", "voice", "位置", "position") >= 0
            is_scene = not is_char
        c_id = _col(headers, "代號", "id", "編號")
        c_name = _col(headers, "名稱", "name", "姓名")
        c_alias = _col(headers, "別名", "alias", "稱呼")
        c_img = _col(headers, "參考圖", "圖片", "image", "ref")
        if is_char:
            c_desc = _col(headers, "外觀", "描述", "description", "appearance")
            c_pos = _col(headers, "位置", "position", "站位")
            c_voice = _col(headers, "聲音", "voice", "嗓音")
            c_lead = _col(headers, "主講", "lead", "主角")
        else:
            c_desc = _col(headers, "描述", "description", "設定", "環境")
            c_cast = _col(headers, "角色", "人物", "cast", "characters")
        for row in rows:
            get = (lambda c: row[c].strip() if 0 <= c < len(row) else "")
            name = get(c_name)
            cid = get(c_id) or name
            if not cid:
                continue
            aliases = _split(get(c_alias))
            if is_char:
                bible.characters[cid] = CharacterSpec(
                    cid, name or cid, get(c_desc), get(c_img), get(c_pos), aliases, get(c_voice),
                    get(c_lead).lower() in _YES)
            else:
                bible.scenes[cid] = SceneAnchorSpec(cid, get(c_desc), get(c_img), name or cid, aliases,
                                                    _split(get(c_cast)))
    return bible


def parse_bible_json(data: dict) -> Bible:
    from .script_loader import load_storyboard_from_json
    tmp = load_storyboard_from_json({"characters": data.get("characters") or {},
                                     "scenes": data.get("scenes") or {}, "shots": []})
    return Bible(tmp.characters, tmp.scenes)


def load_bible(source: str | Path) -> Bible:
    """A bible file (.md table / .json), or the bible text itself."""
    text = str(source)
    if "\n" not in text and Path(text).is_file():
        text = Path(text).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("{"):
        return parse_bible_json(json.loads(text))
    return parse_bible_markdown(text)


def apply_bible(project: StoryboardProject, bible: Bible) -> StoryboardProject:
    """Bible entries replace same-id entries already in the project; the rest are kept."""
    project.characters.update(bible.characters)
    for sid, sc in bible.scenes.items():
        project.scenes[sid] = sc
    return project


# ---------------------------------------------------------------------- alignment
def _names(spec) -> List[str]:
    out = [spec.id, getattr(spec, "name", "")] + list(spec.aliases or [])
    return [n for n in dict.fromkeys(x.strip() for x in out if x and x.strip())]


def _lead(project: StoryboardProject) -> Optional[str]:
    for cid, ch in project.characters.items():
        if ch.lead:
            return cid
    return next(iter(project.characters), None)


def find_character(project: StoryboardProject, text: str) -> Optional[str]:
    """Character id whose id/name/alias equals or is contained in `text` (longest match wins)."""
    text = (text or "").strip().lower()
    if not text:
        return None
    best, best_len = None, 0
    for cid, ch in project.characters.items():
        for n in _names(ch):
            n_l = n.lower()
            if n_l == text:
                return cid
            if len(n_l) >= 2 and n_l in text and len(n_l) > best_len:
                best, best_len = cid, len(n_l)
    return best


def find_scene(project: StoryboardProject, text: str) -> Optional[str]:
    text = (text or "").strip().lower()
    if not text:
        return None
    if text in project.scenes:
        return text
    best, best_len = None, 0
    for sid, sc in project.scenes.items():
        if sid == "default":
            continue
        for n in _names(sc):
            n_l = n.lower()
            if len(n_l) >= 2 and (n_l in text or (len(text) >= 2 and text in n_l)) and len(n_l) > best_len:
                best, best_len = sid, len(n_l)
    return best


def is_framing(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in FRAMING_WORDS)


def resolve_speaker(project: StoryboardProject, cue: str) -> tuple:
    """(speaker id or "" for the narrator, addressee id or None, recognised?)"""
    cue = (cue or "").strip()
    if not cue:
        return "", None, True
    low = cue.lower()
    if cue in project.characters:
        return cue, None, True
    if low in NARRATOR_CUES or any(low.startswith(x) for x in ("旁白", "畫外音")):
        return "", None, True
    if cue.startswith(("對", "向", "跟", "to ")):
        target = find_character(project, re.sub(r"^(對|向|跟|to\s+)", "", cue))
        return _lead(project) or "", target, True
    if any(cue.startswith(x) for x in LEAD_CUES):
        return _lead(project) or "", None, True
    who = find_character(project, cue)
    if who:
        return who, None, True
    return cue, None, False


@dataclass
class ShotAlignment:
    shot_id: str
    scene_id: str
    characters: List[str]
    speaker: str
    pictures: List[str]
    warnings: List[str]


def align_project(project: StoryboardProject, base_dirs: Optional[List[str | Path]] = None
                  ) -> List[ShotAlignment]:
    """Rewrite scene_id / characters / speaker of every shot against the bible. Returns a per-shot report."""
    dirs = [Path(b) for b in (base_dirs or []) if b]
    order = {cid: i for i, cid in enumerate(project.characters)}
    real_scenes = [s for s in project.scenes if s != "default"]
    report: List[ShotAlignment] = []
    prev_scene: Optional[str] = None

    for shot in project.shots:
        warn: List[str] = []
        # who is in the shot: listed names, names mentioned in the text, the speaker and who they talk to
        chars: List[str] = []
        for c in shot.characters:
            cid = c if c in project.characters else find_character(project, c)
            if cid:
                chars.append(cid)
            else:
                warn.append(f"角色「{c}」不在角色表")
        haystack = " ".join((shot.title, shot.visual_prompt, shot.dialogue))
        for cid, ch in project.characters.items():
            if any(len(n) >= 2 and n in haystack for n in _names(ch)):
                chars.append(cid)
        if shot.speaker and shot.speaker not in project.characters:
            spk, addressee, ok = resolve_speaker(project, shot.speaker)
            if not ok:
                warn.append(f"說話者「{shot.speaker}」對不到角色，請在角色表加別名")
            shot.speaker = spk
            if addressee:
                chars.append(addressee)
        if shot.speaker in project.characters:
            chars.append(shot.speaker)

        # where: bible scene by name/alias; a framing-only line ("特寫上臂") stays in a room that holds
        # everyone in the shot, preferring the previous shot's room
        raw_scene = shot.scene_id if shot.scene_id != "default" else ""
        sid = find_scene(project, raw_scene) if raw_scene else None
        if sid is None and raw_scene and is_framing(raw_scene):
            shot.camera_motion = ", ".join(x for x in (raw_scene, shot.camera_motion) if x)
            fits = [x for x in real_scenes if set(chars) <= set(project.scenes[x].cast)]
            sid = prev_scene if prev_scene in fits or not fits else fits[0]
            sid = sid or (real_scenes[0] if real_scenes else "default")
        elif sid is None and raw_scene:
            if len(real_scenes) == 1:
                sid = real_scenes[0]
                shot.visual_prompt = f"{shot.visual_prompt}（位置：{raw_scene}）"
            elif real_scenes:
                warn.append(f"場景「{raw_scene}」對不到場景表，請加別名；暫用 {prev_scene or real_scenes[0]}")
                sid = prev_scene or real_scenes[0]
            else:
                sid = "default"
                warn.append(f"沒有場景表：「{raw_scene}」只寫進文字，畫面以預設場景圖為準")
                shot.visual_prompt = f"{shot.visual_prompt}（位置：{raw_scene}）"
        elif sid is None:
            sid = prev_scene or (real_scenes[0] if real_scenes else "default")
        shot.scene_id = sid
        prev_scene = sid
        scene = project.scenes.get(sid)
        if scene:
            chars += [c for c in scene.cast if c in project.characters]

        shot.characters = sorted(dict.fromkeys(chars), key=lambda c: order.get(c, 99))
        if len(shot.characters) > MAX_REFERENCE_IMAGES - 1:
            warn.append(f"{len(shot.characters)} 位角色超過 H3 參考圖上限（場景圖 + 8 人），只送前 8 位")

        pics: List[str] = []
        sc_img = _resolve_path(scene.image_path, dirs) if scene else ""
        if sc_img and Path(sc_img).is_file():
            pics.append(Path(sc_img).name)
        elif not shot.reference_images:
            warn.append(f"場景 {sid} 沒有參考圖（<Picture 1>）：H3 每鏡至少要一張圖，請上傳場景照片"
                        f"（實際技能教室照片最穩）")
        for cid in shot.characters:
            img = _resolve_path(project.characters[cid].image_path, dirs)
            if img and Path(img).is_file():
                pics.append(Path(img).name)
            else:
                warn.append(f"角色 {cid} 沒有參考圖：長相只靠文字描述，跨鏡容易跑掉")
        report.append(ShotAlignment(shot.shot_id, sid, list(shot.characters), shot.speaker, pics, warn))
    return report


def format_alignment_report(project: StoryboardProject, report: List[ShotAlignment]) -> str:
    def name(cid: str) -> str:
        ch = project.characters.get(cid)
        return ch.name if ch else cid

    lines = ["| 鏡號 | 秒數 | 場景 | 角色（<Picture 2..> 順序） | 說話者 | 參考圖 |",
             "| --- | --- | --- | --- | --- | --- |"]
    by_id = {s.shot_id: s for s in project.shots}
    for r in report:
        shot = by_id[r.shot_id]
        lines.append(f"| {r.shot_id} | {shot.duration_sec:.1f} | {r.scene_id} | "
                     f"{'、'.join(name(c) for c in r.characters) or '—'} | "
                     f"{name(r.speaker) if r.speaker else ('旁白' if shot.dialogue else '—')} | "
                     f"{', '.join(r.pictures) or '（無）'} |")
    grouped: Dict[str, List[str]] = {}
    for r in report:
        for w in r.warnings:
            grouped.setdefault(w, []).append(r.shot_id)
    if grouped:
        lines += ["", f"⚠️ {len(grouped)} 個提醒："]
        lines += [f"- {w}（鏡 {', '.join(ids)}）" for w, ids in grouped.items()]
    return "\n".join(lines)

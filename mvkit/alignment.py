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
    for k in keys:                       # exact header first: "參考圖" must not pick "其他參考圖"
        if k in headers:
            return headers.index(k)
    for i, h in enumerate(headers):
        if any(k in h for k in keys):
            return i
    return -1


_TIERS = {"主要": "main", "主角": "main", "男主": "main", "女主": "main", "main": "main",
          "次要": "secondary", "配角": "secondary", "secondary": "secondary",
          "路人": "npc", "npc": "npc", "群眾": "npc", "臨演": "npc"}


def _tier(cell: str) -> str:
    c = (cell or "").strip().lower()
    return next((v for k, v in _TIERS.items() if k in c), "")


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
            c_tier = _col(headers, "等級", "分級", "tier", "層級")
            c_extra = _col(headers, "其他參考圖", "三視圖", "特寫圖", "extra")
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
                    get(c_lead).lower() in _YES, _tier(get(c_tier)), _split(get(c_extra)))
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


OFFSCREEN_MARKS = ("畫外音", "畫面外", "畫外", "內心", "o.s.", "v.o.", "os", "vo")
WIDE_FRAMINGS = ("遠景", "全景", "wide", "full shot", "establishing")


def split_offscreen(cue: str) -> tuple:
    """'考生畫外' -> ('考生', True); '考生OS' -> ('考生', True); '對病人' -> ('對病人', False)."""
    c = (cue or "").strip()
    low = c.lower()
    for m in OFFSCREEN_MARKS:
        if m in ("os", "vo"):
            if low.endswith(m) and (len(low) == len(m) or not low[-len(m) - 1].isascii()
                                    or not low[-len(m) - 1].isalpha()):
                return c[:len(c) - len(m)].strip(" -_/、,，"), True
            continue
        if m in low:
            i = low.index(m)
            return (c[:i] + c[i + len(m):]).strip(" -_/、,，"), True
    return c, False


def _resolve_names(project: StoryboardProject, names: List[str], warn: List[str]) -> List[str]:
    out: List[str] = []
    for c in names:
        cid = c if c in project.characters else find_character(project, c)
        if cid:
            out.append(cid)
        elif not re.fullmatch(r"(無|none|沒有人|無人|—|-)", c.strip(), flags=re.I):
            warn.append(f"角色「{c}」不在角色表")
    return out


def _mentioned(project: StoryboardProject, text: str) -> List[str]:
    return [cid for cid, ch in project.characters.items()
            if any(len(n) >= 2 and n in (text or "") for n in _names(ch))]


def _resolve_cue(project: StoryboardProject, cue: str, warn: List[str]) -> tuple:
    """Script cue -> (speaker id or "", addressee id or None, voice_over forced?)."""
    raw, off = split_offscreen(cue)
    if off and not raw:                  # （OS）/（畫外音） alone: the off-screen narrator, as before
        return "", None, True
    if raw in project.characters:
        return raw, None, off
    spk, addressee, ok = resolve_speaker(project, raw)
    if not ok:
        warn.append(f"說話者「{raw}」對不到角色，請在角色表加別名")
    return spk, addressee, off


def align_beat_cast(project: StoryboardProject, beat, segment_cast: List[str], fallback_cast: List[str],
                    warn: List[str]) -> None:
    """Decide who is VISIBLE in one 分鏡 and whether its line is a voice-over.

    * `出場：考生、病人` written in the 分鏡 wins (`出場：無` = nobody on screen).
    * otherwise: people named in the 畫面描述, the speaker (unless it is a hands / instrument close-up or the
      cue says 畫外), and the person spoken to in a wide or medium shot;
    * a wide shot that names nobody shows the segment's `角色：` (or the scene's cast).
    Only visible people get a reference picture, so a close-up never drags the whole cast into frame.
    """
    from .script_loader import is_detail_beat

    detail = is_detail_beat(beat)
    framing = (beat.framing or "").lower()
    wide = any(w in framing for w in WIDE_FRAMINGS)
    spk, addressee, off = "", None, False
    if beat.speaker and beat.speaker not in project.characters:
        spk, addressee, off = _resolve_cue(project, beat.speaker, warn)
        beat.speaker = spk
    if beat.cast:
        vis = _resolve_names(project, beat.cast, warn)
    else:
        vis = _mentioned(project, beat.visual)
        if beat.speaker in project.characters and not detail and not off:
            vis.append(beat.speaker)
        if addressee and not detail:
            vis.append(addressee)
        if not vis and wide:
            vis = list(segment_cast or fallback_cast)
    order = {cid: i for i, cid in enumerate(project.characters)}
    beat.on_screen = sorted(dict.fromkeys(vis), key=lambda c: order.get(c, 99))
    beat.voice_over = bool(beat.speaker) and (off or detail or beat.speaker not in beat.on_screen)


def align_project(project: StoryboardProject, base_dirs: Optional[List[str | Path]] = None
                  ) -> List[ShotAlignment]:
    """Rewrite scene_id / characters / speaker of every shot against the bible. Returns a per-shot report.

    `shot.characters` ends up as the people VISIBLE somewhere in the clip (the union of its 分鏡), in bible
    order. They - and only they - get `<Picture 2..>`; a scene's full cast is no longer added to every clip,
    because H3 Ref2VA tends to paint every person it is shown (extra / duplicated people).
    """
    dirs = [Path(b) for b in (base_dirs or []) if b]
    order = {cid: i for i, cid in enumerate(project.characters)}
    real_scenes = [s for s in project.scenes if s != "default"]
    report: List[ShotAlignment] = []
    prev_scene: Optional[str] = None

    for shot in project.shots:
        warn: List[str] = []
        listed = _resolve_names(project, shot.characters, warn)

        # where: bible scene by name/alias; a framing-only line ("特寫上臂") stays in a room that holds
        # everyone in the shot, preferring the previous shot's room
        raw_scene = shot.scene_id if shot.scene_id != "default" else ""
        sid = find_scene(project, raw_scene) if raw_scene else None
        if sid is None and raw_scene and is_framing(raw_scene):
            shot.camera_motion = ", ".join(x for x in (raw_scene, shot.camera_motion) if x)
            people = listed + _mentioned(project, shot.visual_prompt)
            fits = [x for x in real_scenes if set(people) <= set(project.scenes[x].cast)]
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
        scene_cast = [c for c in (scene.cast if scene else []) if c in project.characters]

        chars: List[str] = []
        if shot.beats:
            for b in shot.beats:
                align_beat_cast(project, b, listed, scene_cast, warn)
                chars += b.on_screen
        else:
            # one-cut shots: listed + named people, the speaker and who they talk to; the scene's cast
            # only when the shot names nobody at all
            chars = listed + _mentioned(project, " ".join((shot.title, shot.visual_prompt, shot.dialogue)))
            if shot.speaker and shot.speaker not in project.characters:
                spk, addressee, off = _resolve_cue(project, shot.speaker, warn)
                shot.speaker = spk
                if addressee:
                    chars.append(addressee)
            if shot.speaker in project.characters:
                chars.append(shot.speaker)
            if not chars:
                chars = scene_cast

        shot.characters = sorted(dict.fromkeys(chars), key=lambda c: order.get(c, 99))
        if len(shot.characters) > MAX_REFERENCE_IMAGES - 1:
            warn.append(f"{len(shot.characters)} 位角色超過 H3 參考圖上限（場景圖 + 8 人），只送前 8 位")
        if len(shot.characters) >= 4:
            warn.append(f"同一支片段有 {len(shot.characters)} 人入鏡，人越多越容易重複或換臉；可考慮拆段或用「出場：」限定")

        from .script_loader import shot_gen_mode
        if shot_gen_mode(project, shot) in ("first_frame", "first_last") and not shot.first_frame \
                and shot.beats and shot.beats[0].on_screen:
            warn.append("首幀/首尾幀模式沒有「首幀圖」：會用場景圖當第一格，人物可能憑空出現或重複；建議指定首幀圖")

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
        report.append(ShotAlignment(shot.shot_id, sid, list(shot.characters), shot.speaker, pics,
                                    list(dict.fromkeys(warn))))
    return report


def _speakers(shot, name) -> str:
    if shot.beats:
        who = [name(b.speaker) if b.speaker else "旁白" for b in shot.beats if b.dialogue]
        return "、".join(dict.fromkeys(who)) or "—"
    return name(shot.speaker) if shot.speaker else ("旁白" if shot.dialogue else "—")


def format_alignment_report(project: StoryboardProject, report: List[ShotAlignment]) -> str:
    def name(cid: str) -> str:
        ch = project.characters.get(cid)
        return ch.name if ch else cid

    from .script_loader import shot_gen_mode
    mode_zh = {"reference": "參考圖", "reference_keyframes": "參考圖+首幀", "first_frame": "首幀",
               "first_last": "首尾幀"}
    lines = ["| 鏡號 | 秒數 | 分鏡 | 模式 | 場景 | 角色（<Picture 2..> 順序） | 說話者 | 參考圖 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    by_id = {s.shot_id: s for s in project.shots}
    for r in report:
        shot = by_id[r.shot_id]
        beats = "→".join(b.framing or "?" for b in shot.beats) or "—"
        lines.append(f"| {r.shot_id} | {shot.duration_sec:.1f} | {beats} | {mode_zh[shot_gen_mode(project, shot)]} | "
                     f"{r.scene_id} | "
                     f"{'、'.join(name(c) for c in r.characters) or '—'} | "
                     f"{_speakers(shot, name)} | "
                     f"{', '.join(r.pictures) or '（無）'} |")
    grouped: Dict[str, List[str]] = {}
    for r in report:
        for w in r.warnings:
            grouped.setdefault(w, []).append(r.shot_id)
    if grouped:
        lines += ["", f"⚠️ {len(grouped)} 個提醒："]
        lines += [f"- {w}（鏡 {', '.join(ids)}）" for w, ids in grouped.items()]
    return "\n".join(lines)

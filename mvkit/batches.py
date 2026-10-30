"""Generate a long video in review batches (default 2 minutes) so GPU time is never spent past a bad shot.

    plan_batches()          storyboard -> consecutive batches of <= 120 s, cut at scene changes when possible
    preflight_batch()       every shot has a picture before any model is downloaded
    H3ColabRunner.run_storyboard(..., shot_ids=batch.shot_ids)   generate only that batch
    assemble_batch_preview() preview MP4 + contact sheet + review checklist for that batch
    approve_batch() / request_redo()   the gate: batch k+1 only starts after batch k is approved

State lives in `<work_dir>/review_state.json` next to `progress.json`:
  * approvals are stored per shot with a fingerprint of everything that shapes the clip, so editing a shot's
    text after approval re-opens it automatically;
  * a redo bumps the shot's seed (the same seed and prompt would give the same clip again) and moves the old
    clip to `raw_clips/_rejected/` for comparison.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .script_loader import (
    ShotSpec,
    StoryboardProject,
    _resolve_path,
    framing_en,
    shot_engine,
    shot_gen_mode,
    shot_picture_plan,
)

DEFAULT_BATCH_SECONDS = 120.0
REVIEW_FILE = "review_state.json"


@dataclass
class Batch:
    index: int                        # 1-based
    shot_ids: List[str]
    start_sec: float
    seconds: float

    def label(self) -> str:
        m0, s0 = divmod(int(self.start_sec), 60)
        m1, s1 = divmod(int(self.start_sec + self.seconds), 60)
        return f"第 {self.index} 批（{m0:02d}:{s0:02d}–{m1:02d}:{s1:02d}，{len(self.shot_ids)} 支片段）"


def plan_batches(project: StoryboardProject, max_seconds: float = DEFAULT_BATCH_SECONDS) -> List[Batch]:
    """Greedy split into batches of at most `max_seconds`; if a scene change falls in the last 40 % of a batch,
    cut there instead so a review batch does not end in the middle of a scene."""
    shots = project.shots
    batches: List[Batch] = []
    i, t = 0, 0.0
    while i < len(shots):
        j, total, cut = i, 0.0, None
        while j < len(shots) and (j == i or total + shots[j].duration_sec <= max_seconds + 1e-6):
            if j > i and shots[j].scene_id != shots[j - 1].scene_id and total >= 0.6 * max_seconds:
                cut = j
            total += shots[j].duration_sec
            j += 1
        if j < len(shots) and cut is not None:
            j = cut
        group = shots[i:j]
        secs = sum(s.duration_sec for s in group)
        batches.append(Batch(len(batches) + 1, [s.shot_id for s in group], round(t, 3), round(secs, 3)))
        t += secs
        i = j
    return batches


# ---------------------------------------------------------------------- state
def shot_fingerprint(project: StoryboardProject, shot: ShotSpec) -> str:
    """Hash of everything that changes the generated clip (text, seed, pictures, project-wide prompt text)."""
    keys = ("style_prefix", "continuity_rules", "negative_rules", "narrator", "overall_soundscape",
            "audio_mode", "inference_mode", "width", "height", "tail_guard_sec", "continuity_mode",
            "ref_image_size")
    d = {k: getattr(project, k) for k in keys}
    d["shot"] = {k: v for k, v in asdict(shot).items() if k not in ("index", "title", "subtitle_text", "notes")}
    scene = project.scenes.get(shot.scene_id)
    d["scene"] = asdict(scene) if scene else None
    d["characters"] = [asdict(project.characters[c]) for c in shot.characters if c in project.characters]
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def load_review_state(work_dir: str | Path) -> Dict[str, Any]:
    f = Path(work_dir) / REVIEW_FILE
    state = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    for key in ("approved", "seeds", "attempts"):
        state.setdefault(key, {})
    state.setdefault("log", [])
    return state


def save_review_state(work_dir: str | Path, state: Dict[str, Any]) -> None:
    f = Path(work_dir) / REVIEW_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def apply_review_seeds(project: StoryboardProject, work_dir: str | Path) -> None:
    """Re-apply seeds chosen by earlier redo requests (call after loading the storyboard, before generating)."""
    seeds = load_review_state(work_dir)["seeds"]
    for s in project.shots:
        if s.shot_id in seeds:
            s.seed = int(seeds[s.shot_id])


def _clip_path(work_dir: str | Path, shot: ShotSpec) -> Path:
    return Path(work_dir) / "raw_clips" / f"{shot.index:02d}_{shot.shot_id}.mp4"


def _progress(work_dir: str | Path) -> Dict[str, Any]:
    f = Path(work_dir) / "progress.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def shot_is_generated(project: StoryboardProject, work_dir: str | Path, shot: ShotSpec,
                      progress: Optional[Dict[str, Any]] = None) -> bool:
    progress = _progress(work_dir) if progress is None else progress
    clip = _clip_path(work_dir, shot)
    rec = progress.get(shot.shot_id)
    return bool(rec) and clip.is_file() and clip.stat().st_size > 0 and \
        rec.get("fingerprint", shot_fingerprint(project, shot)) == shot_fingerprint(project, shot)


def batch_status(project: StoryboardProject, work_dir: str | Path, batch: Batch) -> str:
    """'approved' | 'generated' (waiting for review) | 'partial' | 'pending'."""
    state, progress = load_review_state(work_dir), _progress(work_dir)
    shots = [s for s in project.shots if s.shot_id in batch.shot_ids]
    if all(state["approved"].get(s.shot_id) == shot_fingerprint(project, s) for s in shots):
        return "approved"
    done = [shot_is_generated(project, work_dir, s, progress) for s in shots]
    if all(done):
        return "generated"
    return "partial" if any(done) else "pending"


def next_batch(project: StoryboardProject, work_dir: str | Path, batches: Sequence[Batch]) -> Optional[Batch]:
    for b in batches:
        if batch_status(project, work_dir, b) != "approved":
            return b
    return None


def check_gate(project: StoryboardProject, work_dir: str | Path, batches: Sequence[Batch], batch: Batch) -> None:
    """Refuse to generate batch k while an earlier batch is not approved."""
    for b in batches:
        if b.index >= batch.index:
            break
        st = batch_status(project, work_dir, b)
        if st != "approved":
            raise RuntimeError(f"{b.label()} 尚未確認（狀態：{st}）。請先在審核步驟按「approve」或指定重拍鏡號，"
                               f"確認無誤後再生成下一批，避免浪費 Colab 運算單元。")


def approve_batch(project: StoryboardProject, work_dir: str | Path, batch: Batch, note: str = "") -> None:
    state, progress = load_review_state(work_dir), _progress(work_dir)
    missing = [s.shot_id for s in project.shots
               if s.shot_id in batch.shot_ids and not shot_is_generated(project, work_dir, s, progress)]
    if missing:
        raise RuntimeError(f"{batch.label()} 還有未生成的鏡頭 {missing}，不能確認")
    for s in project.shots:
        if s.shot_id in batch.shot_ids:
            state["approved"][s.shot_id] = shot_fingerprint(project, s)
    state["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "batch": batch.index,
                         "decision": "approve", "note": note})
    save_review_state(work_dir, state)


def chained_dependents(project: StoryboardProject, shot_id: str) -> List[str]:
    """Later shots whose first frame is taken from this shot's clip (same scene, chaining continuity)."""
    out: List[str] = []
    ids = [s.shot_id for s in project.shots]
    if shot_id not in ids:
        return out
    i = ids.index(shot_id)
    for prev, s in zip(project.shots[i:], project.shots[i + 1:]):
        if _chains(project, prev, s):
            out.append(s.shot_id)
        else:
            break
    return out


def _chains(project: StoryboardProject, prev: ShotSpec, s: ShotSpec) -> bool:
    """Does `s` start from `prev`'s last frame? (mirrors H3ColabRunner.plan_generation)"""
    if s.first_frame:
        return False
    if prev.end_framing and s.start_framing and framing_en(prev.end_framing) != framing_en(s.start_framing):
        return False
    mode = s.continuity_override or project.continuity_mode
    return mode == "chain_last_frame" or (mode == "hybrid" and s.scene_id == prev.scene_id)


def request_redo(project: StoryboardProject, work_dir: str | Path, shot_ids: Sequence[str], *,
                 note: str = "", new_seed: bool = True) -> List[str]:
    """Mark shots for regeneration. Returns already-generated shots chained from them (consider redoing them too)."""
    state = load_review_state(work_dir)
    by_id = {s.shot_id: s for s in project.shots}
    unknown = [x for x in shot_ids if x not in by_id]
    if unknown:
        raise ValueError(f"找不到鏡號 {unknown}；可用鏡號：{list(by_id)}")
    progress_file = Path(work_dir) / "progress.json"
    progress = _progress(work_dir)
    rejected = Path(work_dir) / "raw_clips" / "_rejected"
    affected: List[str] = []
    for sid in shot_ids:
        shot = by_id[sid]
        n = int(state["attempts"].get(sid, 0)) + 1
        state["attempts"][sid] = n
        if new_seed:
            shot.seed = (shot.seed + 7919 * n) % (2 ** 63)
            state["seeds"][sid] = shot.seed
        state["approved"].pop(sid, None)
        clip = _clip_path(work_dir, shot)
        if clip.is_file():
            rejected.mkdir(parents=True, exist_ok=True)
            clip.replace(rejected / f"{clip.stem}_take{n}.mp4")
        progress.pop(sid, None)
        affected += [d for d in chained_dependents(project, sid)
                     if d not in shot_ids and shot_is_generated(project, work_dir, by_id[d], progress)]
    if progress_file.is_file():
        progress_file.write_text(json.dumps(progress, indent=1, ensure_ascii=False), encoding="utf-8")
    state["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "decision": "redo",
                         "shots": list(shot_ids), "note": note})
    save_review_state(work_dir, state)
    return list(dict.fromkeys(affected))


# ---------------------------------------------------------------------- before / after a batch
def preflight_batch(project: StoryboardProject, batch: Batch, base_dirs: Optional[List[str | Path]] = None
                    ) -> List[str]:
    """Problems that would make the batch fail on the GPU (checked before any model is downloaded)."""
    problems: List[str] = []
    dirs = [Path(b) for b in (base_dirs or []) if b]
    shots = project.shots
    for i, s in enumerate(shots):
        if s.shot_id not in batch.shot_ids:
            continue
        can_chain = i > 0 and _chains(project, shots[i - 1], s)
        mode = shot_gen_mode(project, s)
        for raw, what in ((s.first_frame, "首幀圖"), (s.last_frame, "尾幀圖")):
            if raw and not Path(_resolve_path(raw, dirs)).is_file():
                problems.append(f"{s.shot_id}: {what} {raw} 找不到（請在 Step 2 上傳）")
        if mode == "first_last" and not s.last_frame:
            problems.append(f"{s.shot_id}: 首尾幀模式需要「尾幀圖」")
        if mode == "reference_keyframes" and not (s.first_frame or s.last_frame):
            problems.append(f"{s.shot_id}: 參考圖+首幀模式需要「首幀圖」或「尾幀圖」")
        if not shot_picture_plan(project, s, dirs) and not can_chain and not s.first_frame:
            problems.append(f"{s.shot_id}: 沒有任何參考圖（場景 {s.scene_id} 需要 <Picture 1>）")
    return problems


def engines_for(project: StoryboardProject, batch: Batch) -> List[str]:
    """H3 models this batch needs ('reference' = Ref2VA, 'first_frame' = FL2VA)."""
    return sorted({shot_engine(project, s) for s in project.shots if s.shot_id in batch.shot_ids})


def estimate_gpu_minutes(project: StoryboardProject, work_dir: str | Path,
                         shot_ids: Optional[Sequence[str]] = None) -> Optional[Dict[str, float]]:
    """GPU minutes for the listed (default: all not yet generated) shots, from the speed measured so far."""
    progress = _progress(work_dir)
    timed = [r for r in progress.values() if r.get("elapsed_s") and r.get("seconds")]
    if not timed:
        return None
    rate = sum(r["elapsed_s"] for r in timed) / sum(r["seconds"] for r in timed)   # GPU s per video s
    todo = [s for s in project.shots if (shot_ids is None and not shot_is_generated(project, work_dir, s, progress))
            or (shot_ids is not None and s.shot_id in shot_ids)]
    secs = sum(project.generation_seconds(s) for s in todo)
    return {"gpu_sec_per_video_sec": round(rate, 1), "video_sec": round(secs, 1),
            "gpu_minutes": round(rate * secs / 60, 1), "shots": len(todo)}


def write_review_checklist(project: StoryboardProject, batch: Batch, path: str | Path,
                           preview: Optional[Dict[str, Any]] = None) -> Path:
    """Markdown checklist for the reviewer (content expert): one row per shot with the scoring points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {cid: ch.name for cid, ch in project.characters.items()}
    lines = [f"# {project.title} — {batch.label()} 審核表", ""]
    if preview:
        lines += [f"- 預覽影片：`{preview.get('subtitled_mp4') or preview.get('clean_mp4')}`",
                  f"- 抽幀檢視圖：`{preview.get('review_sheet')}`", ""]
    lines += ["每支片段請確認：角色長相/服裝一致 · 場景與道具一致 · 動作/技術步驟正確 · 台詞與口型正確 · "
              "沒有多餘人物、文字或變形的手。", "",
              "| 鏡號 | 秒數 | 角色 | 台詞 | 評分重點 / 備註 | 結果 |", "| --- | --- | --- | --- | --- | --- |"]
    for s in project.shots:
        if s.shot_id in batch.shot_ids:
            who = "、".join(names.get(c, c) for c in s.characters) or "—"
            lines.append(f"| {s.shot_id} | {s.duration_sec:.1f} | {who} | {s.dialogue or '—'} | "
                         f"{s.notes or '—'} | ☐ 通過 ☐ 重拍 |")
            for b in s.beats:
                lines.append(f"| ↳ {b.beat_id} {b.framing} | {b.duration_sec:.1f} | | {b.dialogue or '—'} | "
                             f"{b.visual[:40]} | |")
    lines += ["", "全部通過 → 審核步驟選 `approve`；有問題 → 選 `redo_shots` 並填入要重拍的鏡號（逗號分隔）。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _slice_audio(src: Path, start: float, seconds: float, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds + 0.5:.3f}",
                    "-i", str(src), str(dst)], check=True)
    return dst


def batch_project(project: StoryboardProject, batch: Batch) -> StoryboardProject:
    """A copy of the project holding only this batch's shots (absolute subtitle cues shifted to its start)."""
    sub = copy.deepcopy(project)
    sub.shots = [s for s in sub.shots if s.shot_id in batch.shot_ids]
    sub.title = f"{project.title}_batch{batch.index:02d}"
    end = batch.start_sec + batch.seconds
    sub.subtitle_cues = [(max(0.0, a - batch.start_sec), min(b, end) - batch.start_sec, t)
                         for a, b, t in project.subtitle_cues if b > batch.start_sec and a < end]
    return sub


def assemble_batch_preview(project: StoryboardProject, batch: Batch, work_dir: str | Path, *,
                           base_dir: Optional[Path] = None, burn_subtitles: bool = True) -> Dict[str, Any]:
    """Preview of one batch: the same post-production as the final film, plus the review checklist."""
    from .colab_pipeline import assemble_movie

    sub = batch_project(project, batch)
    out = Path(work_dir) / "batches" / f"batch{batch.index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if project.audio_mode == "master_audio" and project.master_audio_path:
        master = Path(project.master_audio_path)
        if not master.is_absolute() and base_dir is not None:
            master = Path(base_dir) / master
        if master.is_file():
            sub.master_audio_path = str(_slice_audio(master, batch.start_sec, batch.seconds,
                                                     out / f"master_slice{master.suffix or '.wav'}"))
    clips = [_clip_path(work_dir, s) for s in sub.shots]
    manifest = assemble_movie(sub, clips, out, base_dir=base_dir, burn_subtitles=burn_subtitles)
    manifest["checklist"] = str(write_review_checklist(project, batch, out / "review_checklist.md", manifest))
    return manifest


def all_approved(project: StoryboardProject, work_dir: str | Path, batches: Sequence[Batch]) -> bool:
    return all(batch_status(project, work_dir, b) == "approved" for b in batches)


def batch_overview(project: StoryboardProject, work_dir: str | Path, batches: Sequence[Batch]) -> str:
    icon = {"approved": "✅ 已確認", "generated": "👀 待審核", "partial": "⏳ 部分完成", "pending": "⬜ 未生成"}
    return "\n".join(f"{icon[batch_status(project, work_dir, b)]}  {b.label()}  {', '.join(b.shot_ids)}"
                     for b in batches)

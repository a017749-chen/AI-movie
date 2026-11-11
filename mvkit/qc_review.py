"""Automatic checks that catch the two defects seen in real runs, so the reviewer looks at the right clips.

Before the GPU (Step 3, CPU only)
  check_reference_images()  character pictures must show ONE person in ONE view; scene pictures should be
                            empty rooms. A turnaround sheet or a scene photo with people in it is the most
                            common reason H3 Ref2VA paints the same person twice.

After a batch is generated (Step 4)
  speech_issues()           faster-whisper (CPU) listens to every clip and compares, per 分鏡, what is said
                            and when with the script: wrong / missing lines, a line spoken in the wrong 分鏡,
                            voices where nobody should speak.
  face_issues()             OpenCV counts faces at the middle of every 分鏡; more faces than people written in
                            the script is flagged as a possible duplicate / extra person.

Both are hints for the reviewer, never a reason to skip watching the preview. Missing optional packages
(faster-whisper, opencv, opencc) only switch the corresponding check off.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .script_loader import (
    FACE_WORDS,
    ShotSpec,
    StoryboardProject,
    _resolve_path,
    clean_dialogue,
    is_detail_beat,
)

_PUNCT = re.compile(r"[\s，。、；：！？,.!?;:「」『』（）()…—\-~～\"'“”‘’《》〈〉]+")


# ---------------------------------------------------------------------- helpers
def _faces_detector():
    try:
        import cv2
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(str(path))   # OpenCV 4.x (Colab); 5.x moved it out: check is skipped
    except Exception:
        return None
    return None if det.empty() else det


def _profile_detector():
    try:
        import cv2
        path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
        det = cv2.CascadeClassifier(str(path))
    except Exception:
        return None
    return None if det.empty() else det


def count_faces(image, detector=None, *, portrait: bool = False) -> Optional[int]:
    """Frontal faces in a PIL image or numpy RGB array (None when OpenCV is not installed).

    When `portrait=True` (checking a single-character reference portrait), uses a larger relative minSize to
    suppress tiny coat-button/badge false positives, and falls back to `haarcascade_profileface.xml` when the
    character is drawn at a slight 3/4 angle.
    """
    detector = detector or _faces_detector()
    if detector is None:
        return None
    import cv2
    import numpy as np
    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    short = min(gray.shape[:2])
    ratio = 0.09 if portrait else 0.06
    size = max(24, int(short * ratio))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(size, size))
    boxes = list(faces) if len(faces) else []
    if portrait and len(boxes) > 1:
        max_w = max(int(b[2]) for b in boxes)
        boxes = [b for b in boxes if int(b[2]) >= max_w * 0.45]
    if portrait and not boxes:
        prof = _profile_detector()
        if prof is not None:
            p1 = prof.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(size, size))
            if len(p1):
                boxes = [p1[0]]
            else:
                p2 = prof.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=6, minSize=(size, size))
                if len(p2):
                    boxes = [p2[0]]
    return int(len(boxes))


_converter = None


def _to_simplified(text: str) -> str:
    """Whisper often answers in Simplified Chinese; compare both sides in one script when opencc exists."""
    global _converter
    if _converter is None:
        try:
            from opencc import OpenCC
            _converter = OpenCC("t2s")
        except Exception:                       # not installed: compare as written
            _converter = False
    return _converter.convert(text) if _converter else text


def normalise_text(text: str) -> str:
    return _to_simplified(_PUNCT.sub("", clean_dialogue(text or "")).lower())


def similarity(expected: str, heard: str) -> float:
    a, b = normalise_text(expected), normalise_text(heard)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _says_more(expected: str, heard: str) -> bool:
    """The scripted line is there, followed (or preceded) by clearly more speech than was written."""
    e, h = normalise_text(expected), normalise_text(heard)
    return bool(e) and len(h) > len(e) * 1.3 + 4


def _beat_windows(shot: ShotSpec) -> List[Tuple[float, float]]:
    out, t = [], 0.0
    for b in shot.beats:
        out.append((t, t + b.duration_sec))
        t += b.duration_sec
    return out


# ---------------------------------------------------------------------- reference pictures
def check_reference_images(project: StoryboardProject, base_dirs: Optional[Sequence[str | Path]] = None
                           ) -> List[str]:
    """Problems with the uploaded pictures that lead to duplicated people or drifting faces."""
    from PIL import Image

    dirs = [Path(b) for b in (base_dirs or []) if b]
    det = _faces_detector()
    warnings: List[str] = []
    used: Dict[str, str] = {}

    def look(raw: str, what: str, *, want_faces: Optional[int], portrait: bool) -> None:
        if not raw:
            return
        path = Path(_resolve_path(raw, dirs))
        if not path.is_file():
            return                                   # missing files are reported by the alignment report
        key = path.resolve().as_posix()
        if key in used and used[key] != what:
            warnings.append(f"{what} 和 {used[key]} 用了同一張圖 {path.name}：H3 會把它們當成同一個人/地方")
        used.setdefault(key, what)
        try:
            with Image.open(path) as im:
                w, h = im.size
                has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
                faces = count_faces(im, det, portrait=portrait) if det is not None else None
        except Exception as e:                       # unreadable picture
            warnings.append(f"{what}：{path.name} 打不開（{e}）")
            return
        if has_alpha and portrait:
            warnings.append(f"{what}：{path.name} 帶有透明背景（執行時會自動補乾淨素色灰白底，避免髮絲黑邊破圖）")
        if min(w, h) < 512:
            warnings.append(f"{what}：{path.name} 只有 {w}×{h}，太小，臉容易跑掉（建議短邊 ≥ 768）")
        if portrait and w > h * 1.25:
            warnings.append(f"{what}：{path.name} 是橫幅（{w}×{h}），像三視圖或多人合照；H3 可能把每個人像都畫成"
                            f"一個人（重複人物）。請改用只有一個人、單一角度的直幅半身照")
        if faces is None or want_faces is None:
            return
        if want_faces == 1 and faces > 1:
            warnings.append(f"{what}：{path.name} 偵測到 {faces} 張臉。角色圖只能有這個人一張臉，"
                            f"否則容易出現重複人物")
        elif want_faces == 1 and faces == 0:
            warnings.append(f"{what}：{path.name} 偵測不到正面臉，長相可能跑掉（請用正面、臉部清楚的照片）")
        elif want_faces == 0 and faces > 0:
            warnings.append(f"{what}：{path.name} 裡有 {faces} 個人。場景圖請用沒有人的空景，"
                            f"否則場景裡的人會和角色圖的人同時出現（重複人物）")

    for cid, ch in project.characters.items():
        look(ch.image_path, f"角色 {cid}", want_faces=1, portrait=True)
        if project.send_extra_views:
            for raw in ch.extra_images:
                look(raw, f"角色 {cid} 其他參考圖", want_faces=None, portrait=True)
    for sid, sc in project.scenes.items():
        look(sc.image_path, f"場景 {sid}", want_faces=0, portrait=False)
    canvas_ratio = project.width / max(1, project.height)
    for shot in project.shots:
        for raw, label in ((shot.first_frame, f"段落 {shot.shot_id} 首幀圖"),
                           (shot.last_frame, f"段落 {shot.shot_id} 尾幀圖")):
            if not raw:
                continue
            kp = Path(_resolve_path(raw, dirs))
            if not kp.is_file():
                continue
            try:
                with Image.open(kp) as kim:
                    kw, kh = kim.size
            except Exception as e:
                warnings.append(f"{label}：{kp.name} 打不開（{e}）")
                continue
            if min(kw, kh) < 512:
                warnings.append(f"{label}：{kp.name} 只有 {kw}×{kh}，解析度太低容易破圖（建議短邊 ≥ 768）")
            if abs(math.log((kw / max(1, kh)) / canvas_ratio)) > 0.28:
                warnings.append(f"{label}：{kp.name} 長寬比（{kw}×{kh}）與影片畫布（{project.width}×{project.height}）"
                                f"差距較大，生成時會自動置中裁切以避免人物被拉扁變形；建議直接用同比例關鍵幀圖")
    if det is None:
        warnings.append("（沒有可用的 OpenCV 人臉偵測，略過「圖裡有幾張臉」的檢查）")
    return warnings


# ---------------------------------------------------------------------- speech
def transcribe_words(clip: str | Path, model, prompt: str = "") -> List[Tuple[float, float, str]]:
    """[(start, end, word)] from faster-whisper (word timestamps)."""
    from .subtitles import decode_audio_16k
    segments, _ = model.transcribe(decode_audio_16k(clip), language="zh", initial_prompt=prompt[:220] or None,
                                   word_timestamps=True, vad_filter=True, beam_size=5)
    words: List[Tuple[float, float, str]] = []
    for seg in segments:
        for w in getattr(seg, "words", None) or []:
            if w.word.strip():
                words.append((float(w.start), float(w.end), w.word.strip()))
    return words


def speech_issues(project: StoryboardProject, shot: ShotSpec, words: Iterable[Tuple[float, float, str]],
                  *, min_similarity: float = 0.6, slack: float = 0.35) -> List[str]:
    """Compare what was heard in every 分鏡's time window with its scripted line."""
    words = list(words)
    if project.audio_mode != "native_h3_audio":
        return []
    beats = shot.beats
    windows = _beat_windows(shot)
    if not beats:                                   # one-cut shot: the whole clip is one window
        heard = "".join(w for _, _, w in words)
        exp = clean_dialogue(shot.dialogue)
        if exp and _says_more(exp, heard):
            return [f"{shot.shot_id}: 台詞之外還多說了話。聽到「{heard}」，腳本只有「{exp}」"]
        if exp and similarity(exp, heard) < min_similarity:
            return [f"{shot.shot_id}: 台詞不符（相似度 {similarity(exp, heard):.0%}）聽到「{heard or '（無）'}」"]
        if not exp and len(normalise_text(heard)) >= 3:
            return [f"{shot.shot_id}: 不該有人說話，卻聽到「{heard}」"]
        return []

    issues: List[str] = []
    whole = "".join(w for _, _, w in words)
    for b, (t0, t1) in zip(beats, windows):
        heard = "".join(w for s, e, w in words if t0 - slack <= (s + e) / 2 < t1 + slack)
        exp = clean_dialogue(b.dialogue)
        if exp:
            if _says_more(exp, heard):
                issues.append(f"分鏡 {b.beat_id}: 台詞之外還多說了話。聽到「{heard}」，腳本只有「{exp}」")
                continue
            sim = similarity(exp, heard)
            if sim >= min_similarity:
                continue
            if not normalise_text(heard):
                near = SequenceMatcher(None, normalise_text(exp), normalise_text(whole)).find_longest_match(
                    0, len(normalise_text(exp)), 0, len(normalise_text(whole)))
                if near.size >= max(3, len(normalise_text(exp)) // 2):
                    issues.append(f"分鏡 {b.beat_id}: 台詞時間錯位（{t0:.1f}–{t1:.1f} 秒內沒聽到，出現在別的分鏡）")
                else:
                    issues.append(f"分鏡 {b.beat_id}: 沒有聽到台詞「{exp}」")
            else:
                issues.append(f"分鏡 {b.beat_id}: 台詞不符（相似度 {sim:.0%}）聽到「{heard}」，腳本是「{exp}」")
        elif len(normalise_text(heard)) >= 3:
            issues.append(f"分鏡 {b.beat_id}: 腳本沒有台詞，卻聽到「{heard}」（多出的人聲）")
    tail = "".join(w for s, e, w in words if (s + e) / 2 >= (windows[-1][1] if windows else 0) + slack)
    if len(normalise_text(tail)) >= 3:
        issues.append(f"段落 {shot.shot_id}: 結尾（會被裁掉的部分）還在說話「{tail}」，台詞可能被切斷")
    return issues


# ---------------------------------------------------------------------- faces in the generated clip
def _frame_at(clip: Path, t: float):
    from PIL import Image
    import io
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(png)) if png else None


def face_issues(project: StoryboardProject, shot: ShotSpec, clip: str | Path, detector=None) -> List[str]:
    detector = detector or _faces_detector()
    if detector is None:
        return []
    issues: List[str] = []
    if shot.beats:
        for b, (t0, t1) in zip(shot.beats, _beat_windows(shot)):
            if is_detail_beat(b):
                continue
            img = _frame_at(Path(clip), (t0 + t1) / 2)
            if img is None:
                continue
            n = count_faces(img, detector) or 0
            want = len(b.on_screen)
            if n > want:
                issues.append(f"分鏡 {b.beat_id}: 偵測到 {n} 張臉，腳本只有 {want} 人入鏡（可能重複人物或多出路人，請確認）")
        return issues
    vis_text = " ".join((shot.title, shot.visual_prompt, shot.camera_motion)).lower()
    if any(w in vis_text for w in ("大特寫", "特寫", "extreme close", "close-up", "insert")) \
            and not any(w in (shot.visual_prompt or "").lower() for w in FACE_WORDS):
        return []
    if project.characters:
        img = _frame_at(Path(clip), max(0.2, shot.duration_sec / 2.0))
        if img is not None:
            n = count_faces(img, detector) or 0
            want = len(shot.characters)
            if n > want:
                issues.append(f"段落 {shot.shot_id}: 偵測到 {n} 張臉，腳本只有 {want} 人入鏡（可能重複人物或多出路人，請確認）")
    return issues


def visual_artifact_issues(project: StoryboardProject, shot: ShotSpec, clip: str | Path) -> List[str]:
    """Detect broken/corrupted frames, blackouts, flash spikes or severe mid-clip blur ('破圖') in a clip."""
    import numpy as np
    clip_p = Path(clip)
    if not clip_p.is_file():
        return [f"段落 {shot.shot_id}: 找不到生成片段 {clip_p.name}"]
    sample_times: List[Tuple[str, float]] = [("開頭", 0.08)]
    if shot.beats:
        for b, (t0, t1) in zip(shot.beats, _beat_windows(shot)):
            sample_times.append((f"分鏡 {b.beat_id}", (t0 + t1) / 2.0))
    else:
        sample_times.append((f"段落 {shot.shot_id} 中段", max(0.25, shot.duration_sec / 2.0)))
    sample_times.append(("結尾", max(0.15, shot.duration_sec - 0.12)))

    issues: List[str] = []
    lumas: List[Tuple[str, float]] = []
    sharpnesses: List[Tuple[str, float]] = []
    for label, ts in sample_times:
        img = _frame_at(clip_p, ts)
        if img is None:
            issues.append(f"{label}（{ts:.1f}s）無法解碼畫面（疑似檔案損毀或破圖）")
            continue
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        mean_luma = float(gray.mean())
        std_luma = float(gray.std())
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        sharp = float(np.sqrt((gx[:-1, :] ** 2 + gy[:, :-1] ** 2)).mean()) if gray.shape[0] > 2 and gray.shape[1] > 2 else 0.0
        lumas.append((label, mean_luma))
        sharpnesses.append((label, sharp))
        if mean_luma < 5.0 and std_luma < 4.0:
            issues.append(f"{label}（{ts:.1f}s）出現全黑畫面（疑似閃黑或破圖）")
        elif mean_luma > 250.0 and std_luma < 4.0:
            issues.append(f"{label}（{ts:.1f}s）出現過曝全白畫面（疑似閃白或破圖）")
    if len(sharpnesses) >= 3:
        vals = sorted(s for _, s in sharpnesses)
        median_sharp = vals[len(vals) // 2]
        if median_sharp > 8.0:
            for label, s in sharpnesses:
                if s < median_sharp * 0.25:
                    issues.append(f"{label} 清晰度驟降（疑似轉場融化或破圖，建議確認預覽）")
    return issues


# ---------------------------------------------------------------------- one batch
def load_asr_model(model_size: str = "small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("（沒有安裝 faster-whisper，略過台詞自動檢查；Step 1 勾 INSTALL_QC_TOOLS）")
        return None
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def run_batch_qc(project: StoryboardProject, shot_ids: Sequence[str], clips: Dict[str, Path],
                 out_json: Optional[str | Path] = None, *, speech: bool = True, faces: bool = True,
                 artifacts: bool = True, asr_model: str = "small") -> Dict[str, List[str]]:
    """{shot_id: [issues]} for the clips of one batch (raw H3 clips, before the tail is trimmed)."""
    model = load_asr_model(asr_model) if speech and project.audio_mode == "native_h3_audio" else None
    det = _faces_detector() if faces else None
    result: Dict[str, List[str]] = {}
    for shot in project.shots:
        if shot.shot_id not in shot_ids or shot.shot_id not in clips:
            continue
        clip = Path(clips[shot.shot_id])
        issues: List[str] = []
        if model is not None:
            prompt = "，".join(clean_dialogue(b.dialogue) for b in shot.beats if b.dialogue) or shot.dialogue
            try:
                issues += speech_issues(project, shot, transcribe_words(clip, model, prompt))
            except Exception as e:                   # never block the review on a QC failure
                issues.append(f"段落 {shot.shot_id}: 台詞檢查失敗（{e}）")
        if det is not None:
            issues += face_issues(project, shot, clip, det)
        if artifacts:
            try:
                issues += visual_artifact_issues(project, shot, clip)
            except Exception:
                pass
        result[shot.shot_id] = issues
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result

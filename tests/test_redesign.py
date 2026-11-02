"""Redesign: who is on screen per 分鏡, voice-over lines, head count, sample-exact assembly, automatic QC."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.alignment import align_project, apply_bible, load_bible, split_offscreen
from mvkit.batches import format_plan_report, plan_batches, write_batch_prompts, write_review_checklist
from mvkit.presets import apply_preset
from mvkit.qc_review import check_reference_images, similarity, speech_issues
from mvkit.script_import import import_script, import_script_text
from mvkit.script_loader import PictureRef, ShotSpec, StoryboardProject, build_h3_prompt, preview_shot

EXAMPLE = REPO_ROOT / "examples" / "osce_demo"
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")

SCRIPT = """
段落 1（0–10 秒｜參考圖）：測試
場景：病床旁
分鏡 1-1｜全景｜3 秒
- 色調打光：日光燈
- 畫面描述：考生走到病床旁
- 旁白對白：無
- 音效配樂：環境音
分鏡 1-2｜特寫｜4 秒
- 色調打光：同上
- 畫面描述：壓脈帶綁在上臂，手指檢查鬆緊
- 旁白對白：（口述）「壓脈帶下緣在肘窩上兩公分。」
- 音效配樂：魔鬼氈聲
分鏡 1-3｜中景｜3 秒
- 出場：考生
- 色調打光：同上
- 畫面描述：考生看著血壓計
- 旁白對白：（病人畫外）「會痛嗎？」
- 音效配樂：環境音
"""


def _project(tmp_path: Path, text: str = SCRIPT, images: bool = False):
    if images:
        from PIL import Image
        for n in ("scene_bedside.png", "scene_cart.png", "char_examinee.png", "char_patient.png"):
            Image.new("RGB", (600, 800), (90, 120, 150)).save(tmp_path / n)
    p = import_script_text(text, title="T")
    apply_bible(p, load_bible(EXAMPLE / "bible.md"))
    apply_preset(p, "osce_demo")
    align_project(p, [tmp_path])
    p.validate()
    return p


def test_offscreen_cues():
    assert split_offscreen("考生畫外") == ("考生", True)
    assert split_offscreen("病人OS") == ("病人", True)
    assert split_offscreen("對病人") == ("對病人", False)
    assert split_offscreen("videos") == ("videos", False)


def test_only_visible_people_get_pictures_and_close_ups_are_voice_over(tmp_path: Path):
    p = _project(tmp_path)
    b1, b2, b3 = p.shots[0].beats
    assert b1.on_screen == ["examinee"]                        # only the person the 畫面描述 names
    assert b2.on_screen == [] and b2.voice_over                # hands close-up: nobody's face, line is voice-over
    assert b3.on_screen == ["examinee"]                        # 出場：考生 wins
    assert b3.speaker == "patient" and b3.voice_over           # （病人畫外）
    assert p.shots[0].characters == ["examinee"]               # the patient only speaks off screen: no picture


def test_scene_cast_is_not_added_to_every_clip(tmp_path: Path):
    text = SCRIPT.replace("考生走到病床旁", "考生走到治療車旁").replace("（病人畫外）「會痛嗎？」", "無")
    text = text.replace("- 出場：考生\n", "").replace("場景：病床旁", "場景：治療台")
    p = _project(tmp_path, text)
    assert p.shots[0].scene_id == "cart"
    assert p.shots[0].characters == ["examinee"]               # the patient is never mentioned: not in frame


def test_prompt_states_head_count_on_screen_and_voice_over(tmp_path: Path):
    p = _project(tmp_path.joinpath(), SCRIPT.replace("考生走到病床旁", "考生走到病床旁，病人坐在床上"), images=True)
    prompt = preview_shot(p, p.shots[0], base_dirs=[tmp_path])
    assert "People in this video: exactly two people - the person in <Picture 2> (examinee doctor) and " \
           "the person in <Picture 3> (standardized patient)" in prompt
    assert "no twins, no clones" in prompt
    assert "[Shot 2] 3.0-7.0 seconds, close-up: On screen: no people, no faces." in prompt
    assert "Voice-over by examinee doctor, whose face is not visible in this shot" in prompt
    assert "Voice-over by standardized patient" in prompt
    assert "On screen: the person in <Picture 2> (examinee doctor); nobody else." in prompt
    assert "do not copy any person from it" in prompt


def test_chained_frame_is_tied_to_the_character_pictures(tmp_path: Path):
    p = _project(tmp_path)
    pics = [PictureRef("chain.png", "chain", "last frame"),
            PictureRef("a.png", "character", "examinee doctor", "examinee"),
            PictureRef("b.png", "character", "standardized patient", "patient")]
    prompt = build_h3_prompt(p, p.shots[0], pics)
    assert "fully referenced" in prompt
    assert "The people visible in <Picture 1> are the same individuals shown in the character pictures" in prompt


def test_plan_report_checklist_and_prompt_files(tmp_path: Path):
    p = _project(tmp_path, images=True)
    batch = plan_batches(p, 120)[0]
    rep = format_plan_report(p, batch, [tmp_path])
    assert "<Picture 1>=scene_bedside.png" in rep and "standardized patient（畫外）" in rep
    files = write_batch_prompts(p, batch, tmp_path / "prompts", [tmp_path])
    assert files and "# <Picture 2> =" in files[0].read_text(encoding="utf-8")
    md = write_review_checklist(p, batch, tmp_path / "c.md", qc={"01": ["分鏡 1-2: 台詞不符"]}).read_text("utf-8")
    assert "自動檢查發現" in md and "**01** ⚠️" in md and "（無人）" in md


def test_speech_qc_flags_wrong_missing_misplaced_and_extra_voices(tmp_path: Path):
    p = _project(tmp_path)
    shot = p.shots[0]
    good = [(3.4, 6.0, "壓脈帶下緣在肘窩上兩公分"), (7.4, 8.4, "會痛嗎")]
    assert speech_issues(p, shot, good) == []
    wrong = [(3.4, 6.0, "今天天氣很好我們去公園"), (7.4, 8.4, "會痛嗎")]
    assert any("台詞不符" in x for x in speech_issues(p, shot, wrong))
    misplaced = [(0.5, 2.5, "壓脈帶下緣在肘窩上兩公分"), (7.4, 8.4, "會痛嗎")]
    issues = speech_issues(p, shot, misplaced)
    assert any("分鏡 1-1" in x and "多出的人聲" in x for x in issues)
    assert any("分鏡 1-2" in x and "時間錯位" in x for x in issues)
    tail = good + [(10.1, 10.6, "還有一句話")]
    assert any("結尾" in x for x in speech_issues(p, shot, tail))
    assert similarity("「您好。」", "您好") == 1.0


def test_reference_image_checks(tmp_path: Path):
    from PIL import Image
    p = _project(tmp_path, images=True)
    Image.new("RGB", (1600, 600)).save(tmp_path / "char_examinee.png")        # turnaround-like sheet
    p.characters["patient"].image_path = "char_examinee.png"
    warnings = check_reference_images(p, [tmp_path])
    assert any("橫幅" in w for w in warnings)
    assert any("同一張圖" in w for w in warnings)


@needs_ffmpeg
def test_assembly_is_sample_exact_no_audio_drift(tmp_path: Path):
    """Old assembly: voice ran ~60 ms ahead of the picture per clip. Now every clip boundary is exact."""
    import numpy as np
    from mvkit.colab_pipeline import assemble_movie

    p = StoryboardProject(title="sync")
    p.width, p.height = 320, 192
    clips = []
    for i in range(4):
        raw = tmp_path / f"raw{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x192:r=24:d=5.375",
                        "-f", "lavfi", "-i", "sine=f=1000:sample_rate=48000:d=5.375",
                        "-vf", "drawbox=c=white:t=fill:enable='between(t,1.0,1.04)'",
                        "-af", "volume='if(between(t,1.0,1.1),1,0)':eval=frame",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(raw)], check=True)
        clips.append(raw)
        p.shots.append(ShotSpec(shot_id=f"{i:02d}", index=i + 1, duration_sec=4.5, visual_prompt="x"))
    m = assemble_movie(p, clips, tmp_path / "out", burn_subtitles=False)
    f = m["clean_mp4"]
    luma = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", "scale=8:8,format=gray", "-f",
                                         "rawvideo", "-"], capture_output=True).stdout, np.uint8).reshape(-1, 64).mean(1)
    assert len(luma) == 4 * 108
    flashes = [i / 24 for i in range(1, len(luma)) if luma[i] > 128 >= luma[i - 1]]
    pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", "48000", "-f", "s16le",
                                        "-"], capture_output=True).stdout, np.int16)
    loud = np.abs(pcm.astype(float)) > 3000
    beeps, i = [], 0
    while i < len(loud):
        if loud[i]:
            beeps.append(i / 48000)
            i += 48000
        else:
            i += 1
    assert len(flashes) == len(beeps) == 4
    assert max(abs(a - v) for a, v in zip(beeps, flashes)) < 0.02

"""Regression tests for custom AI character/scene images, anti-extra-people guards, and anti-artifact ('破圖') fixes."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.alignment import (
    align_project,
    apply_bible,
    discover_bible_from_refs,
    load_bible,
    supplement_bible_from_refs,
)
from mvkit.h3_colab_runner import H3ColabRunner, build_h3_graph, choose_models
from mvkit.presets import apply_preset
from mvkit.qc_review import check_reference_images, visual_artifact_issues
from mvkit.script_import import import_script, import_script_text
from mvkit.script_loader import (
    ShotSpec,
    StoryboardProject,
    _resolve_path,
    build_h3_prompt,
    preview_shot,
    shot_picture_plan,
)
from mvkit.segment_script import check_segments

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")


def test_user_generated_images_match_across_extensions_and_colab_duplicate_names(tmp_path: Path):
    """Bible says `.png`, user uploaded `.webp` and Colab-renamed ` (1).jpg` -> resolves cleanly."""
    Image.new("RGB", (1024, 1536), (120, 140, 160)).save(tmp_path / "char_hero (1).jpg")
    Image.new("RGB", (1344, 768), (80, 100, 120)).save(tmp_path / "SCENE_CAFE.webp")

    assert Path(_resolve_path("char_hero.png", [tmp_path])).name == "char_hero (1).jpg"
    assert Path(_resolve_path("scene_cafe.png", [tmp_path])).name == "SCENE_CAFE.webp"


def test_discover_and_supplement_bible_from_chinese_and_english_ref_filenames(tmp_path: Path):
    Image.new("RGB", (896, 1152), (100, 120, 140)).save(tmp_path / "人物_小明.webp")
    Image.new("RGB", (896, 1152), (110, 130, 150)).save(tmp_path / "char_mentor.jpg")
    Image.new("RGB", (1344, 768), (60, 80, 100)).save(tmp_path / "場景_咖啡廳.png")
    Image.new("RGB", (1344, 768), (70, 90, 110)).save(tmp_path / "kf_01_first.png")

    bible = discover_bible_from_refs(tmp_path)
    assert set(bible.characters) == {"小明", "mentor"}
    assert set(bible.scenes) == {"咖啡廳"}

    script = """
段落 1（0–10 秒｜參考圖）：相遇
場景：咖啡廳
分鏡 1-1｜全景｜5 秒
- 色調打光：暖光
- 畫面描述：小明走進咖啡廳，mentor坐在窗邊
- 旁白對白：（小明）「早安！」
- 音效配樂：室內環境音；配樂：無
分鏡 1-2｜特寫｜5 秒
- 出場：無
- 色調打光：同上
- 畫面描述：桌上的熱咖啡冒著蒸氣
- 旁白對白：無
- 音效配樂：環境音；配樂：無
"""
    p = import_script_text(script, title="Cafe")
    apply_bible(p, bible)
    supplement_bible_from_refs(p, tmp_path)
    align_project(p, [tmp_path])
    p.validate()

    plan = shot_picture_plan(p, p.shots[0], [tmp_path])
    assert [x.role for x in plan] == ["scene", "character", "character"]
    prompt = build_h3_prompt(p, p.shots[0], plan)
    assert "People in this video: exactly two people" in prompt
    assert "[Shot 2] 5.0-10.0 seconds, close-up: On screen: no people, no faces." in prompt


def test_markdown_storyboard_without_bible_never_erases_listed_characters(tmp_path: Path):
    """Previously, `_people_line` checked `if c in project.characters` and emitted 'No people appear... no faces'
    for standalone Markdown tables that had `shot.characters` but an empty `project.characters`."""
    for name in ("scene_anchor.png", "char_learner.png", "char_mentor.png"):
        Image.new("RGB", (800, 800), (90, 90, 90)).save(tmp_path / name)

    p = import_script(REPO_ROOT / "examples" / "storyboard_template.md")
    prompt = preview_shot(p, p.shots[0], base_dirs=[tmp_path])
    assert "No people appear in this video" not in prompt
    assert "People in this video: exactly two people" in prompt
    assert "<Picture 2> shows learner; keep this person's face, hair and clothing identical." in prompt


def test_one_cut_empty_room_and_hand_closeup_do_not_pull_scene_cast_or_talking_faces(tmp_path: Path):
    for n in ("scene_bedside.png", "scene_cart.png", "char_examinee.png", "char_patient.png"):
        Image.new("RGB", (800, 1000), (100, 110, 120)).save(tmp_path / n)

    script = """
分鏡 1：空景開場
場景：病床旁
角色：無
畫面：空景，安靜整潔的技能教室病床與治療車。
對話/口白：無

分鏡 2：手部特寫
場景：特寫上臂
畫面：壓脈帶綁在右上臂，手指伸入檢查鬆緊度。
對話/口白：（考生畫外）「壓脈帶下緣距離肘窩兩到三公分。」
"""
    p = import_script_text(script, title="Clean")
    apply_bible(p, load_bible(REPO_ROOT / "examples" / "osce_demo" / "bible.md"))
    apply_preset(p, "osce_demo")
    align_project(p, [tmp_path])
    p.validate()

    # Shot 1: explicit `角色：無` + `空景` -> zero character pictures sent, strict no-people guard emitted
    assert p.shots[0].characters == []
    p1 = preview_shot(p, p.shots[0], base_dirs=[tmp_path])
    assert "No people appear in this video" in p1
    assert "<Picture 2>" not in p1

    # Shot 2: off-screen voice-over + detail close-up -> voice-over prompt, no talking-head demand
    assert p.shots[1].voice_over is True
    p2 = preview_shot(p, p.shots[1], base_dirs=[tmp_path])
    assert "Voice-over by examinee doctor, whose face is not visible in this shot" in p2
    assert "Nobody on screen moves their lips." in p2


def test_anti_artifact_lanczos_scaling_rgba_compositing_and_keyframe_fit(tmp_path: Path):
    fl = choose_models("first_frame", "13.0", (8, 0))
    g_fl = build_h3_graph(
        models=fl, prompt="p", width=1344, height=768, seconds=10.0, seed=1,
        picture_names=["first.png"], keyframes={"last": "last.png"}, kept_seconds=10.0,
    )
    assert g_fl["16"]["inputs"]["upscale_method"] == "lanczos"
    assert g_fl["18"]["inputs"]["upscale_method"] == "lanczos"

    ref = choose_models("reference", "13.0", (8, 0))
    g_ref = build_h3_graph(
        models=ref, prompt="p", width=1344, height=768, seconds=10.35, seed=1,
        picture_names=["scene.png", "char.png"], keyframes={"first": "f.png", "last": "l.png"},
        kept_seconds=10.0,
    )
    # 10.0s @ 24fps -> frame 239 (the exact last kept frame before tail_guard is trimmed)
    assert g_ref["21"]["inputs"]["frame_idx"] == 239

    # RGBA transparent character PNG gets clean light-grey background (not black #000000 fringe)
    rgba_src = tmp_path / "char_alpha.png"
    rgba = Image.new("RGBA", (800, 1200), (0, 0, 0, 0))
    rgba.save(rgba_src)
    runner = H3ColabRunner(comfy_dir=tmp_path / "comfy")
    staged_ref = tmp_path / "comfy" / "input" / runner.stage_picture(rgba_src, "ref.png")
    with Image.open(staged_ref) as im:
        assert im.size == (800, 1200)
        assert im.getpixel((0, 0)) == (240, 240, 240)

    # Portrait keyframe staged with fit_size=(1344, 768) is center-cropped to 1344x768 so FL2VA never stretches it
    staged_kf = tmp_path / "comfy" / "input" / runner.stage_picture(rgba_src, "kf.png", fit_size=(1344, 768))
    with Image.open(staged_kf) as im:
        assert im.size == (1344, 768)


def test_first_last_mode_does_not_pad_tail_guard_past_last_frame():
    p = StoryboardProject()
    s_ref = ShotSpec(shot_id="01", index=1, duration_sec=10.0, visual_prompt="a", gen_mode="reference")
    s_fl = ShotSpec(shot_id="02", index=2, duration_sec=10.0, visual_prompt="b",
                    gen_mode="first_last", first_frame="f.png", last_frame="l.png")
    assert p.generation_seconds(s_ref) == pytest.approx(10.35)
    assert p.generation_seconds(s_fl) == pytest.approx(10.0)


def test_short_beat_and_mismatched_keyframe_warnings(tmp_path: Path):
    Image.new("RGB", (800, 1200), (100, 100, 100)).save(tmp_path / "kf_portrait.png")
    script = """
段落 1（0–10 秒｜首幀）：測試
場景：教室
首幀圖：kf_portrait.png
分鏡 1-1｜全景｜1.2 秒
- 色調打光：自然光
- 畫面描述：老師走進教室
- 旁白對白：無
- 音效配樂：環境音；配樂：無
分鏡 1-2｜中景｜8.8 秒
- 色調打光：同上
- 畫面描述：老師站在講台前微笑
- 旁白對白：無
- 音效配樂：環境音；配樂：無
"""
    p = import_script_text(script)
    seg_issues = check_segments(p, "free")
    assert any("1.2 秒" in x and "破圖" in x for x in seg_issues)
    ref_warns = check_reference_images(p, [tmp_path])
    assert any("長寬比" in w and "置中裁切" in w for w in ref_warns)


@needs_ffmpeg
def test_visual_artifact_qc_catches_blackout_clip(tmp_path: Path):
    black_mp4 = tmp_path / "black.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x192:r=24:d=5.0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(black_mp4)],
        check=True,
    )
    p = StoryboardProject()
    shot = ShotSpec(shot_id="01", index=1, duration_sec=5.0, visual_prompt="x")
    issues = visual_artifact_issues(p, shot, black_mp4)
    assert any("全黑畫面" in x for x in issues)

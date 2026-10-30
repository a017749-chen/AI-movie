"""Phase 1: segment scripts (10 s / 15 s), multi-shot prompts, generation modes, character tiers, draft mode."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.alignment import align_project, apply_bible, load_bible
from mvkit.batches import engines_for, plan_batches, preflight_batch
from mvkit.h3_colab_runner import H3ColabRunner, build_h3_graph, choose_models
from mvkit.presets import PRESETS, apply_preset
from mvkit.script_import import import_script, import_script_text
from mvkit.script_loader import (
    CharacterSpec,
    SceneAnchorSpec,
    ShotSpec,
    StoryboardProject,
    build_h3_prompt,
    load_storyboard_from_json,
    preview_shot,
    save_project_json,
    shot_picture_plan,
)
from mvkit.segment_script import check_segments
from mvkit.subtitles import build_cues_from_storyboard

EXAMPLE = REPO_ROOT / "examples" / "osce_demo"
PICS = ("scene_bedside.png", "scene_cart.png", "char_examinee.png", "char_patient.png",
        "kf_03_first.png", "kf_03_last.png")


def _project(refs: Path, real_images: bool = False):
    for n in PICS:
        if real_images:
            from PIL import Image
            Image.new("RGB", (64, 64), (90, 120, 150)).save(refs / n)
        elif not (refs / n).exists():
            (refs / n).write_bytes(b"png")
    p = import_script(EXAMPLE / "segments.txt", title="BP")
    apply_bible(p, load_bible(EXAMPLE / "bible.md"))
    apply_preset(p, "osce_demo")
    align_project(p, [refs])
    p.validate()
    return p


def test_segment_script_parses_beats_modes_and_keyframes(tmp_path: Path):
    p = _project(tmp_path)
    assert [s.shot_id for s in p.shots] == ["01", "02", "03"]
    assert [len(s.beats) for s in p.shots] == [3, 3, 3]
    assert [s.duration_sec for s in p.shots] == [10.0, 10.0, 10.0]
    assert [s.gen_mode for s in p.shots] == ["reference", "reference", "first_last"]
    assert p.shots[2].first_frame == "kf_03_first.png" and p.shots[2].last_frame == "kf_03_last.png"
    b = p.shots[0].beats
    assert [x.framing for x in b] == ["全景", "中景", "近景"]
    assert b[0].dialogue == "" and b[1].dialogue.startswith("王先生您好")
    assert b[1].lighting.startswith("冷白日光燈") and b[1].lighting.endswith("臉部受光均勻")   # 同上，…
    assert all(x.speaker == "examinee" for x in p.shots[1].beats)       # （對病人）/（口述）-> lead
    assert p.shots[0].characters == ["examinee", "patient"]
    assert check_segments(p, "10s_3") == []
    issues = check_segments(p, "15s_6")
    assert any("規格是 15 秒" in x for x in issues) and any("至少 6 個" in x for x in issues)


def test_spec_checker_flags_missing_modules_long_lines_and_clamps_15s():
    text = """段落 1（0–15 秒｜參考圖）
場景：教室
分鏡 1-1｜3 秒
- 畫面描述：講師站在白板前
- 旁白對白：（旁白）「這是一段非常非常非常長的旁白，三秒內一定講不完喔。」
分鏡 1-2｜中景｜12 秒
- 色調打光：暖光
- 畫面描述：講師寫字
- 旁白對白：無
- 音效配樂：粉筆聲；配樂：輕柔鋼琴
"""
    p = import_script_text(text)
    s = p.shots[0]
    assert s.duration_sec == pytest.approx(14.65)                         # 15 s keeps 14.65 s
    assert sum(b.duration_sec for b in s.beats) == pytest.approx(14.65)
    issues = check_segments(p, "15s_6")
    assert any("分鏡 1-1: 缺 景別、色調打光、音效配樂" in x for x in issues)
    assert any("台詞" in x and "超過分鏡長度" in x for x in issues)
    prompt = build_h3_prompt(p, s, [])
    assert "non_diegetic_music: 輕柔鋼琴" in prompt


def test_multi_shot_prompt_timing_framing_and_opening(tmp_path: Path):
    p = _project(tmp_path)
    prompt = preview_shot(p, p.shots[0], base_dirs=[tmp_path])
    assert "integrated_multimodal_description: Setting:" in prompt               # assets first
    assert "[Shot 1] 0.0-3.0 seconds, full shot:" in prompt
    assert "[Shot 2] 3.0-7.0 seconds, medium shot:" in prompt
    assert "[Shot 3] 7.0-10.0 seconds, medium close-up:" in prompt
    assert "The person in <Picture 2> (examinee doctor), voice:" in prompt
    assert "No one speaks in this shot." in prompt
    assert "fully referenced" in prompt                                          # opens on a full shot
    # a segment opening on a close-up uses the scene picture as a place reference only
    p.shots[1].beats[0].framing = "特寫"
    prompt2 = preview_shot(p, p.shots[1], base_dirs=[tmp_path])
    assert "not as the opening frame" in prompt2 and "fully referenced" not in prompt2


def test_picture_budget_follows_character_tiers(tmp_path: Path):
    p = StoryboardProject()
    p.scenes["hall"] = SceneAnchorSpec("hall", "lecture hall", "hall.png")
    (tmp_path / "hall.png").write_bytes(b"x")
    ids = []
    for i in range(10):
        tier = "main" if i == 9 else ("npc" if i < 5 else "secondary")
        cid = f"c{i}"
        ids.append(cid)
        (tmp_path / f"{cid}.png").write_bytes(b"x")
        extras = [f"{cid}_turn.png"] if tier == "main" else []
        for e in extras:
            (tmp_path / e).write_bytes(b"x")
        p.characters[cid] = CharacterSpec(cid, f"person {i}", image_path=f"{cid}.png", tier=tier, extra_images=extras)
    shot = ShotSpec("01", 1, scene_id="hall", visual_prompt="everyone", characters=ids)
    plan = shot_picture_plan(p, shot, [tmp_path])
    chars = [x.char_id for x in plan if x.role == "character"]
    assert len(plan) == 9 and "c9" in chars                       # main kept
    assert set(chars) == {"c5", "c6", "c7", "c8", "c9", "c0", "c1", "c2"}   # npc c3, c4 fall back to words
    prompt = build_h3_prompt(p, shot, plan)
    assert "person 3." in prompt and "person 4." in prompt
    # fewer characters: the main character's turnaround takes a free slot and points at the main picture
    shot2 = ShotSpec("02", 2, scene_id="hall", visual_prompt="two people", characters=["c0", "c9"])
    plan2 = shot_picture_plan(p, shot2, [tmp_path])
    assert [x.role for x in plan2] == ["scene", "character", "character", "extra"]
    assert "<Picture 4> shows another view of person 9 (the same person as <Picture 3>)" in build_h3_prompt(p, shot2, plan2)


def test_bible_tiers_and_extra_images_columns():
    b = load_bible(EXAMPLE / "bible.md")
    assert b.characters["examinee"].tier == "main" and b.characters["patient"].tier == "secondary"
    assert b.characters["examinee"].image_path == "char_examinee.png"        # not "其他參考圖"
    assert b.characters["examinee"].extra_images == ["char_examinee_turnaround.png", "char_examinee_face.png"]


def test_graphs_for_keyframe_modes():
    ref = choose_models("reference", "13.0", (8, 0))
    g = build_h3_graph(models=ref, prompt="p", width=1344, height=768, seconds=10.35, seed=1,
                       picture_names=["a.png", "b.png"], keyframes={"first": "f.png", "last": "l.png"})
    assert g["20"]["class_type"] == "MiniMaxH3AddGuide" and g["20"]["inputs"]["frame_idx"] == 0
    assert g["21"]["inputs"]["frame_idx"] == -1 and g["21"]["inputs"]["positive"] == ["20", 0]
    assert g["8"]["inputs"]["conditioning"] == ["21", 0]
    fl = choose_models("first_frame", "13.0", (8, 0))
    g2 = build_h3_graph(models=fl, prompt="p", width=1344, height=768, seconds=10.35, seed=1,
                        picture_names=["first.png"], keyframes={"last": "l.png"})
    assert g2["7"]["inputs"]["last_frame"] == ["18", 0] and g2["17"]["inputs"]["image"] == "l.png"
    assert g2["8"]["inputs"]["conditioning"] == ["7", 0]


def test_engines_preflight_and_snapshot(tmp_path: Path):
    p = _project(tmp_path)
    batches = plan_batches(p, 120)
    assert engines_for(p, batches[0]) == ["first_frame", "reference"]
    assert preflight_batch(p, batches[0], [tmp_path]) == []
    (tmp_path / "kf_03_last.png").unlink()
    assert any("尾幀圖 kf_03_last.png 找不到" in x for x in preflight_batch(p, batches[0], [tmp_path]))
    q = load_storyboard_from_json(json.loads(save_project_json(p, tmp_path / "p.json").read_text(encoding="utf-8")))
    assert [len(s.beats) for s in q.shots] == [3, 3, 3] and q.shots[2].gen_mode == "first_last"
    assert q.shots[0].beats[1].speaker == "examinee"


def test_beat_subtitles_stay_inside_their_beat(tmp_path: Path):
    p = _project(tmp_path)
    cues = build_cues_from_storyboard(p)
    first = [c for c in cues if c.text.startswith("王先生")][0]
    assert 3.0 <= first.start_sec and first.end_sec <= 7.0
    assert not any(c.start_sec < 3.0 for c in cues)        # 分鏡 1-1 has no line


def test_presets_carry_segment_rules():
    assert PRESETS["osce_demo"]["segment_rule"] == "10s_3"
    assert PRESETS["recruitment"]["segment_rule"] == "15s_6"


def test_notebook_has_quality_and_segment_rule():
    nb = json.loads((REPO_ROOT / "AI_Movie_Maker_Colab.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'QUALITY = "draft"' in code and "SEGMENT_RULE" in code and "engines_for(project, batch)" in code


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")


@needs_ffmpeg
def test_runner_mixes_modes_and_chains_only_on_matching_framing(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    p = _project(refs, real_images=True)
    p.shots[1].beats[0].framing = "近景"          # segment 1 ends 近景 -> segment 2 now opens 近景: chains
    runner = H3ColabRunner(comfy_dir=tmp_path / "comfy")
    runner.models = choose_models("reference", "13.0", (8, 0))
    graphs = []

    def gen(graph):
        frames = graph["7"]["inputs"]["length"]
        out = tmp_path / f"g{len(graphs)}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x184:rate=24",
                        "-f", "lavfi", "-i", "sine=sample_rate=48000", "-t", f"{frames / 24:.3f}", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
        graphs.append(graph)
        return out
    runner.generate_fn = gen

    with pytest.raises(ValueError, match="first_frame"):
        runner.run_storyboard(p, tmp_path / "work", base_dirs=[refs])
    runner.model_sets["first_frame"] = choose_models("first_frame", "13.0", (8, 0))
    runner.run_storyboard(p, tmp_path / "work", base_dirs=[refs])
    prog = json.loads((tmp_path / "work" / "progress.json").read_text(encoding="utf-8"))
    assert [prog[s]["mode"] for s in ("01", "02", "03")] == ["reference", "reference", "first_last"]
    assert prog["02"]["chained"] and not prog["03"]["chained"]          # 03 has its own first frame
    assert graphs[0]["7"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graphs[2]["7"]["class_type"] == "MiniMaxH3ImageToVideo" and "last_frame" in graphs[2]["7"]["inputs"]
    p3 = (tmp_path / "work" / "prompts" / "03_03.txt").read_text(encoding="utf-8")
    assert "<Picture 2> (from [Shot 3]) is fully referenced" in p3

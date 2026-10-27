"""Script upload, character/scene alignment and the 2-minute review gate."""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.alignment import align_project, apply_bible, format_alignment_report, load_bible
from mvkit.batches import (
    approve_batch,
    assemble_batch_preview,
    batch_status,
    chained_dependents,
    check_gate,
    next_batch,
    plan_batches,
    preflight_batch,
    request_redo,
    shot_fingerprint,
)
from mvkit.h3_colab_runner import H3ColabRunner, choose_models
from mvkit.presets import apply_preset
from mvkit.script_import import (
    import_script,
    import_script_text,
    parse_freeform_script,
    read_script_text,
    split_dialogue,
    split_long_line,
)
from mvkit.script_loader import H3_MAX_SECONDS, load_storyboard_from_json, preview_shot, save_project_json

EXAMPLE = REPO_ROOT / "examples" / "osce_demo"


def _osce_project(tmp_path: Path, with_images: bool = True):
    if with_images:
        for n in ("scene_bedside.png", "scene_cart.png", "char_examinee.png", "char_patient.png"):
            if not (tmp_path / n).exists():
                (tmp_path / n).write_bytes(b"png")
    p = import_script(EXAMPLE / "script.txt", title="BP")
    apply_bible(p, load_bible(EXAMPLE / "bible.md"))
    apply_preset(p, "osce_demo")
    report = align_project(p, [tmp_path])
    p.validate()
    return p, report


def test_freeform_script_fields_speaker_cues_and_scoring():
    drafts = parse_freeform_script((EXAMPLE / "script.txt").read_text(encoding="utf-8"))
    assert [d.number for d in drafts] == ["1", "2", "3", "4", "5"]
    assert drafts[0].act.startswith("第一幕")
    assert drafts[0].speaker_cue == "對病人" and drafts[1].speaker_cue == "口述"
    assert drafts[0].dialogue.startswith("王先生您好") and "「" not in drafts[0].dialogue
    assert "確認病人身分" in drafts[0].scoring
    assert split_dialogue("考生：好，我們開始。") == ("好，我們開始。", "考生")


def test_docx_and_ai_studio_json_are_read(tmp_path: Path):
    body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in
                   ("分鏡 1：開場", "場景：病床旁", "畫面：考生向病人點頭。", "對話/口白：（對病人）「您好。」"))
    doc = tmp_path / "s.docx"
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                   f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    p = import_script(doc)
    assert len(p.shots) == 1 and p.shots[0].dialogue == "您好。" and p.shots[0].speaker == "對病人"

    studio = tmp_path / "studio_export"
    studio.write_text(json.dumps({"veoPrompt": {"prompt": "分鏡 1：A\n畫面：B\n對話/口白：「C。」"}}), encoding="utf-8")
    assert "分鏡 1" in read_script_text(studio)
    assert import_script(studio).shots[0].visual_prompt == "B"


def test_long_lines_split_into_chained_clips():
    line = "這是一段非常長的教學旁白，用來說明整個操作的每一個細節。" * 4
    pieces = split_long_line(line, H3_MAX_SECONDS - 0.35)
    assert len(pieces) > 1 and "".join(pieces) == line
    p = import_script_text(f"分鏡 3：長說明\n畫面：講師說明。\n旁白：{line}")
    assert [s.shot_id for s in p.shots][:2] == ["03a", "03b"]
    assert p.shots[1].continuity_override == "chain_last_frame"
    assert all(s.duration_sec <= H3_MAX_SECONDS - 0.35 for s in p.shots)
    p.validate()


def test_markdown_table_still_goes_to_table_loader():
    p = import_script(REPO_ROOT / "examples" / "storyboard_template.md")
    assert len(p.shots) == 4 and p.shots[0].shot_id == "01_opening"


def test_bible_alignment_scenes_characters_and_speakers(tmp_path: Path):
    p, report = _osce_project(tmp_path)
    scenes = [s.scene_id for s in p.shots]
    assert scenes == ["bedside", "cart", "bedside", "bedside", "bedside"]   # 特寫 shots stay at the bed
    assert "特寫上臂" in p.shots[2].camera_motion
    assert all(s.speaker == "examinee" for s in p.shots)                   # （口述）/（對病人） -> lead
    assert p.shots[0].characters == ["examinee", "patient"]                # bible order, stable <Picture n>
    assert p.shots[1].characters == ["examinee"]
    assert not [w for r in report for w in r.warnings]
    assert "| 01 |" in format_alignment_report(p, report)

    prompt = preview_shot(p, p.shots[0], base_dirs=[tmp_path])
    assert "<Picture 2> shows examinee doctor" in prompt and "<Picture 3> shows standardized patient" in prompt
    assert "The person in <Picture 2> (examinee doctor), voice: a calm, clear young male voice" in prompt
    assert "Only examinee doctor speaks" in prompt
    assert "OSCE examination room" in prompt


def test_missing_pictures_are_reported_before_the_gpu(tmp_path: Path):
    p, report = _osce_project(tmp_path, with_images=False)
    text = format_alignment_report(p, report)
    assert "場景 bedside 沒有參考圖" in text and "角色 examinee 沒有參考圖" in text
    batch = plan_batches(p, 120)[0]
    problems = preflight_batch(p, batch, [tmp_path])
    assert problems and problems[0].startswith("01")


def test_project_json_snapshot_round_trip(tmp_path: Path):
    p, _ = _osce_project(tmp_path)
    f = save_project_json(p, tmp_path / "project.json")
    q = load_storyboard_from_json(json.loads(f.read_text(encoding="utf-8")))
    assert q.characters["examinee"].lead and q.characters["examinee"].voice
    assert q.scenes["bedside"].cast == ["examinee", "patient"] and q.preset == "osce_demo"
    assert [shot_fingerprint(q, s) for s in q.shots] == [shot_fingerprint(p, s) for s in p.shots]


def test_batches_respect_length_and_cover_every_shot(tmp_path: Path):
    p, _ = _osce_project(tmp_path)
    for limit in (120, 30, 20, 9):
        batches = plan_batches(p, limit)
        assert [i for b in batches for i in b.shot_ids] == [s.shot_id for s in p.shots]
        for b in batches:
            assert b.seconds <= limit + 1e-6 or len(b.shot_ids) == 1
    assert len(plan_batches(p, 120)) == 1
    # 01(bedside) 02(cart) 03..: with 20 s, the first batch stops at the scene change after 01+02 = 18.5 s
    assert plan_batches(p, 20)[0].shot_ids == ["01", "02"]
    assert chained_dependents(p, "03") == ["04", "05"]


# ---------------------------------------------------------------------- gate, end to end with fake H3
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")


def _fake_generate(tmp: Path):
    calls = []

    def gen(graph):
        frames = graph["7"]["inputs"]["length"]
        out = tmp / f"fake_{len(calls):02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x184:rate=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", f"{frames / 24:.3f}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
        calls.append(graph)
        return out
    return gen, calls


@needs_ffmpeg
def test_review_gate_generates_one_batch_at_a_time(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    from PIL import Image
    for n in ("scene_bedside.png", "scene_cart.png", "char_examinee.png", "char_patient.png"):
        Image.new("RGB", (64, 64), (90, 120, 150)).save(refs / n)
    p, _ = _osce_project(refs)
    for s in p.shots:                       # keep the test fast
        s.duration_sec = 5.0
    work = tmp_path / "work"
    batches = plan_batches(p, 12)           # 3 batches: [01,02] [03,04] [05]
    assert [b.shot_ids for b in batches] == [["01", "02"], ["03", "04"], ["05"]]

    runner = H3ColabRunner(comfy_dir=tmp_path / "comfy")
    runner.models = choose_models("reference", "13.0", (8, 0))
    runner.generate_fn, calls = _fake_generate(tmp_path)

    b1 = next_batch(p, work, batches)
    assert b1.index == 1
    clips = runner.run_storyboard(p, work, base_dirs=[refs], shot_ids=set(b1.shot_ids))
    assert len(calls) == 2 and len(clips) == 2
    assert batch_status(p, work, b1) == "generated"
    with pytest.raises(RuntimeError, match="尚未確認"):
        check_gate(p, work, batches, batches[1])

    preview = assemble_batch_preview(p, b1, work, burn_subtitles=False)
    assert preview["clean_stream_verification"]["audio_stream"]
    assert Path(preview["checklist"]).read_text(encoding="utf-8").count("☐ 通過") == 2

    # redo shot 02 only: new seed, old take kept, nothing else regenerated
    old_seed = p.shots[1].seed
    request_redo(p, work, ["02"])
    assert p.shots[1].seed != old_seed
    assert (work / "raw_clips" / "_rejected" / "02_02_take1.mp4").is_file()
    runner.run_storyboard(p, work, base_dirs=[refs], shot_ids=set(b1.shot_ids))
    assert len(calls) == 3 and calls[-1]["9"]["inputs"]["noise_seed"] == p.shots[1].seed
    approve_batch(p, work, b1)
    assert batch_status(p, work, b1) == "approved"

    b2 = next_batch(p, work, batches)
    check_gate(p, work, batches, b2)
    runner.run_storyboard(p, work, base_dirs=[refs], shot_ids=set(b2.shot_ids))
    assert len(calls) == 5
    # shot 03 is bedside after 02 at the cart: starts from the bedside anchor, not chained
    assert not json.loads((work / "progress.json").read_text(encoding="utf-8"))["03"]["chained"]
    assert json.loads((work / "progress.json").read_text(encoding="utf-8"))["04"]["chained"]

    # editing an approved shot re-opens it and regenerates only that shot
    p.shots[0].visual_prompt += " The examinee smiles."
    assert batch_status(p, work, b1) != "approved"
    runner.run_storyboard(p, work, base_dirs=[refs], shot_ids=set(b1.shot_ids))
    assert len(calls) == 6

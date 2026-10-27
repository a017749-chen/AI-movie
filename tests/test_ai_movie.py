import ast
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.h3_colab_runner import OUTPUT_NODE, build_h3_graph, choose_models
from mvkit.script_loader import (
    PictureRef,
    build_h3_prompt,
    check_picture_tags,
    h3_frames,
    load_storyboard,
    load_storyboard_from_markdown,
    preview_shot,
    shot_picture_plan,
)
from mvkit.subtitles import build_cues_from_storyboard, split_chinese_clauses, write_ass, write_srt


def test_colab_notebook_is_valid_and_matches_upstream_comfy():
    nb_path = REPO_ROOT / "AI_Movie_Maker_Colab.ipynb"
    assert nb_path.exists(), "AI_Movie_Maker_Colab.ipynb must exist at repo root"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) == 5
    full_code = ""
    for cell in code_cells:
        src = "".join(cell.get("source", []))
        full_code += src + "\n"
        py_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(py_lines))
    assert "https://github.com/Comfy-Org/ComfyUI.git" in full_code
    assert "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" in full_code
    assert "runner.free_vram()" in full_code


def test_h3_frames_matches_upstream_17k_plus_5_rule():
    # Upstream notebook examples: 4s -> 107, 5s -> 124, 12s -> 294, 15s -> 362
    assert h3_frames(4.0) == 107
    assert h3_frames(5.0) == 124
    assert h3_frames(12.0) == 294
    assert h3_frames(15.0) == 362


def test_load_markdown_json_and_mvkit_project(tmp_path: Path):
    # Create dummy reference images in tmp_path so shot_picture_plan attaches them
    for name in ("scene_anchor.png", "char_learner.png", "char_mentor.png"):
        (tmp_path / name).write_bytes(b"png")

    md_proj = load_storyboard(REPO_ROOT / "examples" / "storyboard_template.md")
    md_proj.validate()
    assert len(md_proj.shots) == 4
    assert md_proj.width == 1344 and md_proj.height == 768
    p0 = preview_shot(md_proj, md_proj.shots[0], base_dirs=[tmp_path])
    assert "For the target video, at 0.00 seconds into the target video, <Picture 1>" in p0
    assert "<Picture 2>" in p0 and "<Picture 3>" in p0
    assert "<d>[Chinese]" in p0

    json_proj = load_storyboard(REPO_ROOT / "examples" / "storyboard_template.json")
    json_proj.validate()
    assert len(json_proj.shots) == 4
    plan = shot_picture_plan(json_proj, json_proj.shots[0], base_dirs=[tmp_path])
    assert len(plan) == 3
    p_json = build_h3_prompt(json_proj, json_proj.shots[0], plan)
    check_picture_tags(p_json, len(plan))
    assert "<Picture 2> shows young female clinician" in p_json
    assert "<Picture 3> shows senior male clinician" in p_json

    # Also test autoip-mv-maker project.py loader
    mv_proj = load_storyboard(REPO_ROOT / "examples" / "demo" / "project.py")
    mv_proj.validate()
    assert mv_proj.audio_mode == "master_audio"
    assert len(mv_proj.shots) > 0
    assert len(mv_proj.subtitle_cues) == 8


def test_single_audio_track_prompt_enforcement(tmp_path: Path):
    anchor = tmp_path / "scene_anchor.png"
    anchor.write_bytes(b"png")
    md_text = (REPO_ROOT / "examples" / "storyboard_template.md").read_text(encoding="utf-8")

    # 1. native_h3_audio includes <d>[Chinese] ...</d>
    p_native = load_storyboard_from_markdown(md_text, audio_mode="native_h3_audio")
    prompt_native = preview_shot(p_native, p_native.shots[0], base_dirs=[tmp_path])
    assert "<d>[Chinese]" in prompt_native
    assert "The same single narrator voice in every clip." in prompt_native

    # 2. clip_voiceover forbids H3 speech so external TTS never overlaps with H3 speech
    p_vo = load_storyboard_from_markdown(md_text, audio_mode="clip_voiceover")
    prompt_vo = preview_shot(p_vo, p_vo.shots[0], base_dirs=[tmp_path])
    assert "<d>" not in prompt_vo
    assert "Nobody speaks: no dialogue, no narration" in prompt_vo

    # 3. master_audio forbids H3 speech
    p_master = load_storyboard_from_markdown(md_text, audio_mode="master_audio")
    prompt_master = preview_shot(p_master, p_master.shots[0], base_dirs=[tmp_path])
    assert "<d>" not in prompt_master
    assert "Nobody speaks: no dialogue, no narration" in prompt_master


def test_comfyui_h3_graph_matches_real_nodes():
    m_ref = choose_models("reference", "13.0", (8, 0))
    assert m_ref.diffusion == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert m_ref.steps == 4
    g_ref = build_h3_graph(
        models=m_ref,
        prompt="For the target video...",
        width=1344,
        height=768,
        seconds=8.0,
        seed=12345,
        picture_names=["h3_reference_01.png", "h3_reference_02.png"],
        ref_image_size="max",
    )
    assert g_ref["3"]["inputs"]["type"] == "minimax"
    assert g_ref["7"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert g_ref["7"]["inputs"]["ref_images.ref_image_0"] == ["100", 0]
    assert g_ref["7"]["inputs"]["ref_images.ref_image_1"] == ["101", 0]
    assert g_ref[OUTPUT_NODE]["class_type"] == "VHS_VideoCombine"

    m_fl = choose_models("first_frame", "12.4", (8, 0))
    assert m_fl.diffusion == "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    assert m_fl.steps == 8
    g_fl = build_h3_graph(
        models=m_fl,
        prompt="For the target video...",
        width=1344,
        height=768,
        seconds=8.0,
        seed=12345,
        picture_names=["h3_colab_first_frame.png"],
    )
    assert g_fl["7"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert g_fl["16"]["class_type"] == "ImageScaleToTotalPixels"


def test_subtitle_generation_srt_and_ass(tmp_path: Path):
    proj = load_storyboard(REPO_ROOT / "examples" / "storyboard_template.json")
    cues = build_cues_from_storyboard(proj)
    assert len(cues) >= 4
    for c1, c2 in zip(cues, cues[1:]):
        assert c1.end_sec <= c2.start_sec + 1e-6

    srt_file = write_srt(cues, tmp_path / "out.zh-TW.srt")
    ass_file = write_ass(cues, tmp_path / "out.zh-TW.ass")
    assert srt_file.exists() and "-->" in srt_file.read_text(encoding="utf-8")
    assert ass_file.exists() and "Dialogue:" in ass_file.read_text(encoding="utf-8")
    assert split_chinese_clauses("第一句話，第二句話。第三句話！")

    # Also verify MV absolute lyric cues
    mv_proj = load_storyboard(REPO_ROOT / "examples" / "demo" / "project.py")
    mv_cues = build_cues_from_storyboard(mv_proj)
    assert len(mv_cues) == 8


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    test_colab_notebook_is_valid_and_matches_upstream_comfy()
    test_h3_frames_matches_upstream_17k_plus_5_rule()
    test_load_markdown_json_and_mvkit_project(tmp)
    test_single_audio_track_prompt_enforcement(tmp)
    test_comfyui_h3_graph_matches_real_nodes()
    test_subtitle_generation_srt_and_ass(tmp)
    print("ALL TESTS PASSED!")

import ast
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.h3_colab_runner import build_comfyui_h3_workflow
from mvkit.script_loader import (
    load_storyboard,
    load_storyboard_from_markdown,
)
from mvkit.subtitles import build_cues_from_storyboard, split_chinese_clauses, write_ass, write_srt


def test_colab_notebook_is_valid_and_parsable():
    nb_path = REPO_ROOT / "AI_Movie_Maker_Colab.ipynb"
    assert nb_path.exists(), "AI_Movie_Maker_Colab.ipynb must exist at repo root"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5
    for cell in code_cells:
        src = "".join(cell.get("source", []))
        py_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(py_lines))


def test_load_markdown_and_json_storyboards():
    md_proj = load_storyboard(REPO_ROOT / "examples" / "storyboard_template.md")
    assert len(md_proj.shots) == 4
    assert md_proj.shots[0].shot_id == "01_opening"
    assert "<d>[Chinese]" in md_proj.shots[0].compiled_h3_prompt
    assert "<Picture 1>" in md_proj.shots[0].compiled_h3_prompt

    json_proj = load_storyboard(REPO_ROOT / "examples" / "storyboard_template.json")
    assert len(json_proj.shots) == 4
    assert "learner" in json_proj.characters
    assert "mentor" in json_proj.characters
    assert "<Picture 2>" in json_proj.shots[0].compiled_h3_prompt
    assert "<Picture 3>" in json_proj.shots[0].compiled_h3_prompt


def test_single_audio_track_prompt_enforcement():
    md_text = (REPO_ROOT / "examples" / "storyboard_template.md").read_text(encoding="utf-8")

    # 1. native_h3_audio includes <d>[Chinese] ...</d>
    p_native = load_storyboard_from_markdown(md_text, audio_mode="native_h3_audio")
    assert "<d>[Chinese]" in p_native.shots[0].compiled_h3_prompt

    # 2. clip_voiceover forbids H3 speech so external TTS never overlaps with H3 speech
    p_vo = load_storyboard_from_markdown(md_text, audio_mode="clip_voiceover")
    assert "<d>" not in p_vo.shots[0].compiled_h3_prompt
    assert "外部旁白替換模式" in p_vo.shots[0].compiled_h3_prompt

    # 3. master_audio strips clip audio (-an) and omits <d> dialogue tags
    p_master = load_storyboard_from_markdown(md_text, audio_mode="master_audio")
    assert "<d>" not in p_master.shots[0].compiled_h3_prompt
    assert "-an" in p_master.shots[0].compiled_h3_prompt


def test_comfyui_h3_workflow_builder():
    wf = build_comfyui_h3_workflow(
        prompt="測試防跑掉提示詞",
        mode="reference",
        aspect_ratio="16:9",
        duration_sec=8.0,
        fps=24,
        ref_image_size="max",
        reference_images_in_comfy=["scene_anchor.png", "char_learner.png"],
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3_ref2va_480p_turbo.safetensors"
    assert wf["5"]["inputs"]["ref_image_size"] == "max"
    assert wf["5"]["inputs"]["ref_image_1"] == ["101", 0]
    assert wf["5"]["inputs"]["ref_image_2"] == ["102", 0]


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


if __name__ == "__main__":
    test_colab_notebook_is_valid_and_parsable()
    test_load_markdown_and_json_storyboards()
    test_single_audio_track_prompt_enforcement()
    test_comfyui_h3_workflow_builder()
    test_subtitle_generation_srt_and_ass(Path(tempfile.mkdtemp()))
    print("ALL TESTS PASSED!")

"""Only dialogue is voiced and subtitled: background notes and loose lines never leak into it."""
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.script_import import import_script, import_script_text
from mvkit.script_loader import SPEECH_GUARD, build_h3_prompt, spoken_text_report
from mvkit.segment_script import check_segments
from mvkit.subtitles import build_cues_from_storyboard


def test_segment_background_and_loose_lines_are_not_dialogue():
    p = import_script(REPO_ROOT / "tests" / "fixture_stray_lines.txt")
    b1, b2 = p.shots[0].beats
    assert b1.dialogue == "您好，我是醫師。"
    assert b2.dialogue == "請問您的名字？我姓王。" and b2.speaker == "考生"
    assert "明亮的 OSCE 技能教室" in b1.visual and "神情疲倦" in b1.visual      # picture context only
    assert "語氣溫和" in p.shots[0].notes and "語氣溫和" not in b2.visual
    spoken = spoken_text_report(p)
    for leaked in ("技能教室", "神情疲倦", "名牌", "語氣溫和"):
        assert leaked not in spoken
        assert not any(leaked in c.text for c in build_cues_from_storyboard(p))
    assert any("一位說話者" in x for x in check_segments(p, "free"))
    assert SPEECH_GUARD in build_h3_prompt(p, p.shots[0], [])


def test_freeform_background_after_dialogue_is_not_spoken():
    q = import_script_text("分鏡 1：開場\n場景：病床旁\n畫面：考生洗手\n對話/口白：（口述）「我先洗手。」\n"
                           "背景設定：冬天的早晨\n沒有標籤的補充說明\n動作與評分重點：洗手")
    s = q.shots[0]
    assert s.dialogue == "我先洗手。" and s.subtitle_text == "我先洗手。"
    assert "冬天的早晨" in s.visual_prompt and "冬天" not in spoken_text_report(q)


def test_multiline_quotes_still_continue_dialogue():
    q = import_script_text("分鏡 1：說明\n畫面：講師說明\n旁白：「第一句還沒說完，\n第二行接著說。」")
    assert q.shots[0].dialogue == "第一句還沒說完，第二行接著說。"


def test_notebook_prints_spoken_text():
    nb = json.loads((REPO_ROOT / "AI_Movie_Maker_Colab.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][3]["source"])
    ast.parse(src)
    assert "spoken_text_report(project)" in src

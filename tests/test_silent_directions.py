"""Instruction sentences never reach the H3 prompt, short lines are followed by stated silence, and the
speech check notices words that are not in the script.

Seen on a real run: every 分鏡 had `音效：同一位畫外旁白只說本鏡台詞一次；畫面人物保持安靜；不要額外音效`. The
narrator finished the scripted line, then read that sentence aloud for the rest of the clip.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvkit.alignment import align_project, apply_bible, parse_bible_markdown
from mvkit.qc_review import speech_issues
from mvkit.script_import import import_script_text
from mvkit.script_loader import (QUIET_AFTER, build_h3_prompt, prompt_hygiene_report, speech_seconds,
                                 split_directions)
from mvkit.subtitles import build_cues_from_storyboard

SCRIPT = """
段落 1（0–14 秒｜參考圖）：開場
場景：診間
分鏡 1-1｜全景｜14 秒
- 出場：醫師、病人
- 色調打光：冷白柔和燈光
- 畫面描述：醫師站在左側，病人坐在右側。
- 旁白對白：（旁白）「下背痛合併腿部放射症狀時，先聽清楚疼痛路徑和神經症狀，再用左右比較的神經學檢查定位。」
- 音效：僅低音量診間底噪；同一位畫外旁白只說本鏡台詞一次。畫面中的兩位人物保持安靜；不要其他人聲、重疊、回音或音效。
- 配樂：無

段落 2（14–24 秒｜參考圖）：同意
場景：診間
分鏡 2-1｜中景｜10 秒
- 出場：醫師、病人
- 色調打光：同上
- 畫面描述：病人點頭。
- 旁白對白：（病人）「好，可以。」
- 音效：只由病人於鏡頭開頭說出「好，可以。」一次；說完後兩人保持安靜；衣物摩擦聲
- 配樂：無
"""

BIBLE = """
## 角色
| 代號 | 名稱 | 別名 | 外觀 | 參考圖 | 位置 | 聲音 | 主講 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| doctor | doctor | 醫師 | One Taiwanese man in his twenties, white coat. Use the front reference only; never duplicate him or add another student. | | Follow the staging in each shot; do not force a bedside position. | a calm young male voice | 是 |
| patient | patient | 病人 | Taiwanese man, age 48, navy shirt. | | on the chair, right of frame | a gentle adult male voice | |

## 場景
| 代號 | 名稱 | 別名 | 描述 | 參考圖 | 角色 |
| --- | --- | --- | --- | --- | --- |
| clinic | clinic | 診間 | Modern exam room with an examination bed under soft white light. Follow the shot-specific staging. Show only the people named; no added staff, duplicate or cloned people. No readable text or watermarks. | | |
"""


def _project():
    p = import_script_text(SCRIPT, audio_mode="native_h3_audio")
    apply_bible(p, parse_bible_markdown(BIBLE))
    align_project(p, [])
    return p


def test_split_directions_keeps_descriptions_only():
    kept, dropped = split_directions("Modern exam room with a bed. Follow the staging. No readable text.")
    assert kept == "Modern exam room with a bed." and len(dropped) == 2
    kept, dropped = split_directions("技能教室環境音、腳步聲", clauses=True)
    assert kept == "技能教室環境音、腳步聲" and not dropped
    kept, dropped = split_directions("只由病人說出「好，可以。」一次；衣物摩擦聲", clauses=True)
    assert kept == "衣物摩擦聲" and "可以" not in kept


def test_instruction_sentences_never_reach_the_prompt():
    p = _project()
    for shot in p.shots:
        prompt = build_h3_prompt(p, shot, [])
        for leaked in ("只說本鏡台詞", "保持安靜", "不要其他人聲", "說出", "Follow the", "never duplicate",
                       "added staff", "No readable text", "do not force", "音效："):
            assert leaked not in prompt, leaked
        assert "watermarks.." not in prompt and "light.." not in prompt
    first = build_h3_prompt(p, p.shots[0], [])
    assert "Setting: Modern exam room with an examination bed under soft white light." in first
    assert "底噪" not in first and "Sound:" not in first          # room tone is in overall_soundscape
    assert "white coat" in first and "on the chair, right of frame" in first
    second = build_h3_prompt(p, p.shots[1], [])
    assert "Sound: 衣物摩擦聲." in second and second.index("Sound: 衣物摩擦聲.") < second.index("<d>")


def test_short_line_ends_early_and_silence_is_stated():
    p = _project()
    assert 9.0 <= speech_seconds(p.shots[0].beats[0].dialogue) <= 10.5       # the real clip took 9.5 s
    short = build_h3_prompt(p, p.shots[1], [])
    assert "finishing by 2.0 seconds" in short and QUIET_AFTER.strip() in short
    long_ = build_h3_prompt(p, p.shots[0], [])
    assert "finishing by 11.5 seconds" in long_ and QUIET_AFTER.strip() in long_
    cue = [c for c in build_cues_from_storyboard(p) if c.shot_id == p.shots[1].shot_id][0]
    assert cue.end_sec - cue.start_sec < 3.0                                  # not the whole 10 s 分鏡


def test_step3_report_names_what_was_dropped_and_the_dead_air():
    report = "\n".join(prompt_hygiene_report(_project()))
    assert "場景 clinic" in report and "角色 doctor" in report and "角色 patient" not in report
    assert "分鏡 1-1" in report and "只說本鏡台詞一次" in report
    assert "分鏡 2-1：台詞約" in report and "建議縮到約 5 秒" in report


def test_speech_check_flags_words_beyond_the_script():
    p = _project()
    shot = p.shots[0]
    line = "下背痛合併腿部放射症狀時先聽清楚疼痛路徑和神經症狀再用左右比較的神經學檢查定位"
    extra = "畫面中的兩位人物同一位畫外旁白只說本鏡台詞一次"
    clean = [(0.2, 9.5, line)]
    assert speech_issues(p, shot, clean) == []
    issues = speech_issues(p, shot, clean + [(9.6, 13.8, extra)])
    assert len(issues) == 1 and "多說了話" in issues[0]

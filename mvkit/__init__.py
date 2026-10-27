"""mvkit: Unified AI Movie & MV Maker (autoip-mv-maker + Colab MiniMax H3-Turbo)."""
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from .script_loader import (
    CharacterSpec,
    SceneAnchorSpec,
    ShotSpec,
    StoryboardProject,
    compile_shot_prompt,
    load_storyboard,
    load_storyboard_from_json,
    load_storyboard_from_markdown,
    load_storyboard_from_mvkit_project,
    resolve_shot_references,
)
from .h3_colab_runner import (
    H3ColabRunner,
    build_comfyui_h3_workflow,
    extract_clean_continuity_frame,
)
from .subtitles import (
    SubtitleCue,
    align_cues_with_asr,
    build_cues_from_storyboard,
    burn_subtitles_into_video,
    split_chinese_clauses,
    write_ass,
    write_srt,
)
from .colab_pipeline import (
    assemble_movie,
    build_contact_sheet,
    normalize_shot_clip,
    probe_media_streams,
    verify_single_audio_stream,
)

__all__ = [
    "CharacterSpec",
    "SceneAnchorSpec",
    "ShotSpec",
    "StoryboardProject",
    "compile_shot_prompt",
    "load_storyboard",
    "load_storyboard_from_json",
    "load_storyboard_from_markdown",
    "load_storyboard_from_mvkit_project",
    "resolve_shot_references",
    "H3ColabRunner",
    "build_comfyui_h3_workflow",
    "extract_clean_continuity_frame",
    "SubtitleCue",
    "align_cues_with_asr",
    "build_cues_from_storyboard",
    "burn_subtitles_into_video",
    "split_chinese_clauses",
    "write_ass",
    "write_srt",
    "assemble_movie",
    "build_contact_sheet",
    "normalize_shot_clip",
    "probe_media_streams",
    "verify_single_audio_stream",
]

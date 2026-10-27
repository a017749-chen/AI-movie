"""Look-and-sound presets for the three main uses: OSCE demonstrations, teaching videos, recruitment videos.

A preset only fills the project-wide prompt text (style, continuity, negatives, narrator, soundscape) and a
suggested continuity mode; everything per shot still comes from the script and the bible.
"""
from __future__ import annotations

from typing import Dict

from .script_loader import StoryboardProject

PRESETS: Dict[str, Dict[str, str]] = {
    "osce_demo": {
        "label": "OSCE 示範影片",
        "style_prefix": (
            "Photorealistic clinical skills training video in a hospital OSCE examination room, even neutral "
            "lighting, clean realistic medical equipment, every procedural step performed slowly and correctly, "
            "hands and instruments clearly visible, stable faces, restrained camera motion."),
        "continuity_rules": (
            "Keep every person's face, hairstyle, uniform, gloves and name badge, the bed, the instrument tray "
            "and the room layout, their screen positions and the lighting exactly consistent with the reference "
            "pictures. Medical devices keep the same shape and colour in every shot."),
        "negative_rules": (
            "No flicker, no face changes, no extra people, anatomically correct hands with five fingers, no "
            "unsafe or incorrect clinical technique, no blood unless described, no captions, no logos, no "
            "generated text, labels or watermarks."),
        "narrator": "A calm, clear off-screen instructor (S1) speaks in standard Taiwan Mandarin at a measured pace",
        "overall_soundscape": "Quiet clinical skills room tone, soft rustle of gloves and equipment.",
        "continuity_mode": "hybrid",
    },
    "teaching": {
        "label": "教學影片",
        "style_prefix": (
            "Photorealistic medical teaching video, bright soft key light, clean uncluttered background, "
            "speakers framed in steady medium shots, natural gestures, stable faces, restrained camera motion."),
        "continuity_rules": (
            "Keep every person's face, hairstyle, clothing and props, the room layout, screen positions and "
            "lighting exactly consistent with the reference pictures."),
        "negative_rules": (
            "No flicker, no face changes, no extra people, anatomically correct hands, no captions, no logos, "
            "no generated text or watermarks."),
        "narrator": "A warm, clear off-screen teacher (S1) speaks in standard Taiwan Mandarin at a measured pace",
        "overall_soundscape": "Quiet indoor room tone with soft natural movement sounds.",
        "continuity_mode": "hybrid",
    },
    "recruitment": {
        "label": "招生影片",
        "style_prefix": (
            "Photorealistic cinematic recruitment film for a medical training programme, warm golden light, "
            "shallow depth of field, confident and friendly people, smooth slow camera moves, stable faces, "
            "natural body movement."),
        "continuity_rules": (
            "Keep every person's face, hairstyle, clothing and props, the location, screen positions and the "
            "colour grade exactly consistent with the reference pictures."),
        "negative_rules": (
            "No flicker, no face changes, no extra people, anatomically correct hands, no captions, no logos, "
            "no generated text or watermarks."),
        "narrator": "An upbeat, sincere off-screen narrator (S1) speaks in standard Taiwan Mandarin",
        "overall_soundscape": "Light ambient hospital atmosphere, gentle footsteps.",
        "continuity_mode": "anchor",
    },
}


def apply_preset(project: StoryboardProject, name: str, *, keep_continuity_mode: bool = False) -> StoryboardProject:
    """Fill the project-wide prompt text from a preset. `name` in ("", "none") leaves the project unchanged."""
    if not name or name == "none":
        return project
    if name not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)} or 'none'")
    pr = PRESETS[name]
    for key in ("style_prefix", "continuity_rules", "negative_rules", "narrator", "overall_soundscape"):
        setattr(project, key, pr[key])
    if not keep_continuity_mode:
        project.continuity_mode = pr["continuity_mode"]
    project.preset = name
    return project

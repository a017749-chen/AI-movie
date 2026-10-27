"""Colab ComfyUI + MiniMax H3-Turbo Runner with Visual Continuity & Frame Chaining.

Integrates the workflow from `MiniMax_H3_Turbo_Colab.ipynb` (`1zdxzAm834HesHG2uu7GbM4Kv0kbKOLLu`)
with `autoip-mv-maker` storyboard jobs:
- Supports `reference` (`Ref2VA`, up to 7 `<Picture k>` references), `first_frame` (`FL2VA`), and `text_only` (`T2VA`).
- Uses `ref_image_size="max"` by default to preserve high-resolution facial/clothing details.
- Prevents visual drift across shots via `continuity_mode`:
  - `"anchor"`: every shot uses the locked master scene + character reference sheets.
  - `"hybrid"`: shots within the same `scene_id` chain the last clean frame (taken BEFORE `tail_guard_sec`)
    as `<Picture 1>` while keeping character sheets in `<Picture 2..N>`; scene cuts reset to the clean anchor.
  - `"chain_last_frame"`: always chains the last clean frame from shot N-1 into shot N.
- Supports Google Drive model caching (`/content/drive/MyDrive/ComfyUI_MiniMax_H3/models`) and checkpoint resume (`progress.json`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .script_loader import ShotSpec, StoryboardProject, compile_shot_prompt, resolve_shot_references

H3_HF_REPO = "MiniMaxAI/H3-Turbo"
H3_MODEL_FILES = {
    "diffusion_models": {
        "ref2va": ("diffusion_models/h3_ref2va_480p_turbo_fp8_e4m3fn_scaled.safetensors", "h3_ref2va_480p_turbo.safetensors"),
        "fl2va": ("diffusion_models/h3_fl2va_720p_turbo_fp8_e4m3fn_scaled.safetensors", "h3_fl2va_720p_turbo.safetensors"),
        "t2va": ("diffusion_models/h3_t2va_720p_turbo_fp8_e4m3fn_scaled.safetensors", "h3_t2va_720p_turbo.safetensors"),
    },
    "text_encoders": ("text_encoders/qwen_3_8b_fp8mixed.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
    "vae": (
        ("vae/video_vae_bf16.safetensors", "video_vae_bf16.safetensors"),
        ("vae/audio_vae_bf16.safetensors", "audio_vae_bf16.safetensors"),
    ),
}


def extract_clean_continuity_frame(
    video_path: str | Path,
    output_png: str | Path,
    *,
    tail_guard_sec: float = 0.35,
) -> Path:
    """Extract the last stable frame BEFORE `tail_guard_sec` from `video_path` for shot-to-shot chaining."""
    video_path = Path(video_path)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float((proc.stdout or "5.0").strip())
    target_sec = max(0.1, round(duration - max(0.1, tail_guard_sec) - 0.08, 3))

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{target_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=in_range=tv:out_range=pc",
        str(output_png),
    ]
    subprocess.run(cmd, check=True)
    if not output_png.exists() or output_png.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract continuity frame from {video_path} at {target_sec}s")
    return output_png


def build_comfyui_h3_workflow(
    *,
    prompt: str,
    mode: str = "reference",
    aspect_ratio: str = "16:9",
    duration_sec: float = 6.0,
    fps: int = 24,
    cfg: float = 4.0,
    steps: int = 8,
    seed: int = 1001,
    ref_image_size: str = "max",
    reference_images_in_comfy: Optional[List[str]] = None,
    first_frame_in_comfy: str = "",
    last_frame_in_comfy: str = "",
    filename_prefix: str = "h3_shot",
) -> Dict[str, Any]:
    """Build the ComfyUI API prompt dictionary matching `MiniMax_H3_Turbo_Colab.ipynb`."""
    reference_images_in_comfy = list(reference_images_in_comfy or [])
    fps = int(fps)
    raw_frames = max(int(round(duration_sec * fps)), 17)
    num_frames = ((raw_frames - 1) // 4) * 4 + 1

    if mode == "reference" and reference_images_in_comfy:
        unet_name = "h3_ref2va_480p_turbo.safetensors"
        resolution = "480p"
    elif mode == "first_frame" and first_frame_in_comfy:
        unet_name = "h3_fl2va_720p_turbo.safetensors"
        resolution = "720p"
    else:
        mode = "text_only"
        unet_name = "h3_t2va_720p_turbo.safetensors"
        resolution = "720p"

    wf: Dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "Stable_Diffusion"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "video_vae_bf16.safetensors"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "audio_vae_bf16.safetensors"},
        },
    }

    encoder_inputs: Dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
    }

    if mode == "reference":
        encoder_inputs["video_vae"] = ["3", 0]
        encoder_inputs["ref_image_size"] = ref_image_size
        for i, img_name in enumerate(reference_images_in_comfy[:7], start=1):
            load_id = f"10{i}"
            wf[load_id] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
            encoder_inputs[f"ref_image_{i}"] = [load_id, 0]

    wf["5"] = {
        "class_type": "H3TextEncoder",
        "inputs": encoder_inputs,
    }

    latent_inputs: Dict[str, Any] = {
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
        "fps": fps,
    }
    if mode == "first_frame" and first_frame_in_comfy:
        wf["201"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_in_comfy}}
        latent_inputs["video_vae"] = ["3", 0]
        latent_inputs["first_frame"] = ["201", 0]
        if last_frame_in_comfy:
            wf["202"] = {"class_type": "LoadImage", "inputs": {"image": last_frame_in_comfy}}
            latent_inputs["last_frame"] = ["202", 0]

    wf["6"] = {"class_type": "H3LatentVideoAudio", "inputs": latent_inputs}
    wf["7"] = {
        "class_type": "H3Sampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "video_latent": ["6", 0],
            "audio_latent": ["6", 1],
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": int(seed),
        },
    }
    wf["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
    }
    wf["9"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["7", 1], "vae": ["4", 0]},
    }
    wf["10"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["8", 0],
            "audio": ["9", 0],
            "fps": float(fps),
        },
    }
    wf["11"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["10", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": "h264",
        },
    }
    return wf


class H3ColabRunner:
    """Manages ComfyUI + MiniMax H3 execution, reference staging, and continuity chaining."""

    def __init__(
        self,
        comfy_dir: str | Path = "/content/ComfyUI",
        server_url: str = "http://127.0.0.1:8188",
        drive_models_dir: Optional[str | Path] = "/content/drive/MyDrive/ComfyUI_MiniMax_H3/models",
    ) -> None:
        self.comfy_dir = Path(comfy_dir)
        self.server_url = server_url.rstrip("/")
        self.drive_models_dir = Path(drive_models_dir) if drive_models_dir else None

    def ensure_models(self, mode: str = "reference") -> None:
        """Link models from Google Drive if present, or download missing weights from HuggingFace."""
        from huggingface_hub import hf_hub_download

        models_root = self.comfy_dir / "models"
        for sub in ("diffusion_models", "text_encoders", "vae"):
            (models_root / sub).mkdir(parents=True, exist_ok=True)

        needed: List[tuple[str, str, str]] = [
            ("text_encoders", H3_MODEL_FILES["text_encoders"][0], H3_MODEL_FILES["text_encoders"][1]),
            ("vae", H3_MODEL_FILES["vae"][0][0], H3_MODEL_FILES["vae"][0][1]),
            ("vae", H3_MODEL_FILES["vae"][1][0], H3_MODEL_FILES["vae"][1][1]),
        ]
        if mode == "reference":
            needed.append(("diffusion_models", *H3_MODEL_FILES["diffusion_models"]["ref2va"]))
        elif mode == "first_frame":
            needed.append(("diffusion_models", *H3_MODEL_FILES["diffusion_models"]["fl2va"]))
        else:
            needed.append(("diffusion_models", *H3_MODEL_FILES["diffusion_models"]["t2va"]))

        for subdir, hf_path, local_name in needed:
            target = models_root / subdir / local_name
            if target.exists() and target.stat().st_size > 1024 * 1024:
                continue

            if self.drive_models_dir and (self.drive_models_dir / subdir / local_name).exists():
                drive_src = self.drive_models_dir / subdir / local_name
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(drive_src, target)
                print(f"[H3Runner] Linked cached model from Google Drive: {drive_src}")
                continue

            print(f"[H3Runner] Downloading {hf_path} -> {target} ...")
            downloaded = hf_hub_download(repo_id=H3_HF_REPO, filename=hf_path)
            if self.drive_models_dir and self.drive_models_dir.parent.exists():
                drive_dst = self.drive_models_dir / subdir / local_name
                drive_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, drive_dst)
                if target.exists():
                    target.unlink()
                os.symlink(drive_dst, target)
                print(f"[H3Runner] Cached to Google Drive & linked: {drive_dst}")
            else:
                shutil.copy2(downloaded, target)

    def stage_input_image(self, src_path: str | Path, dest_name: Optional[str] = None) -> str:
        """Copy an image into `/content/ComfyUI/input/` and return its filename."""
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Reference image not found: {src}")
        input_dir = self.comfy_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        name = dest_name or src.name
        dst = input_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return name

    def submit_and_wait(self, workflow: Dict[str, Any], timeout_sec: int = 1800) -> Path:
        """Queue a workflow on ComfyUI and wait until the MP4 output file is written."""
        payload = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prompt_id = data["prompt_id"]

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}", timeout=30) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                for node_out in outputs.values():
                    for key in ("videos", "gifs", "images"):
                        for item in node_out.get(key, []):
                            fname = item.get("filename")
                            sub = item.get("subfolder", "")
                            if fname and fname.endswith(".mp4"):
                                out_path = self.comfy_dir / "output" / sub / fname
                                if out_path.exists():
                                    return out_path
                raise RuntimeError(f"ComfyUI finished prompt {prompt_id} without an MP4 output: {outputs}")
            time.sleep(3.0)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id} after {timeout_sec}s")

    def run_storyboard(
        self,
        project: StoryboardProject,
        output_dir: str | Path,
        *,
        base_dir: Optional[Path] = None,
        resume: bool = True,
    ) -> List[Path]:
        """Run all shots in `project` sequentially with anti-drift continuity chaining and checkpoint resume."""
        project.validate()
        out_root = Path(output_dir)
        raw_clips_dir = out_root / "raw_clips"
        chained_refs_dir = out_root / "chained_refs"
        raw_clips_dir.mkdir(parents=True, exist_ok=True)
        chained_refs_dir.mkdir(parents=True, exist_ok=True)

        progress_file = out_root / "progress.json"
        completed: Dict[str, str] = {}
        if resume and progress_file.exists():
            try:
                completed = json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                completed = {}

        self.ensure_models(mode=project.inference_mode)

        generated_clips: List[Path] = []
        prev_clip_path: Optional[Path] = None
        prev_scene_id: Optional[str] = None

        for shot in project.shots:
            clip_target = raw_clips_dir / f"{shot.index:02d}_{shot.shot_id}.mp4"
            if resume and shot.shot_id in completed and clip_target.exists() and clip_target.stat().st_size > 0:
                print(f"[H3Runner] Skipping already completed shot {shot.shot_id} -> {clip_target}")
                generated_clips.append(clip_target)
                prev_clip_path = clip_target
                prev_scene_id = shot.scene_id
                continue

            # Resolve base reference images for this shot
            base_refs = resolve_shot_references(project, shot, base_dir=base_dir)
            effective_continuity = shot.continuity_override or project.continuity_mode

            should_chain = (
                prev_clip_path is not None
                and prev_clip_path.exists()
                and (
                    effective_continuity == "chain_last_frame"
                    or (effective_continuity == "hybrid" and prev_scene_id == shot.scene_id)
                )
            )

            staged_refs: List[str] = []
            first_frame_staged = ""

            if should_chain and prev_clip_path is not None:
                chain_png = chained_refs_dir / f"chain_from_{shot.index - 1:02d}_to_{shot.index:02d}.png"
                extract_clean_continuity_frame(
                    prev_clip_path,
                    chain_png,
                    tail_guard_sec=project.tail_guard_sec,
                )
                staged_chain = self.stage_input_image(chain_png, dest_name=chain_png.name)
                if project.inference_mode == "reference":
                    # Replace <Picture 1> (scene/axis anchor) with the chained frame, keep <Picture 2..N> character sheets!
                    staged_refs.append(staged_chain)
                    for idx_r, r_path in enumerate(base_refs[1:], start=2):
                        if Path(r_path).exists():
                            staged_refs.append(
                                self.stage_input_image(r_path, dest_name=f"shot{shot.index:02d}_ref{idx_r}_{Path(r_path).name}")
                            )
                elif project.inference_mode == "first_frame":
                    first_frame_staged = staged_chain
            else:
                if project.inference_mode == "reference":
                    for idx_r, r_path in enumerate(base_refs, start=1):
                        if Path(r_path).exists():
                            staged_refs.append(
                                self.stage_input_image(r_path, dest_name=f"shot{shot.index:02d}_ref{idx_r}_{Path(r_path).name}")
                            )
                elif project.inference_mode == "first_frame":
                    ff_src = shot.first_frame_image or (base_refs[0] if base_refs else "")
                    if ff_src and Path(ff_src).exists():
                        first_frame_staged = self.stage_input_image(
                            ff_src, dest_name=f"shot{shot.index:02d}_ff_{Path(ff_src).name}"
                        )

            prompt_text = shot.compiled_h3_prompt or compile_shot_prompt(project, shot)
            # Generate slightly extra duration so tail_guard trimming still leaves full target duration
            gen_duration = round(shot.duration_sec + project.tail_guard_sec, 2)

            wf = build_comfyui_h3_workflow(
                prompt=prompt_text,
                mode=project.inference_mode,
                aspect_ratio=project.aspect_ratio,
                duration_sec=gen_duration,
                fps=project.fps,
                cfg=project.cfg,
                steps=project.steps,
                seed=shot.seed,
                ref_image_size=project.ref_image_size,
                reference_images_in_comfy=staged_refs,
                first_frame_in_comfy=first_frame_staged,
                filename_prefix=f"shot_{shot.index:02d}_{shot.shot_id}",
            )

            print(
                f"[H3Runner] Generating shot {shot.index}/{len(project.shots)} ({shot.shot_id}) | "
                f"mode={project.inference_mode} | chain={should_chain} | refs={staged_refs}"
            )
            raw_out = self.submit_and_wait(wf)
            shutil.copy2(raw_out, clip_target)

            completed[shot.shot_id] = str(clip_target)
            progress_file.write_text(json.dumps(completed, indent=2, ensure_ascii=False), encoding="utf-8")

            generated_clips.append(clip_target)
            prev_clip_path = clip_target
            prev_scene_id = shot.scene_id

        return generated_clips

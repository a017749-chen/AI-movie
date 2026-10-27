"""Run a storyboard on Colab: ComfyUI + MiniMax H3 (Ref2VA or FL2VA), one clip per storyboard row.

All models are downloaded directly into Colab's local disk (`/content/ComfyUI/models`), matching
`MiniMax_H3_Turbo_Colab.ipynb` (no Google Drive model storage or slow Drive I/O):

  UNETLoader -> LoraLoaderModelOnly -> BasicGuider
  CLIPLoader(type="minimax") + VAELoader(video) + VAELoader(audio)
  MiniMaxH3ReferenceToVideo (Ref2VA, ref_images.ref_image_0..8)  |  MiniMaxH3ImageToVideo (FL2VA, first_frame)
  RandomNoise + KSamplerSelect + BasicScheduler -> SamplerCustomAdvanced
  VAEDecode + VAEDecodeAudio -> VHS_VideoCombine (h264 mp4 with ONE audio stream)

Continuity between clips:
  anchor           every clip starts from the locked scene picture.
  hybrid           inside the same scene_id the previous clip's last clean frame (before tail_guard) becomes
                   <Picture 1>; character sheets stay as <Picture 2..>. A new scene_id resets to the anchor.
  chain_last_frame always chain from the previous clip.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional

from .batches import shot_fingerprint
from .script_loader import (
    PictureRef,
    ShotSpec,
    StoryboardProject,
    build_h3_prompt,
    check_picture_tags,
    h3_frames,
    shot_picture_plan,
)

BASE_REPO = "Comfy-Org/MiniMax-H3"
FL2VA_LORA_REPO = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
OUTPUT_NODE = "15"


@dataclass
class ModelSet:
    mode: str
    variant: str
    diffusion: str
    video_vae: str
    lora: str
    lora_repo: str
    lora_remote: str
    steps: int

    def files(self) -> List[tuple]:
        """(repo, remote path, models sub-folder) for every file the graph loads."""
        return [
            (BASE_REPO, "diffusion_models/" + self.diffusion, "diffusion_models"),
            (BASE_REPO, "text_encoders/" + TEXT_ENCODER, "text_encoders"),
            (BASE_REPO, "vae/" + self.video_vae, "vae"),
            (BASE_REPO, "vae/" + AUDIO_VAE, "vae"),
            (self.lora_repo, self.lora_remote, "loras"),
        ]


def choose_models(mode: str, cuda_version: str, capability: tuple) -> ModelSet:
    """Same rules as the upstream notebook: INT8 ConvRot needs CUDA 13, FP8 needs sm_89+, else BF16."""
    cuda_major = int((cuda_version or "0").split(".", 1)[0] or 0)
    if mode == "reference":
        if cuda_major >= 13 and capability >= (8, 0):
            variant = "int8_convrot"
        elif capability >= (8, 9):
            variant = "fp8_scaled"
        else:
            variant = "bf16"
        return ModelSet(
            mode, variant, f"minimax_h3_ref2va_pruned_{variant}.safetensors",
            "minimax_h3_video_vae_int8_convrot.safetensors" if variant == "int8_convrot"
            else "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", BASE_REPO,
            "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 4)
    variant = "int8_convrot" if cuda_major >= 13 else "fp8_scaled"
    lora = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
    return ModelSet(mode, variant, f"minimax_h3_fl2va_pruned_{variant}.safetensors",
                    "minimax_h3_video_vae_fp16.safetensors", lora, FL2VA_LORA_REPO, lora, 8)


def build_h3_graph(*, models: ModelSet, prompt: str, width: int, height: int, seconds: float, seed: int,
                   picture_names: List[str], ref_image_size: str = "max", lora_strength: float = 1.0,
                   sampler: str = "euler", scheduler: str = "beta", crf: int = 18,
                   filename_prefix: str = "ai_movie") -> Dict[str, Any]:
    """ComfyUI API graph for one clip (mirrors the upstream notebook's graph)."""
    frames = h3_frames(seconds)
    g: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": models.diffusion, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": models.lora, "strength_model": lora_strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": models.video_vae}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    if models.mode == "reference":
        if not 1 <= len(picture_names) <= 9:
            raise ValueError("Ref2VA needs 1-9 pictures")
        inputs: Dict[str, Any] = {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": prompt,
                                  "width": width, "height": height, "length": frames,
                                  "ref_image_size": ref_image_size}
        for i, name in enumerate(picture_names):
            nid = str(100 + i)
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
        g["7"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    else:
        if len(picture_names) != 1:
            raise ValueError("FL2VA needs exactly one first-frame picture")
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": picture_names[0]}}
        g["16"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["6", 0], "upscale_method": "nearest-exact", "megapixels": 0.98, "resolution_steps": 32}}
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "first_frame": ["16", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames}}
    g.update({
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "11": {"class_type": "BasicScheduler",
               "inputs": {"model": ["2", 0], "scheduler": scheduler, "steps": models.steps, "denoise": 1.0}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["8", 0], "sampler": ["10", 0], "sigmas": ["11", 0],
            "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        OUTPUT_NODE: {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["13", 0], "audio": ["14", 0], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": filename_prefix, "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": crf,
            "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True}},
    })
    return g


def extract_clean_continuity_frame(video_path: str | Path, output_png: str | Path, *,
                                   tail_guard_sec: float = 0.35) -> Path:
    """Last frame BEFORE the tail guard (H3 tails often morph or flash), saved as PNG."""
    video_path, output_png = Path(video_path), Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(video_path)], capture_output=True, text=True, check=True).stdout.strip()
    t = max(0.0, float(out) - max(0.0, tail_guard_sec) - 1.0 / 24)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1",
                    str(output_png)], check=True)
    if not output_png.is_file() or output_png.stat().st_size == 0:
        raise RuntimeError(f"could not extract a continuity frame from {video_path} at {t:.2f}s")
    return output_png


class H3ColabRunner:
    """Drives a ComfyUI server on the Colab VM with models stored directly in `/content/ComfyUI/models`."""

    def __init__(self, comfy_dir: str | Path = "/content/ComfyUI", base_url: str = "http://127.0.0.1:8188",
                 models: Optional[ModelSet] = None,
                 log_path: str | Path = "/content/comfyui_ai_movie.log") -> None:
        self.comfy_dir = Path(comfy_dir)
        self.base_url = base_url.rstrip("/")
        self.models = models
        self.log_path = Path(log_path)
        self.process: Optional[subprocess.Popen] = None
        self.generate_fn: Callable[..., Path] = self._generate_with_comfy

    # ------------------------------------------------------------------ models
    def detect_models(self, mode: str) -> ModelSet:
        import torch
        assert torch.cuda.is_available(), "Colab: Runtime > Change runtime type > GPU (A100 recommended)"
        self.models = choose_models(mode, torch.version.cuda or "0", torch.cuda.get_device_capability(0))
        return self.models

    def ensure_models(self) -> Dict[str, str]:
        """Download required H3 model weights directly into `/content/ComfyUI/models`."""
        from huggingface_hub import hf_hub_download

        assert self.models, "call detect_models() first"
        models_root = self.comfy_dir / "models"
        found: Dict[str, str] = {}
        for repo, remote, folder in self.models.files():
            name = Path(remote).name
            target = models_root / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size == 0:
                print(f"downloading {repo}/{remote} -> {target}")
                local_dir = models_root if repo == BASE_REPO else models_root / folder
                downloaded = Path(hf_hub_download(repo_id=repo, filename=remote, local_dir=str(local_dir)))
                assert downloaded == target, f"unexpected download path: {downloaded} != {target}"
            found[name] = str(target)
            print(f"  {folder:16s} {name}  {target.stat().st_size / 2**30:.2f} GiB")
        return found

    # ------------------------------------------------------------------ server
    def _get(self, path: str, timeout: float = 10) -> Any:
        import requests
        r = requests.get(self.base_url + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def start_server(self, low_vram: str = "auto", timeout_sec: int = 360) -> None:
        import requests
        try:
            self._get("/system_stats", 3)
            print("ComfyUI already running")
        except Exception:
            launch = [sys.executable, str(self.comfy_dir / "main.py"), "--listen", "127.0.0.1", "--port",
                      self.base_url.rsplit(":", 1)[-1], "--disable-auto-launch"]
            if low_vram == "on":
                launch.append("--lowvram")
            elif low_vram == "auto":
                import torch
                if torch.cuda.get_device_properties(0).total_memory < 48 * 2**30:
                    launch.append("--lowvram")
            log = open(self.log_path, "a")
            self.process = subprocess.Popen(launch, cwd=self.comfy_dir, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout_sec
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("ComfyUI exited:\n" + self.log_path.read_text(errors="replace")[-4000:])
                try:
                    self._get("/system_stats", 3)
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"ComfyUI did not start; see {self.log_path}")
                    time.sleep(2)
        info = self._get("/object_info", 60)
        need = {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "LoadImage", "BasicGuider",
                "RandomNoise", "KSamplerSelect", "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode",
                "VAEDecodeAudio", "VHS_VideoCombine"}
        need |= {"MiniMaxH3ReferenceToVideo"} if self.models and self.models.mode == "reference" else \
            {"MiniMaxH3ImageToVideo", "ImageScaleToTotalPixels"}
        missing = need - set(info)
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {sorted(missing)}; update ComfyUI / VideoHelperSuite "
                               f"and check {self.log_path}")
        self.object_info = info
        print("ComfyUI ready, all H3 nodes present")

    def free_vram(self) -> None:
        import requests
        try:
            requests.post(self.base_url + "/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception:
            pass

    def stage_picture(self, src: str | Path, name: str) -> str:
        from PIL import Image
        dst = self.comfy_dir / "input" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.convert("RGB").save(dst)
        return dst.name

    def _generate_with_comfy(self, graph: Dict[str, Any], timeout_sec: float = 7200) -> Path:
        import requests
        r = requests.post(self.base_url + "/prompt", json={"prompt": graph, "client_id": "ai-movie"}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"ComfyUI rejected the graph (HTTP {r.status_code}): {r.text[:2000]}")
        job = r.json()
        if job.get("node_errors"):
            raise RuntimeError("node errors: " + json.dumps(job["node_errors"], ensure_ascii=False)[:2000])
        pid = job["prompt_id"]
        deadline, last = time.monotonic() + timeout_sec, 0.0
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("ComfyUI exited:\n" + self.log_path.read_text(errors="replace")[-4000:])
            hist = self._get(f"/history/{pid}", 30)
            if pid in hist:
                run = hist[pid]
                status = run.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError("generation failed: " + json.dumps(status, ensure_ascii=False)[:3000])
                items = run.get("outputs", {}).get(OUTPUT_NODE, {}).get("gifs", [])
                mp4 = next((i for i in items if i["filename"].endswith(".mp4")), None)
                if mp4:
                    return self.comfy_dir / "output" / mp4.get("subfolder", "") / mp4["filename"]
                if status.get("status_str") == "success":
                    raise RuntimeError("finished without an MP4: " + json.dumps(run.get("outputs"))[:2000])
            if time.monotonic() - last > 60:
                print("   ...still generating", flush=True)
                last = time.monotonic()
            time.sleep(5)
        raise TimeoutError(f"prompt {pid} took longer than {timeout_sec}s")

    # ------------------------------------------------------------------ storyboard
    def plan_shot(self, project: StoryboardProject, shot: ShotSpec, prev_clip: Optional[Path],
                  prev_scene: Optional[str], work_dir: Path, base_dirs: List[Path]) -> tuple:
        """Return (pictures, chained) for a shot, applying the continuity mode."""
        plan = shot_picture_plan(project, shot, base_dirs)
        mode = shot.continuity_override or project.continuity_mode
        chain = prev_clip is not None and prev_clip.is_file() and (
            mode == "chain_last_frame" or (mode == "hybrid" and prev_scene == shot.scene_id))
        if chain:
            png = extract_clean_continuity_frame(prev_clip, work_dir / "chain_frames" / f"{shot.index:02d}_start.png",
                                                 tail_guard_sec=project.tail_guard_sec)
            start = PictureRef(str(png), "chain", "the last frame of the previous clip")
            rest = [p for p in plan if p.role == "character"]
            plan = [start] + rest
        if not plan:
            raise ValueError(f"shot {shot.shot_id}: no picture found. Upload a scene/first-frame picture or set "
                             f"reference_images; H3 Ref2VA/FL2VA both need at least one image")
        if project.inference_mode == "first_frame":
            plan = plan[:1]
        return plan[:9], chain

    def run_storyboard(self, project: StoryboardProject, work_dir: str | Path, *,
                       base_dirs: Optional[List[str | Path]] = None, resume: bool = True,
                       only: Optional[List[str]] = None,
                       shot_ids: Optional[Collection[str]] = None) -> List[Path]:
        """Generate every shot in order. Finished clips are recorded in progress.json and skipped on resume.

        `only` regenerates the listed shot ids (their old clips are replaced).
        `shot_ids` limits the run to one review batch: other shots are never generated, but their existing
        clips still feed last-frame chaining, and only the batch's clips are returned.
        A clip whose shot was edited (text, seed, pictures, bible entry) since it was made is regenerated.
        """
        project.validate()
        assert self.models, "call detect_models() first"
        if self.models.mode != project.inference_mode:
            raise ValueError(f"models were prepared for {self.models.mode!r} but storyboard uses "
                             f"{project.inference_mode!r}; re-run the model cell")
        work = Path(work_dir)
        clips_dir = work / "raw_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        dirs = [Path(b) for b in (base_dirs or [])]
        progress_file = work / "progress.json"
        progress: Dict[str, Any] = json.loads(progress_file.read_text(encoding="utf-8")) \
            if resume and progress_file.is_file() else {}

        clips: List[Path] = []
        prev_clip: Optional[Path] = None
        prev_scene: Optional[str] = None
        for shot in project.shots:
            clip = clips_dir / f"{shot.index:02d}_{shot.shot_id}.mp4"
            if shot_ids is not None and shot.shot_id not in shot_ids:
                have = clip.is_file() and clip.stat().st_size > 0
                prev_clip, prev_scene = (clip if have else None), shot.scene_id
                continue
            fingerprint = shot_fingerprint(project, shot)
            redo = only is not None and shot.shot_id in only
            done = progress.get(shot.shot_id)
            stale = bool(done) and done.get("fingerprint", fingerprint) != fingerprint
            if stale:
                print(f"[{shot.index}/{len(project.shots)}] {shot.shot_id}: storyboard changed, regenerating")
            if not redo and not stale and done and clip.is_file() and clip.stat().st_size > 0:
                print(f"[{shot.index}/{len(project.shots)}] {shot.shot_id}: already done")
            elif only is not None and not redo and not stale and clip.is_file():
                print(f"[{shot.index}/{len(project.shots)}] {shot.shot_id}: kept")
            else:
                pictures, chained = self.plan_shot(project, shot, prev_clip, prev_scene, work, dirs)
                prompt = build_h3_prompt(project, shot, pictures)
                check_picture_tags(prompt, len(pictures))
                names = [self.stage_picture(p.path, f"aimovie_{shot.index:02d}_{i:02d}.png")
                         for i, p in enumerate(pictures)]
                seconds = project.generation_seconds(shot)
                graph = build_h3_graph(models=self.models, prompt=prompt, width=project.width,
                                       height=project.height, seconds=seconds, seed=shot.seed,
                                       picture_names=names, ref_image_size=project.ref_image_size,
                                       filename_prefix=f"aimovie_{shot.index:02d}")
                (work / "prompts").mkdir(exist_ok=True)
                (work / "prompts" / f"{shot.index:02d}_{shot.shot_id}.txt").write_text(prompt, encoding="utf-8")
                print(f"[{shot.index}/{len(project.shots)}] {shot.shot_id}: {seconds:.2f}s, "
                      f"{len(pictures)} picture(s), chained={chained}", flush=True)
                t0 = time.monotonic()
                out = self.generate_fn(graph)
                shutil.copy2(out, clip)
                progress[shot.shot_id] = {"clip": str(clip), "chained": chained, "seed": shot.seed,
                                          "seconds": seconds, "elapsed_s": round(time.monotonic() - t0, 1),
                                          "fingerprint": fingerprint}
                progress_file.write_text(json.dumps(progress, indent=1, ensure_ascii=False), encoding="utf-8")
            clips.append(clip)
            prev_clip, prev_scene = clip, shot.scene_id
        return clips

"""Run a storyboard on Colab: ComfyUI + MiniMax H3 (Ref2VA or FL2VA), one clip per storyboard row.

All models are downloaded directly into Colab's local disk (`/content/ComfyUI/models`), matching
`MiniMax_H3_Turbo_Colab.ipynb` (no Google Drive model storage or slow Drive I/O):

  UNETLoader -> LoraLoaderModelOnly -> BasicGuider
  CLIPLoader(type="minimax") + VAELoader(video) + VAELoader(audio)
  MiniMaxH3ReferenceToVideo (Ref2VA, ref_images.ref_image_0..8)  |  MiniMaxH3ImageToVideo (FL2VA, first_frame)
  RandomNoise + KSamplerSelect + BasicScheduler -> SamplerCustomAdvanced
  VAEDecode + VAEDecodeAudio -> VHS_VideoCombine (h264 mp4 with ONE audio stream)

Generation modes per segment (`ShotSpec.gen_mode`, see script_loader.GEN_MODE_ENGINE):
  reference            Ref2VA, scene + character pictures
  reference_keyframes  Ref2VA + MiniMaxH3AddGuide pinning a first and/or last keyframe (experimental)
  first_frame          FL2VA with a first frame
  first_last           FL2VA with first_frame + last_frame
Ref2VA and FL2VA are different diffusion models; only the ones a batch needs are downloaded.

Continuity between clips:
  anchor           every clip starts from the locked scene picture.
  hybrid           inside the same scene_id the previous clip's last clean frame (before tail_guard) becomes
                   <Picture 1>; character sheets stay as <Picture 2..>. A new scene_id resets to the anchor.
  chain_last_frame always chain from the previous clip.
  Segments only chain when the next one opens on the same framing (景別) the previous one ended on;
  otherwise the next clip starts from the scene picture or its own keyframe.
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
    _resolve_path,
    build_h3_prompt,
    check_picture_tags,
    framing_en,
    h3_frames,
    shot_engine,
    shot_gen_mode,
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
                   filename_prefix: str = "ai_movie",
                   keyframes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """ComfyUI API graph for one clip (mirrors the upstream notebook's graph).

    keyframes {"first": name, "last": name}: Ref2VA pins them with MiniMaxH3AddGuide at frame 0 / the last
    frame; FL2VA feeds "last" to the node's last_frame input (the first frame is picture_names[0]).
    """
    keyframes = {k: v for k, v in (keyframes or {}).items() if v}
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
        positive = ["7", 0]
        for nid, img_id, key, idx in (("20", "30", "first", 0), ("21", "31", "last", -1)):
            if key in keyframes:
                g[img_id] = {"class_type": "LoadImage", "inputs": {"image": keyframes[key]}}
                g[nid] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
                    "positive": positive, "vae": ["4", 0], "latent": ["7", 1], "image": [img_id, 0],
                    "frame_idx": idx}}
                positive = [nid, 0]
    else:
        if len(picture_names) != 1:
            raise ValueError("FL2VA needs exactly one first-frame picture")
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": picture_names[0]}}
        g["16"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["6", 0], "upscale_method": "nearest-exact", "megapixels": 0.98, "resolution_steps": 32}}
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "first_frame": ["16", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames}}
        if "last" in keyframes:
            g["17"] = {"class_type": "LoadImage", "inputs": {"image": keyframes["last"]}}
            g["18"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": ["17", 0], "upscale_method": "nearest-exact", "megapixels": 0.98, "resolution_steps": 32}}
            g["7"]["inputs"]["last_frame"] = ["18", 0]
        positive = ["7", 0]
    g.update({
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": positive}},
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


@dataclass
class GenerationPlan:
    mode: str                         # reference | reference_keyframes | first_frame | first_last
    engine: str                       # model set: reference | first_frame
    pictures: List[PictureRef]
    keyframes: Dict[str, str]         # {"first"/"last": local path}
    chained: bool
    prompt: str


class H3ColabRunner:
    """Drives a ComfyUI server on the Colab VM with models stored directly in `/content/ComfyUI/models`."""

    def __init__(self, comfy_dir: str | Path = "/content/ComfyUI", base_url: str = "http://127.0.0.1:8188",
                 models: Optional[ModelSet] = None,
                 log_path: str | Path = "/content/comfyui_ai_movie.log") -> None:
        self.comfy_dir = Path(comfy_dir)
        self.base_url = base_url.rstrip("/")
        self.models = models
        self.model_sets: Dict[str, ModelSet] = {models.mode: models} if models else {}
        self.log_path = Path(log_path)
        self.process: Optional[subprocess.Popen] = None
        self.generate_fn: Callable[..., Path] = self._generate_with_comfy

    # ------------------------------------------------------------------ models
    def detect_models(self, mode: str) -> ModelSet:
        """Pick the model files for one engine ('reference' or 'first_frame'); call once per engine needed."""
        import torch
        assert torch.cuda.is_available(), "Colab: Runtime > Change runtime type > GPU (A100 recommended)"
        ms = choose_models(mode, torch.version.cuda or "0", torch.cuda.get_device_capability(0))
        self.model_sets[mode] = ms
        if self.models is None or self.models.mode == mode:
            self.models = ms
        return ms

    def _sets(self) -> Dict[str, ModelSet]:
        if self.models is not None and self.models.mode not in self.model_sets:
            self.model_sets[self.models.mode] = self.models
        return self.model_sets

    def ensure_models(self) -> Dict[str, str]:
        """Download the weights of every detected model set directly into `/content/ComfyUI/models`."""
        from huggingface_hub import hf_hub_download

        assert self._sets(), "call detect_models() first"
        models_root = self.comfy_dir / "models"
        found: Dict[str, str] = {}
        files = list(dict.fromkeys(f for ms in self._sets().values() for f in ms.files()))
        for repo, remote, folder in files:
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
        modes = set(self._sets()) or {"reference"}
        if "reference" in modes:
            need |= {"MiniMaxH3ReferenceToVideo"}
        if "first_frame" in modes:
            need |= {"MiniMaxH3ImageToVideo", "ImageScaleToTotalPixels"}
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
        info = getattr(self, "object_info", None)
        if info:
            missing = {n["class_type"] for n in graph.values()} - set(info)
            if missing:
                raise RuntimeError(f"this ComfyUI has no {sorted(missing)} node; update ComfyUI (git pull) to use "
                                   f"keyframe modes, or switch the segment to 參考圖 mode")
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
    def plan_generation(self, project: StoryboardProject, shot: ShotSpec, prev_clip: Optional[Path],
                        prev_shot: Optional[ShotSpec], work_dir: Path, base_dirs: List[Path]) -> "GenerationPlan":
        """Pictures, keyframes and prompt for one clip, applying the generation and continuity modes."""
        mode, engine = shot_gen_mode(project, shot), shot_engine(project, shot)
        plan = shot_picture_plan(project, shot, base_dirs)
        cont = shot.continuity_override or project.continuity_mode
        same_framing = not (prev_shot and prev_shot.end_framing and shot.start_framing) or \
            framing_en(prev_shot.end_framing) == framing_en(shot.start_framing)
        chain = prev_clip is not None and prev_clip.is_file() and prev_shot is not None and same_framing and (
            cont == "chain_last_frame" or (cont == "hybrid" and prev_shot.scene_id == shot.scene_id))

        def keyframe(raw: str, what: str) -> str:
            if not raw:
                return ""
            path = _resolve_path(raw, base_dirs)
            if not Path(path).is_file():
                raise FileNotFoundError(f"shot {shot.shot_id}: {what} {raw!r} not found in {base_dirs}")
            return path

        first_kf, last_kf = keyframe(shot.first_frame, "首幀圖"), keyframe(shot.last_frame, "尾幀圖")
        chain_png = ""
        if chain and not first_kf:
            chain_png = str(extract_clean_continuity_frame(
                prev_clip, work_dir / "chain_frames" / f"{shot.index:02d}_start.png",
                tail_guard_sec=project.tail_guard_sec))
        chained = bool(chain_png)
        keyframes: Dict[str, str] = {}
        end_picture = 0
        opening: Optional[bool] = None
        if engine == "reference":
            if chain_png:
                plan = [PictureRef(chain_png, "chain", "the last frame of the previous clip")] + \
                       [p for p in plan if p.role in ("character", "extra") and p.char_id]
            if mode == "reference_keyframes":
                keyframes = {k: v for k, v in (("first", first_kf), ("last", last_kf)) if v}
                if first_kf:
                    opening = False                     # frame 0 is pinned by the guide, not by <Picture 1>
            pictures = plan[:9]
        else:
            first = first_kf or chain_png or (plan[0].path if plan else "")
            pictures = [PictureRef(first, "keyframe" if first_kf else ("chain" if chain_png else "scene"),
                                   "the opening frame")] if first else []
            opening = True
            if mode == "first_last":
                if not last_kf:
                    raise ValueError(f"shot {shot.shot_id}: 首尾幀模式需要尾幀圖")
                keyframes = {"last": last_kf}
                end_picture = 2                          # FL2VA hands the text encoder [first, last]
        if not pictures:
            raise ValueError(f"shot {shot.shot_id}: no picture found. Upload a scene/first-frame picture or set "
                             f"reference_images; H3 Ref2VA/FL2VA both need at least one image")
        prompt = build_h3_prompt(project, shot, pictures, opening=opening, end_picture=end_picture)
        check_picture_tags(prompt, len(pictures) + (1 if end_picture else 0))
        return GenerationPlan(mode, engine, pictures, keyframes, chained, prompt)

    def plan_shot(self, project: StoryboardProject, shot: ShotSpec, prev_clip: Optional[Path],
                  prev_scene: Optional[str], work_dir: Path, base_dirs: List[Path]) -> tuple:
        """Backwards-compatible wrapper: (pictures, chained)."""
        prev = ShotSpec(shot_id="_prev", index=0, scene_id=prev_scene or "") if prev_scene is not None else None
        gp = self.plan_generation(project, shot, prev_clip, prev, work_dir, base_dirs)
        return gp.pictures, gp.chained

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
        assert self._sets(), "call detect_models() first"
        scope = [s for s in project.shots if shot_ids is None or s.shot_id in shot_ids]
        need = {shot_engine(project, s) for s in scope}
        if need - set(self._sets()):
            raise ValueError(f"these shots need the {sorted(need - set(self._sets()))} model(s) but only "
                             f"{sorted(self._sets())} were prepared; re-run the model cell")
        work = Path(work_dir)
        clips_dir = work / "raw_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        dirs = [Path(b) for b in (base_dirs or [])]
        progress_file = work / "progress.json"
        progress: Dict[str, Any] = json.loads(progress_file.read_text(encoding="utf-8")) \
            if resume and progress_file.is_file() else {}

        clips: List[Path] = []
        prev_clip: Optional[Path] = None
        prev_shot: Optional[ShotSpec] = None
        for shot in project.shots:
            clip = clips_dir / f"{shot.index:02d}_{shot.shot_id}.mp4"
            if shot_ids is not None and shot.shot_id not in shot_ids:
                have = clip.is_file() and clip.stat().st_size > 0
                prev_clip, prev_shot = (clip if have else None), shot
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
                gp = self.plan_generation(project, shot, prev_clip, prev_shot, work, dirs)
                pictures, chained = gp.pictures, gp.chained
                prompt = gp.prompt
                names = [self.stage_picture(p.path, f"aimovie_{shot.index:02d}_{i:02d}.png")
                         for i, p in enumerate(pictures)]
                kf_names = {k: self.stage_picture(v, f"aimovie_{shot.index:02d}_kf_{k}.png")
                            for k, v in gp.keyframes.items()}
                seconds = project.generation_seconds(shot)
                graph = build_h3_graph(models=self._sets()[gp.engine], prompt=prompt, width=project.width,
                                       height=project.height, seconds=seconds, seed=shot.seed,
                                       picture_names=names, ref_image_size=project.ref_image_size,
                                       filename_prefix=f"aimovie_{shot.index:02d}", keyframes=kf_names)
                (work / "prompts").mkdir(exist_ok=True)
                (work / "prompts" / f"{shot.index:02d}_{shot.shot_id}.txt").write_text(prompt, encoding="utf-8")
                print(f"[{shot.index}/{len(project.shots)}] {shot.shot_id}: {seconds:.2f}s, {gp.mode}, "
                      f"{len(pictures)} picture(s), keyframes={sorted(gp.keyframes) or '-'}, chained={chained}",
                      flush=True)
                t0 = time.monotonic()
                out = self.generate_fn(graph)
                shutil.copy2(out, clip)
                progress[shot.shot_id] = {"clip": str(clip), "chained": chained, "seed": shot.seed, "mode": gp.mode,
                                          "seconds": seconds, "elapsed_s": round(time.monotonic() - t0, 1),
                                          "fingerprint": fingerprint}
                progress_file.write_text(json.dumps(progress, indent=1, ensure_ascii=False), encoding="utf-8")
            clips.append(clip)
            prev_clip, prev_shot = clip, shot
        return clips

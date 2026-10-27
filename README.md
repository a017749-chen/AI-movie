# 🎬 AI-Movie（MiniMax H3-Turbo Colab × autoip-mv-maker 整合版）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)

本專案將 **[autoip/autoip-mv-maker](https://github.com/autoip/autoip-mv-maker)** 的分鏡腳本架構與 **`MiniMax_H3_Turbo_Colab.ipynb` (`1zdxzAm834HesHG2uu7GbM4Kv0kbKOLLu`)** 的真實 ComfyUI 節點工作流完整整合，專為 Google Colab GPU（建議 A100）部署設計，解決四大核心痛點：

1. **🛡️ 畫面連貫不跑掉（Visual Continuity & Anti-Drift）**
   - **對齊官方 ComfyUI 節點與模型庫**：使用官方 [`Comfy-Org/ComfyUI`](https://github.com/Comfy-Org/ComfyUI) 的 `MiniMaxH3ReferenceToVideo`（Ref2VA，支援 1–9 張參照圖 `ref_images.ref_image_0..8`）與 `MiniMaxH3ImageToVideo`（FL2VA 首幀模式），搭配 `Comfy-Org/MiniMax-H3` 及 `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` 權重，自動依 CUDA 版本與 GPU 架構選擇 `int8_convrot`、`fp8_scaled` 或 `bf16`。
   - **官方畫布與影格對齊**：預設採用官方 `1344×768`（16:9，長寬皆為 32 倍數且不超過 `1344×768` 上限），24 fps 影格數嚴格對齊 `17k + 5`（單鏡最短 5 秒／124 幀、最長 15 秒／362 幀）。
   - **跨鏡自動接續乾淨尾幀（`continuity_mode = "hybrid" | "chain_last_frame" | "anchor"`）**：同一場戲內，系統自動在 `tail_guard`（預設裁除最後 `0.35s` 易變形閃白尾幀）之前擷取最後一張穩定畫面作為下一鏡的 `<Picture 1>`，同時保留 `<Picture 2..N>` 角色設定圖；跨場時自動重置回乾淨場景基準圖。
   - **色彩與影格正規化**：所有分鏡統一轉為相同解析度、24 fps、`yuv420p` 與 `bt709` 色彩空間。

2. **🔊 嚴格單一聲軌（Single Audio Track Policy）**
   - 徹底解決「生成出來有兩個聲軌／雙重人聲疊加」的問題，提供三種互斥音軌模式，並在最終輸出透過 `ffprobe` 強制驗證 `len(audio_streams) == 1`：
     - **`native_h3_audio`（預設，H3 原生單一語音）**：Prompt 採用官方 `For the target video... integrated_multimodal_description: [Shot 1] ... <d>[Chinese] ...</d>` 結構，要求場上人物閉口或僅由單一旁白發聲，並在每鏡裁切時自動執行 `-16 LUFS` 響度正規化（`loudnorm=I=-16:TP=-1.5:LRA=11`），確保跨鏡音量一致。
     - **`clip_voiceover`（外部配音／TTS 替換模式）**：Prompt 自動指示 `Nobody speaks: no dialogue, no narration`，後製時以 `-map 0:v:0 -map 1:a:0` 用外部單一旁白 WAV **完全替換** H3 原生音軌，絕不混音疊加。
     - **`master_audio`（MV 或單一總音軌模式）**：所有生成片段先以 `-an` 完全移除音軌，最後只掛載唯一一條主音軌（Master Audio）。

3. **📋 丟分鏡腳本即可生成（Storyboard-Driven）**
   - 支援三種分鏡腳本格式，自動檢查每鏡秒數上限（扣除 `tail_guard` 後單鏡不超過 `14.65s`）與 `<Picture k>` 參照標籤：
     - **Markdown 分鏡表格 (`.md`)**：[`examples/storyboard_template.md`](examples/storyboard_template.md)
     - **JSON 分鏡設定檔 (`.json`)**：[`examples/storyboard_template.json`](examples/storyboard_template.json)（相容 `scenes.json` / `shotlist.json`）
     - **`autoip-mv-maker` 專案 (`project.py`)**：[`examples/demo/project.py`](examples/demo/project.py)

4. **📝 自動時間軸對齊與燒錄繁體中文字幕（Auto Subtitles）**
   - 自動將分鏡旁白／台詞依標點切分為易讀繁體中文字句（每行 ≤ 18 字），亦支援 `project.py` 的歌曲絕對時間軸歌詞。
   - 後製前自動呼叫 ComfyUI `/free` 釋放顯存（`runner.free_vram()`），並支援以 CPU 執行 `faster-whisper` 語音時間軸強制對齊（保留分鏡正確繁體字，不產生同音錯字且不與 ComfyUI 搶顯存）。
   - 一次輸出：
     - `<title>_clean.mp4`（無字幕乾淨版，單一視訊軌 + 單一音訊軌）
     - `<title>_subtitled.mp4`（燒錄半透明底框繁體中文字幕版，單一視訊軌 + 單一音訊軌）
     - `<title>.zh-TW.srt` 與 `<title>.zh-TW.ass` 外掛字幕檔
     - `<title>_review_sheet.jpg` 全片各鏡中點抽幀連貫性檢視圖表

---

## 🚀 在 Google Colab 一鍵啟動

點擊 **[Open in Colab](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)** 開啟 [`AI_Movie_Maker_Colab.ipynb`](AI_Movie_Maker_Colab.ipynb)：

1. **Step 1**：檢查 GPU 與磁碟空間（至少 60 GiB）、在 `/content/ComfyUI` 安裝 `Comfy-Org/ComfyUI` 與 `ComfyUI-VideoHelperSuite`（所有環境與模型直接建在 Colab 本機高速磁碟，不掛載、不佔用 Google Drive）。
2. **Step 2**：上傳場景基準圖（`<Picture 1>`）與角色設定圖（`<Picture 2..N>`）。
3. **Step 3**：貼上 **Markdown 分鏡表格**（或上傳 `.md` / `.json` / `project.py`），選擇 `AUDIO_MODE` 與 `CONTINUITY_MODE`，並預覽完整 H3 Prompt。
4. **Step 4**：直接在 `/content/ComfyUI/models` 下載所需 MiniMax H3 權重、啟動 ComfyUI 並批次生成分鏡（支援 `progress.json` 斷點續傳，以及設定 `ONLY_SHOT_IDS` 單獨重抽特定鏡號）。
5. **Step 5**：釋放顯存、合成單一聲軌、對齊並燒錄繁體中文字幕、通過 `ffprobe` 單一音軌驗證並直接下載成品。

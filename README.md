# 🎬 AI-Movie（MiniMax H3-Turbo Colab × autoip-mv-maker 整合版）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)

本專案將 **[autoip/autoip-mv-maker](https://github.com/autoip/autoip-mv-maker)** 的分鏡腳本架構與 **Google Colab 上的 ComfyUI + MiniMax H3-Turbo (`Ref2VA` / `FL2VA` / `T2VA`)** 深度整合，專門解決 AI 短片、教學影片與 MV 生成時最常見的四大痛點：

1. **🛡️ 畫面連貫不跑掉（Visual Continuity & Anti-Drift）**
   - **多參考圖鎖定 (`Ref2VA` + `ref_image_size="max"`)**：固定 `<Picture 1>` 為主場景與 180 度軸線基準圖、`<Picture 2..N>` 為固定角色設定圖，保留最高 2048px 短邊細節，避免臉部模糊與換臉。
   - **自動尾幀接續 (`continuity_mode = "hybrid" | "chain_last_frame" | "anchor"`)**：同一場戲內，系統自動在 `tail_guard`（預設裁除最後 `0.35s` 易變形閃白尾幀）之前擷取乾淨穩定幀作為下一鏡的 `<Picture 1>`，同時維持 `<Picture 2..N>` 角色鎖定；跨場時自動重置回乾淨基準圖，不會累積畫質劣化。
   - **全片色彩與規格統一**：所有分鏡片段統一正規化為相同解析度、固定影格率 (`fps`)、`yuv420p` 與 `bt709` 色彩空間，杜絕拼接處閃黑或色偏。

2. **🔊 嚴格單一聲軌（Single Audio Track Policy）**
   - 徹底解決「生成出來有兩個聲軌／雙重人聲疊加」的問題，提供三種互斥音軌模式，並在最終輸出透過 `ffprobe` 強制驗證 `len(audio_streams) == 1`：
     - **`native_h3_audio`（預設，H3 原生單一語音）**：Prompt 僅對指定講者加上 `<d>[Chinese] ...</d>`，禁止背景音樂與重疊人聲，合成時僅映射 `-map 0:v:0 -map 0:a:0`。
     - **`clip_voiceover`（外部配音／TTS 替換模式）**：Prompt 自動關閉人物開口說話指令，後製時以 `-map 0:v:0 -map 1:a:0` 用外部單一旁白 WAV **完全替換** H3 原生音軌，絕不混音疊加。
     - **`master_audio`（MV 或單一總音軌模式）**：所有生成片段先以 `-an` 完全移除音軌，最後只掛載唯一一條主音軌（Master Audio）。

3. **📋 丟分鏡腳本即可生成（Storyboard-Driven）**
   - 支援三種分鏡腳本格式，自動解析並編譯結構化防跑掉提示詞：
     - **Markdown 分鏡表格 (`.md`)**：例如 [`examples/storyboard_template.md`](examples/storyboard_template.md)
     - **JSON 分鏡設定檔 (`.json`)**：例如 [`examples/storyboard_template.json`](examples/storyboard_template.json)（亦相容 `scenes.json` / `shotlist.json`）
     - **`autoip-mv-maker` 專案 (`project.py`)**：例如 [`examples/demo/project.py`](examples/demo/project.py)

4. **📝 自動時間軸對齊與燒錄繁體中文字幕（Auto Subtitles）**
   - 自動將分鏡旁白／台詞依標點切分為易讀繁體中文字句（每行約 18 字以內）。
   - 支援選用 `faster-whisper` 針對最終單一音軌做 **語音時間軸強制對齊（Forced Alignment）**，同時保留分鏡腳本上的正確繁體字（不會出現語音辨識同音錯字）。
   - 一次輸出：
     - `<title>_clean.mp4`（無字幕乾淨版，單一視訊軌 + 單一音訊軌）
     - `<title>_subtitled.mp4`（燒錄精緻半透明底框繁體中文字幕版，單一視訊軌 + 單一音訊軌）
     - `<title>.zh-TW.srt` 與 `<title>.zh-TW.ass` 外掛字幕檔
     - `<title>_review_sheet.jpg` 全片鏡頭連貫性檢視圖表

---

## 🚀 在 Google Colab 一鍵啟動

點擊上方 **[Open in Colab](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)** 按鈕開啟 [`AI_Movie_Maker_Colab.ipynb`](AI_Movie_Maker_Colab.ipynb)：

1. **Step 1–2**：掛載 Google Drive（自動快取 MiniMax H3 模型至 `/content/drive/MyDrive/ComfyUI_MiniMax_H3/models`，下次開啟免重抓）並啟動背景 ComfyUI。
2. **Step 3**：上傳你的場景基準圖（`<Picture 1>`）與角色參考圖（`<Picture 2..N>`）。
3. **Step 4**：直接在表單貼上 **Markdown 分鏡表格**（或上傳 `.md` / `.json` 分鏡檔），選擇：
   - `AUDIO_MODE`：`native_h3_audio` / `clip_voiceover` / `master_audio`
   - `CONTINUITY_MODE`：`hybrid`（推薦）/ `chain_last_frame` / `anchor`
4. **Step 5**：批次生成所有分鏡（支援 `progress.json` 斷點續傳）。
5. **Step 6**：自動合成單一聲軌、對齊並燒錄繁體中文字幕、通過 `ffprobe` 單一音軌驗證並下載成品。

---

## 📁 專案目錄結構

```text
AI-movie/
├── AI_Movie_Maker_Colab.ipynb   # Google Colab 一鍵執行主筆記本
├── mvkit/
│   ├── __init__.py              # 套件入口與統一匯出
│   ├── script_loader.py         # Markdown / JSON / project.py 分鏡解析與防跑掉 Prompt 編譯器
│   ├── h3_colab_runner.py       # ComfyUI + MiniMax H3 推論器（含尾幀接續與 Google Drive 快取）
│   ├── colab_pipeline.py        # 單一聲軌合成、tail_guard 裁切、bt709 正規化與 ffprobe 驗證
│   ├── subtitles.py             # 繁體中文字幕切分、faster-whisper 時間軸對齊、SRT/ASS 輸出與燒錄
│   ├── common.py                # autoip-mv-maker 共用工具
│   ├── storyboard.py            # autoip-mv-maker 分鏡規格轉換
│   ├── make_jobs.py             # autoip-mv-maker 工作流生成
│   ├── render_mv.py             # autoip-mv-maker MV 渲染器 (-an 單一主音軌)
│   ├── epic.py                  # autoip-mv-maker 動態歌詞字幕 (Kinetic Typography)
│   ├── audio/                   # 音訊分離、節拍偵測與 faster-whisper 詞級時間戳
│   └── qc/                      # Gemini Vision 品質檢查工具
├── examples/
│   ├── storyboard_template.md   # Markdown 分鏡腳本範本（推薦直接複製修改）
│   ├── storyboard_template.json # JSON 分鏡腳本範本（含多角色與場景鎖定範例）
│   └── demo/                    # autoip-mv-maker 範例專案
├── docs/zh-TW/                  # 繁體中文工作流與實戰經驗文件
└── tests/                       # 自動化測試（分鏡解析、單一聲軌驗證、字幕生成與 Notebook 語法檢查）
```

---

## 🧪 本機測試

```bash
python -m pytest tests -q
```

# 🎬 AI-Movie（MiniMax H3-Turbo Colab × autoip-mv-maker 整合版）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)

本專案將 **[autoip/autoip-mv-maker](https://github.com/autoip/autoip-mv-maker)** 的分鏡腳本架構與 **`MiniMax_H3_Turbo_Colab.ipynb` (`1zdxzAm834HesHG2uu7GbM4Kv0kbKOLLu`)** 的真實 ComfyUI 節點工作流完整整合，專為 Google Colab GPU（建議 A100）部署設計，解決五大核心需求：

1. **🛡️ 畫面連貫不跑掉（Visual Continuity & Anti-Drift）**
   - **對齊官方 ComfyUI 節點與模型庫**：使用官方 [`Comfy-Org/ComfyUI`](https://github.com/Comfy-Org/ComfyUI) 的 `MiniMaxH3ReferenceToVideo`（Ref2VA，支援 1–9 張參照圖 `ref_images.ref_image_0..8`）與 `MiniMaxH3ImageToVideo`（FL2VA 首幀模式），搭配 `Comfy-Org/MiniMax-H3` 及 `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` 權重，自動依 CUDA 版本與 GPU 架構選擇 `int8_convrot`、`fp8_scaled` 或 `bf16`。
   - **官方畫布與影格對齊**：預設採用官方 `1344×768`（16:9，長寬皆為 32 倍數且不超過 `1344×768` 上限），24 fps 影格數嚴格對齊 `17k + 5`（單鏡最短 5 秒／124 幀、最長 15 秒／362 幀）。
   - **只送入鏡的人、寫明人數（0.9.0）**：每個分鏡判斷誰在畫面上（腳本可寫 `出場：`），只有入鏡的人送角色圖，Prompt 寫明「畫面上剛好 N 人、每人只出現一次」；特寫手部的台詞自動改成畫外音。避免人物重複出現。詳見 [`docs/zh-TW/ARCHITECTURE_REVIEW.md`](docs/zh-TW/ARCHITECTURE_REVIEW.md)。
   - **連貫模式（`continuity_mode = "anchor"（預設） | "hybrid" | "chain_last_frame"`）**：`anchor` 每段都從同一組場景＋角色圖出發；`hybrid` 同場景內接上一段 `tail_guard`（預設 `0.35s`）之前的乾淨尾幀，並在 Prompt 寫明尾幀裡的人就是角色圖裡的人。
   - **色彩與影格正規化**：所有分鏡統一轉為相同解析度、24 fps、`yuv420p` 與 `bt709` 色彩空間；每段精準裁到影格與取樣點（PCM）後才串接，聲畫不再越接越偏（舊版 2 分鐘累積 0.66 秒）。
   - **自動檢查**：生成前檢查參考圖（三視圖、合照、場景圖有人）；生成後逐分鏡比對台詞與說話時間（faster-whisper）、偵測多出的人臉，列在審核表最上面。

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

5. **🎓 教學／招生／OSCE 示範影片流程（腳本上傳 × 角色對齊 × 場景設定 × 每 2 分鐘確認）**
   - **分鏡腳本**：條列文字（不用表格），10 秒一段至少 3 個分鏡或 15 秒一段至少 6 個分鏡；每個分鏡寫景別與 4 大模塊（色調打光、畫面描述、旁白對白、音效配樂），一個段落生成一支 H3 影片、段內切鏡。附請 AI 寫腳本的提示詞範本與規格檢查。舊寫法、`.docx`、AI Studio 匯出檔也能讀。
   - **生成模式**：每段可選參考圖、參考圖＋首幀鎖定、首幀、首尾幀（FL2VA `last_frame`／`MiniMaxH3AddGuide`），只下載用得到的模型。
   - **角色對齊**：角色表固定每個人的等級（主要／次要／路人）、參考圖與三視圖、外觀、站位、聲音與別名；`（口述）`、`（對病人）` 自動判斷說話者，角色依固定順序送入 `<Picture 2..>`。
   - **場景設定**：場景表用別名對應腳本的場景文字；「特寫…」不會被當成新場景。
   - **分批審核**：每次只生成下一批（預設 ≤ 2 分鐘），產出預覽、抽幀檢視圖與審核表；`approve` 後才會生成下一批，`redo_shots` 只重拍指定鏡號並自動換 seed。生成前先檢查缺圖，工作資料夾放 Google Drive 可斷線接續。
   - **用途預設**：`osce_demo`、`teaching`、`recruitment`。
   - 詳見 [`docs/zh-TW/EDU_VIDEO_WORKFLOW.md`](docs/zh-TW/EDU_VIDEO_WORKFLOW.md)，範例在 [`examples/osce_demo/`](examples/osce_demo/)。

## 🎨 如何用「你自己生成的人物圖、場景圖」維持一致性且不破圖、不多人

1. **準備圖片（支援 `.png` / `.jpg` / `.jpeg` / `.webp`）**
   - **場景圖（`<Picture 1>`）**：必須是**完全沒有人**的乾淨空景圖（檔名建議 `scene_場景代號.png` 或 `場景_場景名稱.png`，解析度建議 `1344×768` 或短邊 ≥ 768）。若場景圖裡有人，H3 會把背景裡的人跟角色圖的人同時畫出來造成分身。
   - **人物圖（`<Picture 2..N>`）**：每位角色一張**單人、正面（或微側）、單一角度、素色乾淨背景**的半身或全身圖（檔名建議 `char_角色代號.png` 或 `人物_角色名稱.png`，短邊 ≥ 768；透明底 PNG 會自動補乾淨淺灰底避免黑邊破圖）。**切勿上傳三視圖或多人合照**，否則 H3 會把三視圖裡的每個視角畫成三個人。
   - **首尾關鍵幀（選填，大幅運鏡才需要）**：建議與影片同為 `16:9`（若上傳直幅或方形圖，系統會以 `Lanczos` 自動置中裁切，不會把人臉橫向拉扁）。
2. **在 Colab 綁定角色與場景（兩種方式擇一）**
   - **方式 A（推薦，最精準）**：在 Step 3 將 `BIBLE_SOURCE` 選 `inline_text` 或 `upload_file`，在角色表填寫每個角色的「外觀、固定站位、嗓音、參考圖檔名」，在場景表填寫「場景描述、參考圖檔名」（即使寫 `.png` 但上傳 `.jpg`/`.webp` 也會自動對齊）。
   - **方式 B（免寫表快速自動對應）**：在 Step 2 上傳 `人物_小明.png`、`場景_咖啡廳.png`，Step 3 將 `BIBLE_SOURCE` 選 `auto_from_refs`，系統會自動從檔名建立角色與場景設定。
3. **防多人與防破圖四大保護（已內建預設開啟）**
   - 每個分鏡可在腳本加 `- 出場：角色A、角色B`（空景寫 `- 出場：無`）；系統只會把該段真正入鏡的角色圖送進 H3，並在 Prompt 強制鎖定精確人數（禁止分身與背景路人）。
   - 手部／器械特寫自動轉為畫外音（Voice-over），不會硬把說話者的臉塞進特寫造成破圖。
   - `HIGH_FIDELITY_REFS = True` 預設維持最高參考圖細節（安全上限 1280px 不爆顯存），單一分鏡低於 1.8 秒會在生成前提前警告，生成後 `AUTO_QC` 自動檢查台詞、多出的人臉與閃黑／融化破圖。

---

## 🚀 在 Google Colab 一鍵啟動

點擊 **[Open in Colab](https://colab.research.google.com/github/a017749-chen/AI-movie/blob/main/AI_Movie_Maker_Colab.ipynb)** 開啟 [`AI_Movie_Maker_Colab.ipynb`](AI_Movie_Maker_Colab.ipynb)：

1. **Step 1**：安裝環境；工作資料夾（腳本、參考圖、片段、審核紀錄）預設放 Google Drive，模型與 ComfyUI 仍裝在 Colab 本機 `/content`。沒有 GPU 時進入規劃模式。
2. **Step 2**：上傳場景圖（`<Picture 1>`）與角色圖（`<Picture 2..N>`）。
3. **Step 3**：上傳分鏡腳本與角色／場景設定表，選用途預設、段落規則與草稿／定稿 → 規格檢查、對齊報告、分批計畫、Prompt 預覽（可用 CPU 執行階段）。
4. **Step 4**：只生成下一批（≤ 2 分鐘），輸出預覽影片、抽幀檢視圖與審核表，並估算剩餘 GPU 時間。
5. **Step 5**：審核 `approve` 或 `redo_shots`（只重拍指定鏡號）。
6. **Step 6**：全部確認後合成單一聲軌、繁體中文字幕全片並下載（不需要 GPU）。

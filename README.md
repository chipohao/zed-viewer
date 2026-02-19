# ZED 2i Motion Tracker (macOS)

在 macOS 上使用 ZED 2i 立體相機，不需要 ZED SDK。透過 UVC 模式直接讀取相機，搭配 OpenCV 和 MediaPipe 實現以下功能：

- **姿態追蹤** — MediaPipe Pose Landmarker（33 個關節點）
- **立體深度圖** — StereoSGBM + WLS 濾波
- **光流視覺化** — Farneback 密集光流
- **分割視圖** — 左眼 + 深度圖並排顯示
- **OSC 輸出** — 即時傳送資料至 TouchDesigner / Ableton Live
- **Syphon 影像共享** — macOS 原生影像串流
- **虛擬相機輸出** — 透過 OBS Virtual Camera 將畫面輸出為 webcam

## 架構

```
ZED 2i Camera (USB)
       │
  zed_motion_tracker.py
       │
       ├── OSC (UDP) ────→ TouchDesigner :7500  (OSC In CHOP)
       │                    Ableton Live :7500  (Max for Live udpreceive)
       │
       ├── Syphon ──────→ TouchDesigner        (Syphon Spout In TOP)
       │                   任何 Syphon client
       │
       └── Virtual Cam ─→ Max/MSP              (jit.grab)
                           OBS / Zoom / 任何支援 webcam 的軟體
```

## 快速開始

```bash
# 執行（首次會自動建立 venv 並安裝依賴）
./run.sh

# 或手動
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 zed_motion_tracker.py
```

## 命令列參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--osc-ip` | `127.0.0.1` | OSC 目標 IP |
| `--osc-port` | `7500` | OSC 目標 port |
| `--no-osc` | — | 停用 OSC 輸出 |
| `--no-syphon` | — | 停用 Syphon 輸出 |
| `--no-vcam` | — | 停用虛擬相機輸出 |
| `--camera` | `1` | 相機 index |

```bash
# 範例：送 OSC 到其他電腦，停用 Syphon
python3 zed_motion_tracker.py --osc-ip 192.168.1.100 --osc-port 8000 --no-syphon

# 純視覺模式（不送任何外部資料）
python3 zed_motion_tracker.py --no-osc --no-syphon
```

## OSC 地址表

預設目標：`127.0.0.1:7500`

**重要**：所有 OSC 資料在所有模式下都持續發送。切換顯示模式只影響畫面，不影響 OSC 輸出。

### 姿態資料

| 地址 | 值 | 說明 |
|------|---|------|
| `/zed/pose/landmarks` | 132 floats (33×4) | 所有關節 [x, y, z, vis, ...] |
| `/zed/pose/nose` | x, y, z | 鼻子座標 |
| `/zed/pose/left_wrist` | x, y, z | 左手腕 |
| `/zed/pose/right_wrist` | x, y, z | 右手腕 |
| `/zed/pose/left_ankle` | x, y, z | 左腳踝 |
| `/zed/pose/right_ankle` | x, y, z | 右腳踝 |

### 衍生控制值

| 地址 | 值 | 範圍 | 應用範例 |
|------|---|------|---------|
| `/zed/control/hand_height_l` | float | 0–1 | 濾波器 cutoff、音量 |
| `/zed/control/hand_height_r` | float | 0–1 | 濾波器 cutoff、音量 |
| `/zed/control/hand_distance` | float | 0–1 | reverb wet/dry、視覺擴散 |
| `/zed/control/body_lean` | float | -1~1 | pan、pitch shift |
| `/zed/control/motion` | float | 0–1 | 節奏密度、視覺動態強度 |

### 深度資料

| 地址 | 值 | 說明 |
|------|---|------|
| `/zed/depth/average` | float 0–1 | 全畫面平均深度 |
| `/zed/depth/grid` | 9 floats | 3×3 區域平均深度（左上→右下） |

### 光流資料

| 地址 | 值 | 說明 |
|------|---|------|
| `/zed/flow/magnitude` | float 0–1 | 整體動態量 |
| `/zed/flow/direction` | float 0–360 | 主要移動方向（角度） |

### 系統狀態

| 地址 | 值 | 說明 |
|------|---|------|
| `/zed/mode` | int 1–4 | 目前模式 |
| `/zed/fps` | float | 實際 FPS |

## 外部軟體接收設定

### TouchDesigner

1. 新增 **OSC In CHOP**
2. Port 設為 `7500`
3. 在 Active 欄位可以看到所有 `/zed/*` 地址
4. 影像：新增 **Syphon Spout In TOP** → 選擇 "ZED Tracker"

### Max/MSP

**影像接收（虛擬相機 — 推薦）**

使用 OBS Virtual Camera，Max 可透過 `[jit.grab]` 直接收影像，不需要額外安裝 Syphon for Jitter（該 package 僅有 x86_64 binary，在 Apple Silicon 原生模式的 Max 9 無法載入）。

前置準備：安裝 [OBS Studio](https://obsproject.com/)（提供虛擬相機驅動，不需要開啟 OBS）。

```
[jit.grab @vdevice "OBS Virtual Camera" @output_texture 1]
       │
[jit.gl.videoplane @context ctx]
```

> **注意**：首次使用可能需要在 Max 的 `[jit.grab]` inspector 中手動選擇 "OBS Virtual Camera" 裝置。

**影像接收（Syphon for Jitter — 備選）**

如果使用 Max 8 或 x86_64 模式的 Max 9，也可以用 Syphon：

1. 下載 [Syphon for Jitter](https://github.com/Syphon/Jitter)（或從 Max Package Manager 搜尋 "Syphon"）
2. 將 `Syphon` 資料夾放入 `~/Documents/Max 8/Packages/`（Max 8）或對應版本的 Packages 路徑
3. 重啟 Max，在 patch 中使用：

```
[jit.gl.syphonclient @servername "ZED Tracker" @context ctx]
       │
[jit.gl.videoplane @context ctx]
```

**OSC 資料接收**

```
[udpreceive 7500]
       │
[route /zed/control]
       │
[route /hand_height_l /hand_height_r /hand_distance /body_lean /motion]
   │           │            │            │          │
[float]    [float]      [float]      [float]    [float]
```

接到 `[scale 0. 1. 0 127]` 可轉 MIDI CC，或直接用 `[line~]` 平滑後送音訊參數。

### Ableton Live (Max for Live)

在 Max for Live 裝置中使用同樣的 `[udpreceive 7500]` + `[route]` 接收 OSC。也可使用 Connection Kit 等現成 M4L 裝置。

### 建議 Port 配置

| 用途 | Port |
|------|------|
| TouchDesigner 接收 | 7500（預設） |
| Ableton Live 接收 | 7500（共用）或 7501（獨立） |
| 回傳控制 | 7600 |

## 系統需求

- macOS（已在 Apple Silicon 測試）
- Python 3.9+
- ZED 2i 相機（透過 USB 連接）

## Dependencies

| Package | 用途 |
|---------|------|
| opencv-contrib-python | 影像處理、StereoSGBM、WLS 濾波（ximgproc） |
| mediapipe | 身體姿態追蹤 |
| numpy | 數值運算 |
| matplotlib | 資料視覺化 |
| sounddevice | 音訊裝置存取 |
| python-osc | OSC 傳輸（UDP） |
| syphon-python | Syphon Metal server（macOS 影像共享，選用） |
| pyvirtualcam | 虛擬 webcam 輸出（需搭配 OBS，選用） |

## 操作說明

| 按鍵 | 功能 |
|------|------|
| `1` | 姿態追蹤（預設） |
| `2` | 深度圖 |
| `3` | 光流視覺化 |
| `4` | 左眼 + 深度（並排） |
| `d` | 切換深度色彩映射 |
| `+` / `-` | 調整深度靈敏度 |
| `s` | 截圖 |
| `q` | 離開 |

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `zed_motion_tracker.py` | 主程式 — 整合所有功能 + OSC/Syphon 輸出 |
| `zed_preview.py` | 簡易預覽工具 |
| `diagnose.py` | 相機診斷 |
| `test_quick.py` | 快速測試校正與 MediaPipe |
| `SN36291917.conf` | ZED 2i 原廠校正檔（序號 SN36291917） |
| `pose_landmarker_lite.task` | MediaPipe Pose Landmarker 模型 |
| `run.sh` | 啟動腳本（自動建立 venv） |

## 注意事項

- `SN36291917.conf` 是這台 ZED 2i 的專屬校正檔。如果換了相機，需要從 ZED 官網下載對應序號的校正檔
- 需要 `opencv-contrib-python`（不是 `opencv-python`），因為 WLS 濾波器在 `ximgproc` 模組中
- 首次執行 MediaPipe 可能會較慢（需要載入模型）
- `syphon-python` 為選用依賴，未安裝時程式仍可正常運作（Syphon 功能自動停用）
- `pyvirtualcam` 需要先安裝 [OBS Studio](https://obsproject.com/) 提供虛擬相機驅動。未安裝時程式仍可正常運作（虛擬相機功能自動停用）

# ZED 2i Motion Tracker (macOS)

在 macOS 上使用 ZED 2i 立體相機，不需要 ZED SDK。透過 UVC 模式直接讀取相機，搭配 OpenCV 和 MediaPipe 實現以下功能：

- **姿態追蹤** — MediaPipe Pose Landmarker（33 個關節點）
- **立體深度圖** — StereoSGBM + WLS 濾波
- **光流視覺化** — Farneback 密集光流
- **分割視圖** — 左眼 + 深度圖並排顯示

## 快速開始

```bash
# 執行（首次會自動建立 venv 並安裝依賴）
./run.sh
```

## 手動安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 zed_motion_tracker.py
```

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
| `zed_motion_tracker.py` | 主程式 — 整合所有功能 |
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

# Metalens AEI — Critical Dimension (CD) 分析

針對金屬透鏡（Metalens）SEM 影像的 **After Etch Inspection (AEI)** 關鍵尺寸量測與分析工具。  
透過 FFT 偵測晶格週期、SAM 精確分割柱體輪廓，並輸出 L1/L2 雙層散點統計報告。

---

## 專案結構

```
ADI AEI Analysis/
├── AEI_analysis.ipynb              # 主分析 Notebook
├── bk_metalens_5mm_topview.tif     # SEM 原始影像（5 mm 頂視圖 v1）
├── bk_metalens_5mm_top_007.tif     # SEM 原始影像（5 mm 頂視圖 v2）
├── sam_vit_b_01ec64.pth            # SAM ViT-B 模型權重
├── bk_metalens_5mm_topview/        # v1 分析輸出目錄
│   ├── cd_tensor.pt                # PyTorch 張量（CD 量測結果）
│   ├── cd_heatmap.png              # CD 熱圖（Equiv / Long / Short）
│   ├── cd_overlay.png              # L1/L2 標記覆蓋圖（局部）
│   ├── cd_full_overlay.png         # L1/L2 標記覆蓋圖（全圖）
│   ├── cd_distribution.png         # L1 vs L2 分佈直方圖
│   ├── AEI_Analysis_L1.png         # L1 散點分析圖
│   ├── AEI_Analysis_L2.png         # L2 散點分析圖
│   └── AEI_Analysis_All.png        # L1+L2 合併散點分析圖
├── bk_metalens_5mm_top_007/        # v2 分析輸出目錄（同上結構）
└── reference_code/                 # 參考程式碼
    ├── ADIAEI.ipynb
    ├── Madonredo_ADIAEI_ver1.ipynb
    └── guage_point.py
```

---

## 分析流程

```
SEM 影像
    │
    ▼
1. 載入影像 & HFW 校正 (nm/px)
    │
    ▼
2. 2D FFT 偵測晶格週期 (Pitch X / Y)
    │
    ▼
3. 前景疊合最大化，偵測網格原點
    │
    ▼
4. SAM 質心採樣 → 仿射晶格最小二乘擬合（精細校正）
    │
    ▼
5. 逐格 CD 量測（SAM 分割 / Sigmoid 邊緣擬合備援）
   ├─ Equiv CD（等效直徑）
   ├─ Long-axis CD（長軸）
   └─ Short-axis CD（短軸）
    │
    ▼
6. L1 / L2 棋盤分層（(i+j) 偶數 = L1，奇數 = L2）
    │
    ▼
7. 輸出 PyTorch 張量 & 視覺化圖表
    │
    ▼
8. AEI 散點分析（Mask CD vs. Measured CD）
   ├─ L1 分析
   ├─ L2 分析
   └─ L1+L2 合併分析
```

---

## 快速開始

### 環境需求

```bash
pip install numpy opencv-python pillow scipy matplotlib scikit-learn torch torchvision
pip install segment-anything ipywidgets
```

### 下載 SAM 模型權重

```bash
curl -L -o sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 執行分析

1. 開啟 `AEI_analysis.ipynb`
2. 修改以下參數（Cell 3）：

   | 參數 | 說明 | 預設值 |
   |------|------|--------|
   | `IMAGE_PATH` | SEM 影像路徑 | `bk_metalens_5mm_topview.tif` |
   | `HFW_UM` | 水平視野寬度（µm，來自 SEM metadata） | `14.2` |
   | `CROP_HEIGHT` | 裁切高度（排除底部資訊列，px） | `3535` |
   | `PITCH_NM` | 設計週期（nm） | `360.0` |

3. 依序執行所有 Cell

---

## 主要參數說明

### SEM 校正

| 參數 | 說明 |
|------|------|
| `HFW_UM` | 水平視野寬度（µm），決定 nm/px 換算比例 |
| `NM_PER_PX` | 自動計算：`HFW_UM × 1000 / 影像寬度（px）` |

### CD 量測

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `HALF_CELL` | 單格分析半徑（px） | `PITCH / 2 - 2` |
| `MAX_CD_NM` | CD 上限（超過視為污染） | `PITCH × 0.92` |
| `BG_SIGMA_FRAC` | 背景 Gaussian sigma 比例 | `0.60` |
| `THRESH_MULT` | 二值化閾值乘數 | `0.30` |

### 狀態分類

| 狀態 | 說明 |
|------|------|
| `ok` | 正常量測 |
| `missing` | 柱體缺失 |
| `contamination` | 污染（CD 異常偏大或位置偏移） |
| `low_contrast` | 對比度不足 |
| `boundary` | 位於影像邊界 |

---

## 輸出說明

### `cd_tensor.pt`（PyTorch 張量）

```python
import torch
data = torch.load('bk_metalens_5mm_topview/cd_tensor.pt')

data['cd_long']    # shape (N_ROWS, N_COLS)，長軸 CD（nm）
data['cd_short']   # shape (N_ROWS, N_COLS)，短軸 CD（nm）
data['cd_equiv']   # shape (N_ROWS, N_COLS)，等效 CD（nm）
data['flags']      # shape (N_ROWS, N_COLS)，狀態碼（int8）
data['L1_mask']    # shape (N_ROWS, N_COLS)，L1 布林遮罩
data['L2_mask']    # shape (N_ROWS, N_COLS)，L2 布林遮罩
data['nm_per_px']  # 校正比例（nm/px）
data['pitch_x_px'] # X 方向週期（px）
data['pitch_y_px'] # Y 方向週期（px）
```

### AEI 散點分析統計量

每張散點圖包含：R²、Slope、Intercept、RMSE、MAE、Bias、3σ 範圍、超出 3σ 標記點

---

## 典型分析結果（`bk_metalens_5mm_topview.tif`）

| 項目 | 數值 |
|------|------|
| 校正比例 | 3.4668 nm/px |
| 偵測週期 | 368.9 × 366.6 nm |
| 分析網格 | 33 × 38 = 1254 格 |
| 正常柱體 | 1163 (92.7%) |
| 缺失柱體 | 83 (6.6%) |
| 污染柱體 | 8 (0.6%) |
| Equiv CD 均值 ± 標準差 | 224.7 ± 88.3 nm |
| Δ(L1–L2) 長軸 | −13.3 nm |
| Δ(L1–L2) 短軸 | −17.8 nm |

---

## 技術細節

### 週期偵測（2D FFT）
對影像施加 Hann 視窗後做 2D FFT，在功率譜中尋找一階繞射峰，再以抛物線內插取得次像素精度週期。

### 原點偵測（前景疊合最大化）
建立全域二值遮罩後，搜尋使網格中心落在柱體前景最多的 (ox, oy)，先粗搜（1 px 步長）再細搜（0.25 px 步長）。

### 晶格精細校正（SAM + 最小二乘）
使用 SAM（Segment Anything Model）對均勻採樣的網格格點做質心量測，再以仿射晶格模型（含剪切分量）做最小二乘擬合，取代初始 FFT 估計。

### CD 量測（SAM 分割）
SAM 以網格中心為點提示，輸出三個遮罩候選；選取面積合理且包含提示點的最佳遮罩，再以強度加權質心取得次像素精度，橢圓擬合輪廓得到長短軸。

### Mask CD（名義值）
預設以量測 CD 圖做三階二維多項式擬合（捕捉透鏡徑向漸變輪廓），誤差即為蝕刻/微影偏差。有設計檔時可直接替換為設計值。

---

## 依賴套件

| 套件 | 用途 |
|------|------|
| NumPy / SciPy | 數值計算、FFT、擬合 |
| OpenCV | 輪廓偵測、橢圓擬合 |
| Pillow | 影像讀取 |
| Matplotlib | 視覺化 |
| PyTorch | 張量輸出、GPU 加速（支援 MPS） |
| segment-anything | SAM 柱體分割 |
| scikit-learn | R²、MAE 統計量 |
| ipywidgets | 互動式參數調整（非 SAM 模式） |

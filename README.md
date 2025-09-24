# SAVANT  
Seismic Acceleration and Velocity Alerting Neuralnet Transformer  
同時預測 PGA（Peak Ground Acceleration）與 PGV（Peak Ground Velocity）的多任務地震預警深度學習模型


---

## Overview
SAVANT is a research-oriented multi-task deep learning model for earthquake early warning. It jointly predicts Peak Ground Acceleration (PGA) and Peak Ground Velocity (PGV) in a single forward pass. The architecture integrates:
- CNN feature extractors for local waveform patterns
- Transformer Encoder (self-attention) to model inter-station spatial dependencies
- A Mixture Density Network (MDN) head to produce probabilistic outputs (multi-modal uncertainty)
- Dual-branch input design to ingest both acceleration and velocity waveforms simultaneously

This design eliminates the need for two separate inference runs (traditional PGA-only / PGV-only models), improving both computational efficiency and prediction robustness. Tested on Taiwan 2016 earthquake events, SAVANT shows strong recall for early warning thresholds while maintaining controlled false alarms — enabling faster, more informative alerts for seismic risk mitigation.

## 中文簡介
SAVANT 是一個用於地震早期預警的多任務深度學習模型，能在單次推論中同時預測 PGA 與 PGV。透過：
- CNN 萃取地震波形局部特徵
- Transformer Encoder 建構測站間空間關聯
- Mixture Density Network（MDN）輸出具不確定性的機率型預測
- 並行輸入分支（同時餵入加速度 / 速度訊號）

相較過去需分別推論 PGA 與 PGV 的流程，SAVANT 降低時間延遲與資源消耗，並讓 PGV 預測也受益於加速度資訊，提升整體準確度與早期預警效益。

---

## Key Results / 成果摘要
| 指標 (Metric) | 任務 | 數值 (Result) | 補充說明 |
|---------------|------|---------------|----------|
| Accuracy (分類準確率) | PGA | 62.6% | 基於 CWA 強度對應分類 |
| ±1 Intensity Range | PGA | 97.9% | 預測誤差在 ±1 級內 |
| Accuracy | PGV | 56.4% | 同上標準 |
| ±1 Intensity Range | PGV | 96.3% | |
| Early Warning Recall | PGA ≥ 0.08 m/s² 門檻 | 93.9% | 減少漏報 |
| Early Warning Precision | 同上 | 78.4% | 控制誤報率 |

> 以上數據基於 2016 年台灣地震事件資料。

---

## Project Structure / 目前資料夾結構

```
SAVANT/
├── README.md
├── .gitignore
├── data/                      
├── data_preprocess/           
├── model/                     
├── model_train_predict/       
├── model_performance_analysis/ 
```

### 目錄說明
- `data/`：存放資料、模型讀取資料模組。
- `data_preprocess/`：資料前處理，例如：格式轉換、濾波、切片、強度標籤對應。
- `model/`：核心模型組件（CNN、Transformer Encoder、Mixture Density Network、multi-task heads）。
- `model_train_predict/`：模型訓練、測試模型結果。
- `model_performance_analysis/`：評估結果分析腳本（混淆矩陣、Recall/Precision 統計）。

---

## Model Architecture / 模型架構

特點：
- Dual Input Branch：並行處理 acceleration / velocity 波形
- CNN：擷取波形特徵
- Transformer Encoder：建構跨測站依存（Self-Attention）
- MDN Head：輸出多高斯混合參數（均值 / 變異 / 混合權重）
- Multi-Task Heads：共享主幹，分支輸出 PGA / PGV 機率分布或其期望值

---


## Tech Stack
| 類別 | 使用 |
|------|------|
| 語言 | Python 3.10+ |
| 深度學習 | PyTorch |
| 實驗管理 | MLflow |
| 資料處理 | NumPy / SciPy / Obspy / Pandas / HDF5 |
| 視覺化 | Matplotlib / Seaborn |

---

## Why Multi-Task Matters
| 傳統流程 | SAVANT |
|----------|--------|
| PGA 模型 + PGV 模型，各自推論 | 單模型同時輸出 |
| 雙倍計算資源 | 計算合併 |
| 資料訊息分離 | 跨任務互補（PGV 受益於加速度） |
| 較難統一不確定性 | MDN 統一機率框架 |


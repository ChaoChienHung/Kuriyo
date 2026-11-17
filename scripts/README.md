# Repository entry points

從 repo 根目錄執行。資料下載、準備、轉換與舊 log 匯入使用標準函式庫；模型訓練另需安裝元件 requirements。

| 入口 | 責任 |
|---|---|
| `download_amazon_reviews_2014.py` / `download_amazon_reviews_2023.py` | 分年份下載原始資料，保留來源 manifest |
| `prepare_amazon_reviews_2014.py` | chronological leave-one-out 與完整 catalog |
| `prepare_amazon_reviews_2014_beauty.py` | 驗證或建立 Beauty 輸入，重用完整資料 |
| `materialize_amazon_sids.py` | 將共用版本化 SID 對照表套用到三個 split |
| `run.sbatch` | prepare → sid → materialize → train；test 獨立執行 |
| `import_slurm_training_log.py` | 從舊 log 重建有來源標記的 training/event 紀錄 |
| `train_tiger_sid.py` | 舊指令相容 wrapper，轉交 `Representations.SID.scripts.generate_sids` |
| `train_tiger_sequence.py` | 舊指令相容 wrapper，轉交 TIGER smoke trainer；不是 paper baseline |

新指令優先使用元件的 `python -m ...` 入口，不複製模型實作進本目錄。相容 wrapper 仍保留，避免破壞舊使用方式。完整流程見 [TIGER baseline README](../Recall/TIGER/baseline/README.md)，當前進度見 [HANDOFF](../HANDOFF.md)。

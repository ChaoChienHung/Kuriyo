# Cross-stage core

跨 Recall、Ranking 與資料表示階段可共用的工具放在這裡。模型專有邏輯留在各模型；只有 Recall 共用的元件放 `Recall/core/`。各模型只寫自己的模型與訓練迴圈，下面這些零件由這裡提供；需要客製時在模型裡另寫，不改共用行為。

- [`experiment.py`](experiment.py)：檔案 SHA-256、Git revision、全域 seed、JSON 寫出。
- [`run_logging.py`](run_logging.py)：單一 writer 的 `RunLogger`。`train()` 追加訓練紀錄（預設 `train.jsonl`，可指定相對路徑），`event()` 寫 JSON Lines 格式的 `events.log`。每次 invocation 用獨立 session ID，追加模式保留續訓歷史；context manager 記錄開始、完成或例外，不吞掉例外，強制終止不保證有結束事件。
- [`runs.py`](runs.py)：run 目錄的版面、`settings.json`、建立／續訓／只跑一次 test 的流程、checkpoint（含 RNG 狀態）、以 validation NDCG@10 選最佳與 early stopping、共用的命令列參數與裝置檢查。

## Run 目錄

SASRec、FMRec 與 TIGER baseline 的 `runs/<RUN_NAME>/` 都是同一個版面：

```text
runs/<RUN_NAME>/
├── settings.json                  這個 run 是什麼：資料與前處理、評估集、模型、訓練、seed、程式版本
├── events.log                     生命週期事件（開始、續訓、validation、checkpoint、early stop、test、例外）
├── train/train.jsonl              每個 optimizer step 一筆（loss、learning rate、epoch、session ID）
├── checkpoints/
│   ├── latest.pt                  續訓用：模型、optimizer、抽樣與 RNG 狀態、選擇狀態
│   └── best.pt                    validation NDCG@10 最佳的權重，test 只用它
├── validation/<set>/
│   ├── epoch-000005.json          每次評估一個檔（TIGER 用 step-020000.json）
│   ├── best.json                  best.pt 對應的指標
│   └── best-groups.json           --eval-groups：best.pt 依目標熱門度分組的指標
└── test/<set>/metrics.json        只有 --eval-test-only 會寫，每個 run 一次；含選定的 validation、test 時的 Git revision 與 Slurm job ID
    test/<set>/groups.json         --eval-groups，只在該 run 已跑過 test 時才寫
```

`<set>` 是評估集名稱，例如 `amazon2014-beauty`（2014 Beauty 的留一法切分）；之後加入其他評估集（例如冷啟動協定）時各自一個資料夾，結果不混在一起。

`settings.json` 分成 `run`（名稱、家族、變體、實作、Git revision、建立時間、Slurm job、裝置、seed）、`data`（catalog、各 split 的路徑與 SHA-256、前處理規則與筆數、歷史上限、樣本上限、訓練樣本如何構成；TIGER 另有 SID）、`model`、`training`、`evaluation`。續訓與 test 只比對影響結果的欄位（各模型 `train.py` 的 `RESUME_KEYS`、`TEST_KEYS`），不符就拒絕並說明是哪一項。

`--eval-groups`（`run.sbatch` 的 `STAGE=groups`）不訓練，只把 `checkpoints/best.pt` 再評估一次，輸出依目標在 train 當目標的次數（0、1–4、5 以上）分組的指標（`Recall/core/popularity.py`），並記下整體數字與原紀錄的最大差異 `max_abs_difference`。validation 一定做；test 只在該 run 已經跑過那一次 test 時才做，所以不會產生新的 test 結果。2026-11-19 之後的 validation 與 test 指標本身也帶 `groups`。

> ⚠️ 2026-11-19 以前的 run 是平的（`config.json`、`latest.pt`、`validation-epoch-5.json`……）。對這種 run 執行 `--resume` 或 `--eval-test-only` 時，`core/runs.py` 先用 `config.json` 核對設定，相符才把檔案搬進上面的版面、寫出 `settings.json`（保留 `config.json`），並在 `events.log` 記一筆 `legacy_layout_migrated`。不要在舊 run 仍有 job 寫入時執行。

呼叫端不可讓多個 job 同時寫同一 run。測試在 `tests/test_run_logging.py` 與 `tests/test_runs.py`。舊 log 匯入在 `scripts/import_slurm_training_log.py`，使用不同 provenance，不能把匯入時間當作原執行時間。

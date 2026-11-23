# Cross-stage core

跨 Recall、Ranking 與資料表示階段可共用的工具放在這裡。模型專有邏輯留在各模型；只有 Recall 共用的元件放 `Recall/core/`。各模型只寫自己的模型與訓練迴圈，下面這些零件由這裡提供；需要客製時在模型裡另寫，不改共用行為。

- [`experiment.py`](experiment.py)：檔案 SHA-256、Git revision、全域 seed、JSON 寫出。
- [`run_logging.py`](run_logging.py)：單一 writer 的 `RunLogger`。`train()` 追加訓練紀錄（預設 `train.jsonl`，可指定相對路徑），`event()` 寫 JSON Lines 格式的 `events.log`。每次 invocation 用獨立 session ID，追加模式保留續訓歷史；context manager 記錄開始、完成或例外，不吞掉例外，強制終止不保證有結束事件。
- [`stats.py`](stats.py)、[`compare.py`](compare.py)：README「實驗設計與統計檢定」的檢定。`python -m core.compare --setting <名稱> <run 目錄…> --setting …` 讀各 run 的逐使用者排名，輸出每個設定的平均 ± 標準差與 t 95% CI，以及對第一個設定的使用者配對檢定（各 seed 平均後的 paired t 與 10,000 次 paired bootstrap）、seed 層級 Welch t、Holm 校正與 README 的判定；`--split test`、`--metric recall@10`、`--json` 可選。只用標準函式庫與 NumPy。
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
│   ├── best-ranks.json            best.pt 下每位使用者的目標排名（配對檢定用）
│   └── best-groups.json           --eval-groups：best.pt 依目標熱門度分組的指標
└── test/<set>/metrics.json        只有 --eval-test-only 會寫，每個 run 一次；含選定的 validation、test 時的 Git revision 與 Slurm job ID
    test/<set>/ranks.json          那一次 test 的逐使用者排名
    test/<set>/groups.json         --eval-groups，只在該 run 已跑過 test 時才寫
```

`<set>` 是評估集名稱，例如 `amazon2014-beauty`（2014 Beauty 的留一法切分）；之後加入其他評估集（例如冷啟動協定）時各自一個資料夾，結果不混在一起。

`settings.json` 分成 `run`（名稱、家族、變體、實作、Git revision、建立時間、Slurm job、裝置、seed）、`data`（catalog、各 split 的路徑與 SHA-256、前處理規則與筆數、歷史上限、樣本上限、訓練樣本如何構成；TIGER 另有 SID）、`model`、`training`、`evaluation`。續訓與 test 只比對影響結果的欄位（各模型 `train.py` 的 `RESUME_KEYS`、`TEST_KEYS`），不符就拒絕並說明是哪一項。

`--eval-groups`（`run.sbatch` 的 `STAGE=groups`）不訓練，只把 `checkpoints/best.pt` 再評估一次，輸出依目標在 train 當目標的次數（0、1–4、5 以上）分組的指標（`Recall/core/popularity.py`），並記下整體數字與原紀錄的最大差異 `max_abs_difference`。validation 一定做；test 只在該 run 已經跑過那一次 test 時才做，所以不會產生新的 test 結果。2026-11-19 之後的 validation 與 test 指標本身也帶 `groups`。 TIGER 另有 `--eval-beams`（`STAGE=beams`），以不同 beam 寬度只重新評估 validation（見 TIGER baseline README）。

逐使用者排名檔（`best-ranks.json`、`ranks.json`）記錄 `users` 與同序的 `ranks`：目標在前 `max_rank`（10）名中的名次，不在其中記 0；任何 K ≤ 10 的 Recall@K、NDCG@K 都可由它逐使用者重算，規則與整體指標相同（同一商品取第一次出現；TIGER 的無效 beam 不算命中）。評估器回傳的 `per_user` 由 `runs.py` 取出另存，不進指標 JSON 與 `events.log`。訓練中每次 best 更新就重寫 `best-ranks.json`。之前的 run 沒有排名檔：`--eval-groups` 會補上缺的分組檔與排名檔，已有的不覆寫，全都有時拒絕。一個檔案約數百 KB（Beauty 22,363 位使用者），在 Git 白名單內。

> ⚠️ 2026-11-19 以前的 run 是平的（`config.json`、`latest.pt`、`validation-epoch-5.json`……）。對這種 run 執行 `--resume` 或 `--eval-test-only` 時，`core/runs.py` 先用 `config.json` 核對設定，相符才把檔案搬進上面的版面、寫出 `settings.json`（保留 `config.json`），並在 `events.log` 記一筆 `legacy_layout_migrated`。不要在舊 run 仍有 job 寫入時執行。

呼叫端不可讓多個 job 同時寫同一 run。測試在 `tests/test_run_logging.py` 與 `tests/test_runs.py`。舊 log 匯入在 `scripts/import_slurm_training_log.py`，使用不同 provenance，不能把匯入時間當作原執行時間。

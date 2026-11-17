# Cross-stage core

跨 Recall、Ranking 與資料表示階段可共用的工具放在這裡。模型專有邏輯留在各模型；只有 Recall 共用的元件放 `Recall/core/`。

目前 `run_logging.py` 提供單一 writer 的 `RunLogger`。輸出目錄由呼叫端建立；`train()` 寫 `train.jsonl`，`event()` 寫 JSON Lines 格式的 `events.log`。每次 invocation 用獨立 session ID，追加模式保留續訓歷史。context manager 記錄區段開始、正常完成或可捕捉例外，不會吞掉例外；強制終止不保證有結束事件。

呼叫端不可讓多個 job 同時寫同一 run。測試在 `tests/test_run_logging.py`，目前使用者是 TIGER paper trainer 與 SID generator。舊 log 匯入在 `scripts/import_slurm_training_log.py`，使用不同 provenance，不能把匯入時間當作原執行時間。

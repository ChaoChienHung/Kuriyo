# REVIEW

已完成、等人審查的工作。未勾選代表還沒審查，不代表實作沒完成。審查通過就移除該項；審查發現新工作時，另開一項到 [TODO.md](TODO.md)。agent 目前停在哪裡見 [HANDOFF.md](HANDOFF.md)。

## Awaiting review

- [ ] 每個變體一份 Experiment-log，INSIGHT／TODO 改成模板格式
  - **Delivered:** 新增 `Recall/{TIGER,SASRec,FMRec}/baseline/Experiment-log.md` 與 `Representations/SID/Experiment-log.md`；run 的逐 step 數字從 INSIGHT 搬進這些日誌，INSIGHT 只留結論並連結。七份 INSIGHT 加上建立／更新日期與日誌連結，根 INSIGHT 分 `### General`／`### Data`。TODO 改成 Agent-ready／Owner-led／Ideas；新增本檔。AGENTS.md、README 與 docs 跟著更新。
  - **Evidence:** 完整 unittest discovery 通過；docs lint 通過；搬移前後逐項比對過數字。
  - **Review focus:** 2026-11-19 以前的條目是事後補記，Expectation 依當時的動機補寫（各檔開頭已註明）；TIGER 的無編號假設改為 H6；已完成的 TODO 項目被移除（紀錄在 git 歷史）。
- [ ] `run.sbatch` 以 `MODEL` 訓練 SASRec 與 FMRec（`2c946fe`）
  - **Delivered:** `MODEL=tiger|sasrec|fmrec`；SASRec／FMRec 只接受 `STAGE=prepare|train|test|all`，有 `latest.pt` 就續訓，有目錄沒 checkpoint 就拒絕。
  - **Evidence:** `tests/test_run_sbatch.py` 新增 6 項 orchestration 測試並通過；SoC job 1165520（FMRec）已正常訓練。
  - **Review focus:** 同時提交多個 job 時要用 `STAGE=train`，`all`／`prepare` 會同時寫同一份切分，目前只有文件警告，沒有鎖。
- [ ] TIGER 的 T5X 式 Adafactor 選項（`83f321a`）
  - **Delivered:** `--adafactor-scale-parameter`／`ADAFACTOR_SCALE_PARAMETER=1`；預設行為不變，續訓拒絕改變此設定。
  - **Evidence:** TIGER baseline 測試通過；SoC job 1165519 前 1,500 steps 的 train loss 比 baseline 低約 1.0（validation 未出）。
  - **Review focus:** 舊 run 的 `config.json` 沒有此欄位時視為 False 是否合理。
- [ ] SASRec baseline（`17727c0`）
  - **Delivered:** `Recall/SASRec/baseline/`，原論文架構、預設 full-softmax CE，讀 Kuriyo ASIN 切分；本機完整 run validation Recall@10 0.0938。
  - **Evidence:** 變體測試通過；本機 run 紀錄見其 Experiment-log。
  - **Review focus:** 預設 CE 而非原論文的 BCE；本機 run 的 revision 早於程式 commit。
- [ ] FMRec baseline（`1ab74dc`）
  - **Delivered:** `Recall/FMRec/baseline/`，依作者程式 `1c39b17` 重新實作；與作者程式的差異列在其 README。
  - **Evidence:** 變體測試通過；SoC 完整訓練進行中。
  - **Review focus:** attention 改遮 padding key（作者程式遮 query 列）、s 預設 1.0（論文另提 0.4）。
- [ ] SoC 操作手冊（Kuriyo `c7d9de2`、Ludwigia `707a2c7`）
  - **Delivered:** Kuriyo `skills/nus-soc-slurm/SKILL.md` 改寫；Ludwigia 新增通用版 skill 與 Toolkit 項目。
  - **Evidence:** 內容來自本週的 SoC job 與 sacct／sinfo 輸出；兩邊都已推上 main。
  - **Review focus:** Ludwigia 是公開網站，確認沒有帳號、路徑或其他私人資訊；標了日期的節點觀察要不要保留。

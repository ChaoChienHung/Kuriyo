# REVIEW

已完成、等人審查的工作。未勾選代表還沒審查，不代表實作沒完成。審查通過就移除該項；審查發現新工作時，另開一項到 [TODO.md](TODO.md)。agent 目前停在哪裡見 [HANDOFF.md](HANDOFF.md)。

## Awaiting review

- [ ] 多資料集、seed 與單一因素旗標；統計規則
  - **Delivered:** `run.sbatch` 的 `CATEGORY`、`SEED`、`TRAIN_ARGS`（皆驗證字元，train 與 test 共用）；`scripts/ensure_amazon_reviews_2014.py` 處理任一或全部三個 2014 分類，舊的 Beauty 腳本改為 wrapper；SID 生成改傳 `--items` 指定分類的 catalog。README 新增「實驗設計與統計檢定」（seed 數、命名、平均 ± 標準差與 95% CI、使用者配對檢定加 seed 層級 Welch、Holm 校正、只在 validation 選擇），AGENTS.md 規定比較前必讀。TODO 新增逐使用者排名輸出、Beauty 多 seed、SASRec BCE、Sports／Toys 四項。
  - **Evidence:** 新增 2 項 orchestration 測試（分類、seed、`TRAIN_ARGS` 傳到 train 與 test；TIGER 新分類用自己的 SID 與 run 名稱），並把無效 `CATEGORY`／`SEED`／`TRAIN_ARGS` 加入拒絕測試；完整 unittest discovery 67 項通過。Beauty 預設呼叫除多傳 `--category Beauty --seed 42`（與預設值相同）外不變。
  - **Review focus:** 預設 run 名稱在 Beauty 以外帶分類（`sasrec-toys_and_games-v1`、`tiger-paper-sports_and_outdoors-v1`）；`TRAIN_ARGS` 只允許英數、`_.=-` 與空白；統計規則的門檻（3／5 個 seed、兩種檢定都要 p < 0.05）是提議，請確認。

- [ ] 共用 run 目錄與訓練零件；Experiment-log 移到家族層
  - **Delivered:** 新增 `core/runs.py`（run 版面、`settings.json`、建立／續訓／只跑一次 test、checkpoint 含 RNG、`Selection` 選最佳與 early stop、共用 CLI 與裝置檢查）與 `Recall/core/evaluation.py`；`AsinSplit`、`load_catalog`、`read_examples` 移入 `Recall/core/data.py`，`experiment.py` 移到 `core/`。三個 baseline 改用它們。run 改成 `settings.json`、`train/`、`checkpoints/`、`validation/<set>/`、`test/<set>/`；舊的平版面在續訓或 test 時核對後搬移。Experiment-log 移到 `Recall/<family>/Experiment-log.md`，條目加 Variant 與資料／前處理。INSIGHT 補上 BCE 與 CE 的機制（SASRec H1）與冷啟動限制（`Recall/INSIGHT.md` H2、TIGER H7）。
  - **Evidence:** 重構前後以固定 seed 跑 SASRec、FMRec（真實資料子集）與 TIGER（合成資料），各含一次續訓與 test，loss、validation、test 指標逐位相同；舊版面 run 的搬移、拒絕與續訓在本機驗證；完整 unittest discovery 65 項通過（新增 `tests/test_runs.py` 3 項）；docs lint 通過。
  - **Review focus:** 舊 run 搬移後保留 `config.json`，`settings.json` 中未被核對的欄位取自搬移時的指令；`validation/` 檔名改為補零的 `epoch-000005.json`／`step-020000.json`；TIGER test 的 `selected_validation` 改存完整指標。smoke trainer 仍用舊版面。
- [ ] 每個變體一份 Experiment-log，INSIGHT／TODO 改成模板格式
  - **Delivered:** 新增 `Recall/{TIGER,SASRec,FMRec}/baseline/Experiment-log.md` 與 `Representations/SID/Experiment-log.md`；run 的逐 step 數字從 INSIGHT 搬進這些日誌，INSIGHT 只留結論並連結。七份 INSIGHT 加上建立／更新日期與日誌連結，根 INSIGHT 分 `### General`／`### Data`。TODO 改成 Agent-ready／Owner-led／Ideas；新增本檔。AGENTS.md、README 與 docs 跟著更新。
  - **Evidence:** 完整 unittest discovery 通過；docs lint 通過；搬移前後逐項比對過數字。
  - **Review focus:** 2026-11-19 以前的條目是事後補記，Expectation 依當時的動機補寫（各檔開頭已註明）；TIGER 的無編號假設改為 H6；已完成的 TODO 項目被移除（紀錄在 git 歷史）。
- [ ] `run.sbatch` 以 `MODEL` 訓練 SASRec 與 FMRec（`2c946fe`）
  - **Delivered:** `MODEL=tiger|sasrec|fmrec`；SASRec／FMRec 只接受 `STAGE=prepare|train|test|all`，有 `latest.pt` 就續訓，有目錄沒 checkpoint 就拒絕。
  - **Evidence:** `tests/test_run_sbatch.py` 新增 6 項 orchestration 測試並通過；SoC job 1165520（FMRec）已正常訓練。
  - **Review focus:** 同時提交多個 job 時要用 `STAGE=train`，`all`／`prepare` 會同時寫同一份切分，目前只有文件警告，沒有鎖。
- [ ] TIGER 的 T5X 式 Adafactor 選項（`83f321a`）
  - **Delivered:** `--adafactor-scale-parameter`／`ADAFACTOR_SCALE_PARAMETER`；續訓拒絕改變此設定。job 1165519 證實有效後，預設改為 T5X 式（1），`--no-adafactor-scale-parameter`／`0` 還原舊行為。
  - **Evidence:** TIGER baseline 測試通過；SoC job 1165519 前 1,500 steps 的 train loss 比 baseline 低約 1.0（validation 未出）。
  - **Review focus:** 預設改成 1 後，續訓舊 run（如 `tiger-paper-v1`）必須明確設 0；舊 `config.json` 沒有此欄位時視為 False。
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

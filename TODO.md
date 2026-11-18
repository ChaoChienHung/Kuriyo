# TODO

更新：2026-11-17。背景與操作交接見 [HANDOFF.md](HANDOFF.md)，研究證據與假設見 [INSIGHT.md](INSIGHT.md) 與各層的 INSIGHT。勾選表示已完成該項工作，不代表論文重現成功。

## P0 — 確認完整 baseline 與 SID 問題

- [x] SoC job 1153307 對完整 Beauty 生成 SID，完成 200,000-step TIGER 訓練並記錄 10 次 validation。
- [x] 同步 Slurm log，確認最佳已記錄 validation 在 step 200,000；尚無最終 test 結果。
- [x] 舊 log 匯入獨立 train.jsonl/events.log，保留來源 SHA、行號、缺失資訊與採樣頻率。
- [ ] 取得／核對該 run 的 config、SID manifest、best-metrics、實際 checkpoint 存在性與 job 最終狀態；不要提交大型產物。
- [x] 檢查 RQ-VAE 首 epoch 巨大 loss 與低碼本使用率：PyTorch Adagrad accumulator 預設 0 造成第一步發散；穩定 optimizer 下第一層仍塌縮。診斷工具 `Representations/SID/scripts/diagnose_rqvae.py`，結果見 `Representations/SID/INSIGHT.md`。
- [x] `generate_sids --recipe`：Adagrad accumulator、未使用碼重設、embedding 標準化；`run.sbatch` 的 `SID_RECIPE` 與配方不符時拒絕重用；補元件與 orchestration 測試。
- [ ] 在 SoC 以 `SID_RECIPE=stable` 產生新 SID 版本，檢查 manifest 的 `codebook_usage_ratio` 與碰撞；可選再跑 `paper-tf-adagrad` 對照。
- [ ] 以新 SID、同一 T5 設定、新 RUN_NAME 在 SoC 訓練，比較 20k／60k step validation 與現有 run（驗證 `Recall/TIGER/INSIGHT.md` H3）。
- [ ] 對 SoC 上現有 SID 統計碰撞商品比例與最大 collision suffix（本機診斷已含此統計，可直接套用 `embeddings.npz`）。
- [ ] 核對 paper/T5X 與 HF 設定：實際參數量 5,587,968、metadata 文句、user hash、prefix expansion、optimizer、解碼與評估。
- [x] 熱門商品基線：validation Recall@10 0.01704、test 0.01203（見 `Recall/INSIGHT.md`）。
- [ ] 設計單一因素對照實驗（SID 方法、模型容量），使用新 run 名稱保留現有結果。
- [ ] 模型選定後執行一次 held-out test，再與論文 test 指標比較；不用 test 調參。

## P1 — 可重現訓練與維護

- [x] SID 與完整 catalog 獨立於推薦模型；SID/splits 按版本隔離。
- [x] 共用 Recall Transformer、padding、metrics、實驗工具與模組測試。
- [x] TIGER 專有 tests 分離；驗證 causal mask、padding invariance、訓練／續訓／test。
- [x] 分階段 run.sbatch、SoC skill、Torch cu126 明確安裝指令。
- [x] 獨立 run logger 接入 TIGER paper trainer 與 SID；測試追加、session 與錯誤傳遞。
- [x] 舊資料目錄 ignore、清除 tracked .DS_Store；Slurm log 曾提交，已於 `2ad21c7` 移除，保留在 git 歷史。
- [x] README/AGENTS 記錄家族與變體原則，建立 HANDOFF。
- [ ] SID 增加 checkpoint/resume（目前無中途恢復）。
- [ ] 完整環境 lock／環境快照與 GPU preflight；既有 Torch pin 不是完整 lockfile。
- [ ] 增加小型 run summary 匯出機制，方便分享 config／metrics／manifest，而不提交 runs/checkpoints。
- [x] TIGER 家族改為每個變體自成完整空間，現有實作全部放入 `baseline/`，記錄於 README/AGENTS。
- [ ] 真正新增變體時實作具名設定或複製 `baseline/` 為 `Recall/TIGER/<variant-name>/`，補該變體的測試。

## P1 — FMRec 同切分比較

- [x] `Recall/FMRec/baseline/`：依作者程式 `1c39b17` 實作，讀 Kuriyo ASIN 切分（20 件歷史上限），共用指標 full ranking；測試與 CPU smoke 通過。
- [ ] 在 SoC GPU 以作者預設跑完整訓練（新 RUN_NAME），記錄 validation，與 TIGER 及熱門基線的 validation 比較。
- [ ] 視結果做單一因素對照：`--mode-scale 0.4`、`--max-history 50`（需另建 50 件歷史的切分）、`--attention causal`。
- [ ] 若要多次跑 FMRec，再為 `scripts/run.sbatch` 加 FMRec 的 STAGE；目前只有手動指令。

## P1 — SASRec 資料／評估檢查

- [x] `Recall/SASRec/baseline/`：原論文架構、預設 full-softmax CE，讀 Kuriyo ASIN 切分（20 件歷史上限），共用指標 full ranking；測試通過。
- [x] 完整訓練並記錄 validation：本機 `sasrec-beauty-v1` best Recall@10 0.0938（約 TIGER 的 3 倍），共用資料／評估正常，差距在 TIGER 流程。
- [ ] 找出 TIGER 流程的問題（訓練樣本、T5 設定與 token 排列、解碼）；見 `Recall/TIGER/INSIGHT.md` Problems。
- [ ] 視結果做單一因素對照：`--loss bce`（原論文 loss）、hidden 50／1 head。

## P2 — 後續模型與資料

- [ ] 定義跨 Recall 模型的候選輸出、載入與評估契約。
- [ ] 實作 sentence-t5_kmeans、sentence-t5_lsh；目前只有預留方法名稱。
- [ ] 明定固定 catalog 與 cold-start 協定，量測同日時間 ties 的敏感度。
- [ ] 若啟用 2023 All_Beauty，另外定義切分與評估，不與 2014 混用。
- [ ] Ranking 候選集、負例與基線；有實際模型後定義跨階段 pipelines。

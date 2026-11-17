# TODO

更新：2026-11-17。背景與操作交接見 [HANDOFF.md](HANDOFF.md)，研究證據與假設見 [INSIGHT.md](INSIGHT.md)。勾選表示已完成該項工作，不代表論文重現成功。

## P0 — 確認完整 baseline 與 SID 問題

- [x] SoC job 1153307 對完整 Beauty 生成 SID，完成 200,000-step TIGER 訓練並記錄 10 次 validation。
- [x] 同步 Slurm log，確認最佳已記錄 validation 在 step 200,000；尚無最終 test 結果。
- [x] 舊 log 匯入獨立 train.jsonl/events.log，保留來源 SHA、行號、缺失資訊與採樣頻率。
- [ ] 取得／核對該 run 的 config、SID manifest、best-metrics、實際 checkpoint 存在性與 job 最終狀態；不要提交大型產物。
- [ ] 檢查 RQ-VAE 首 epoch 約 8.744e9 loss，以及每層僅使用 6/26/25 個碼的原因；增加分項 loss、embedding 尺度與梯度診斷。
- [ ] 統計碼本佔用分布、碰撞商品比例與最大 collision suffix；228 個碰撞群組不等於碰撞率。
- [ ] 核對 paper/T5X 與 HF 設定：實際參數量 5,587,968、metadata 文句、user hash、prefix expansion、optimizer、解碼與評估。
- [ ] 固定 split/SID 建立熱門商品 validation baseline；設計單一因素對照實驗，使用新 run 名稱保留現有結果。
- [ ] 模型選定後執行一次 held-out test，再與論文 test 指標比較；不用 test 調參。

## P1 — 可重現訓練與維護

- [x] SID 與完整 catalog 獨立於推薦模型；SID/splits 按版本隔離。
- [x] 共用 Recall Transformer、padding、metrics、實驗工具與模組測試。
- [x] TIGER 專有 tests 分離；驗證 causal mask、padding invariance、訓練／續訓／test。
- [x] 分階段 run.sbatch、SoC skill、Torch cu126 明確安裝指令。
- [x] 獨立 run logger 接入 TIGER paper trainer 與 SID；測試追加、session 與錯誤傳遞。
- [x] 舊資料目錄 ignore、Slurm log 可提交，清除 tracked .DS_Store。
- [x] README/AGENTS 記錄家族與變體原則，建立 HANDOFF。
- [ ] SID 增加 checkpoint/resume（目前無中途恢復）。
- [ ] 完整環境 lock／環境快照與 GPU preflight；既有 Torch pin 不是完整 lockfile。
- [ ] 增加小型 run summary 匯出機制，方便分享 config／metrics／manifest，而不提交 runs/checkpoints。
- [x] TIGER 家族改為 `baseline/` 子目錄結構，記錄於 README/AGENTS。
- [ ] 真正新增變體時實作具名設定或 `Recall/TIGER/<variant-name>/`，共用訓練與評估，補變體專有測試。

## P2 — 後續模型與資料

- [ ] 定義跨 Recall 模型的候選輸出、載入與評估契約。
- [ ] 實作 sentence-t5_kmeans、sentence-t5_lsh；目前只有預留方法名稱。
- [ ] 明定固定 catalog 與 cold-start 協定，量測同日時間 ties 的敏感度。
- [ ] 若啟用 2023 All_Beauty，另外定義切分與評估，不與 2014 混用。
- [ ] Ranking 候選集、負例與基線；有實際模型後定義跨階段 pipelines。

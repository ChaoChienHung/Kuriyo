# TODO

更新：2026-11-19。未完成的工作放這裡；已完成、等人審查的工作放 [REVIEW.md](REVIEW.md)；agent 目前停在哪裡見 [HANDOFF.md](HANDOFF.md)；研究證據與假設見 [INSIGHT.md](INSIGHT.md) 與各層的 INSIGHT。

## Task rules

- 每個 checkbox 一件工作，寫清楚結果與驗收方式。
- 只有影響實作時才寫範圍與相依。
- 需要 SoC 的工作由 agent 準備指令、使用者提交；agent 不自行提交、取消或修改 job。
- 長期構想留在 Ideas to discuss，接受、否決或完成後再移走。

## Agent-ready

依優先順序。

- [ ] 找出 T5X 式 Adafactor TIGER 的最佳 step 與 20k 後下降的原因（H8）
  - **Context:** `Recall/TIGER/INSIGHT.md` H8：job 1165519 的 validation 在 20k 最好（0.0800），到 100k 降到 0.0546，無效 beam 增加。
  - **Context（更新）：**train loss 在 validation 下降時仍在降，支持過擬合；job 1165519 已由使用者結束，`best.pt` 在 20k。
  - **Scope:** 新 run `RUN_NAME=tiger-t5x-adafactor-eval2k-v1 TRAIN_ARGS="--eval-every 2000 --steps 40000"`（同 seed、同 SID）找最佳 step；第一次提交 job 1167529 因 Slurm controller 連線失敗沒有開始訓練，需重送。之後再決定對抗過擬合的單一因素（dropout、提早停止）。
  - **Acceptance:** 2k 間隔的 validation 曲線記在 TIGER Experiment-log，最佳 step 寫進 H8，並寫下一個實驗。
- [ ] 逐使用者評估輸出與比較工具
  - **Context:** README「實驗設計與統計檢定」的主要檢定是使用者層級的配對檢定；目前 validation／test 只存整體平均，無法配對。
  - **Scope:** `Recall/core/evaluation.py` 另存每位使用者的目標排名（`validation/<set>/` 與 `test/<set>/` 旁的小檔，不改既有指標 JSON）；`core/` 或 `scripts/` 加比較指令：讀多個 run（多 seed），輸出平均 ± 標準差、95% CI、paired bootstrap／paired t、Welch t 與 Holm 校正後的 p 值。TIGER 的 beam 結果同樣輸出排名或 miss。
  - **Acceptance:** 合成資料測試驗證 bootstrap 與 t 檢定的數值；既有整體指標逐位不變（共用程式的回歸規則）；有 checkpoint 的 run 可重新評估補出排名檔。
- [ ] Beauty 多 seed：SASRec 兩個設定、FMRec 各補 seed 43–46
  - **Context:** README 規定定案結果要 5 個 seed；既有 `sasrec-beauty-v1`（CE 設定）、`fmrec-beauty-v1` 即 seed 42。
  - **Scope:** agent 準備指令，使用者提交；SASRec CE 設定 `SEED=43 RUN_NAME=sasrec-ce-beauty-s43 TRAIN_ARGS="--loss ce --hidden 64 --heads 2"`，FMRec `SEED=43 RUN_NAME=fmrec-beauty-s43`；每個 job 一張 GPU，先查配額。
  - **Acceptance:** 五個 seed 的 validation 平均 ± 標準差記在各家族 Experiment-log 與 `Recall/INSIGHT.md` 對照表。
- [ ] Sports_and_Outdoors、Toys_and_Games 資料與三個模型
  - **Context:** TIGER 論文在三個 2014 分類上報告；只看 Beauty 的結論可能不穩。`run.sbatch` 已支援 `CATEGORY`。
  - **Scope:** `Sports_and_Outdoors` 已準備（README）；`Toys_and_Games` 仍需 `CATEGORY=Toys_and_Games STAGE=prepare`；之後 SASRec、FMRec、TIGER（各自的 SID 版本）用預設 run 名稱訓練，多 seed 依 README。
  - **Acceptance:** 兩個分類的切分筆數、商品數與論文資料表對照；三個模型的 validation 分資料集記在各 Experiment-log 與 `Recall/INSIGHT.md`。
- [ ] 對既有 run 跑分組評估並記錄（冷啟動與長尾）（優先：TIGER 的 test 比 validation 低 31%，item-ID 模型只低 17–22%）
  - **Context:** `Recall/INSIGHT.md` H2、`Recall/TIGER/INSIGHT.md` H7：SASRec／FMRec 的 ID embedding 隨機初始化，對沒學過的商品無能為力；validation 目標有 20.8% 在 train 當目標少於 5 次、0.23% 完全沒出現。整體指標由熱門商品主導。
  - **Scope:** 程式已完成（`Recall/core/popularity.py`、`--eval-groups`／`STAGE=groups`）；第一批 9 個 job（1167705–13）用 pull 前的程式提交，全部在 stage 檢查就結束，需重送。使用者在 SoC 對 `sasrec-beauty-v1`、`sasrec-paper-beauty-*`、`sasrec-paperce-beauty-*`、`fmrec-beauty-v1`、`tiger-t5x-adafactor-v1` 執行 `STAGE=groups` 並推上紀錄。
  - **Acceptance:** 各 run 的 `best-groups.json`（有 test 的另有 `test/<set>/groups.json`）`max_abs_difference` 為 0 或接近 0；分組 Recall@10／NDCG@10 並列在 `Recall/INSIGHT.md`，更新 H2 與 TIGER H7。
- [ ] 以各論文自己的協定重現（確認沒有低估模型）
  - **Context:** 共用切分與全 catalog 評估適合比較模型，但無法和論文數字直接對照。原始 SASRec 論文用 50 件歷史、每位使用者 1 個正例 + 100 個未互動過的隨機商品排名；TIGER test 比論文低 14%。
  - **Scope:** 在 `--eval-groups` 的重新評估框架加「抽樣負例」協定（100 個、排除使用者看過的商品、固定 seed），各模型都能跑；之後視結果建立 50 件歷史的切分（另一個資料版本，結果分開報告）並用論文設定重訓。
  - **Acceptance:** 各模型在論文協定下的數字與論文並列；差距在合理範圍即確認實作正確，否則記下差異來源。
- [ ] 若 TIGER 跑滿後仍明顯落後 item-ID 模型，找剩下的差距
  - **Context:** 20k 時與 SASRec CE／FMRec 仍差約 0.02；但已高於 SASRec 論文設定（0.060）。
  - **Scope:** 容量（H4）、訓練樣本、token 排列、解碼；每項新 RUN_NAME。
  - **Acceptance:** 找到能縮小差距的單一因素，或逐項排除並記錄在 Experiment-log。
- [ ] 核對論文 T5X 與 HF T5 的設定
  - **Context:** `Recall/TIGER/INSIGHT.md` H4；HF T5 4.8–5.6M 參數，論文約 1,300 萬。
  - **Scope:** 容量（embedding 共享、詞表）、商品文句、user hash、prefix expansion、解碼與評估。
  - **Acceptance:** TIGER INSIGHT 有逐項對照，事實與猜測分開。
- [ ] SoC 上舊 SID 的碰撞商品比例與最大 collision suffix
  - **Context:** 碰撞群組數（228／279）不能換算成碰撞商品比例。
  - **Scope:** 對 SoC 的 `embeddings.npz`／`semantic_ids.jsonl` 套用 `diagnose_rqvae` 的統計；使用者在 SoC 執行指令。
  - **Acceptance:** 兩個 SID 版本的碰撞商品數與最大群記在 `Representations/SID/Experiment-log.md`。
- [ ] 核對 job 1153307 的產物
  - **Context:** 只有 log，沒有 config、SID manifest、best-metrics、checkpoint 存在性與 sacct 最終狀態。
  - **Scope:** 使用者在 SoC 執行檢查指令並回報；不提交大型產物。
  - **Acceptance:** HANDOFF 與 TIGER baseline Experiment-log 註明哪些產物存在、job 最終狀態。
- [ ] SID 分項 loss 記錄
  - **Context:** `Representations/SID/INSIGHT.md` Problems to solve 1。
  - **Scope:** `generate_sids` 的 `train.jsonl` 加入重建、碼本、commitment 分項 loss 與 latent 範數；補測試。
  - **Acceptance:** 小資料 run 的 `train.jsonl` 每 epoch 有分項欄位，測試通過。
- [ ] SID checkpoint／resume
  - **Context:** 20,000 epochs 中斷就要重跑。
  - **Scope:** `Representations/SID/`、`scripts/run.sbatch` 的 sid 階段與 orchestration 測試。
  - **Acceptance:** 中斷後以同一 SID_VERSION 續訓，結果與不中斷一致（小資料測試）。
- [ ] 單一因素對照（各自新 RUN_NAME，等主要問題釐清後）
  - **Context:** `Recall/FMRec/INSIGHT.md` H1、`Recall/SASRec/INSIGHT.md` H1。
  - **Scope:** FMRec `--mode-scale 0.4`、`--attention causal`、`--sample-steps 1` 對 30；SASRec hidden 50／1 head（BCE 另列上方）。以 `TRAIN_ARGS` 傳旗標；要下結論時依 README 至少 3 個 seed。
  - **Acceptance:** 每項在對應的 Experiment-log 有一條完整紀錄，並更新假設狀態。
- [ ] 完整環境 lock 與 GPU preflight
  - **Context:** 現有 Torch pin 不是完整 lockfile；壞節點要等 Python 啟動後才發現。
  - **Scope:** 環境快照、`run.sbatch` 在安裝前檢查 `nvidia-smi`。
  - **Acceptance:** 壞節點上的 job 在 pip 之前就以明確訊息結束；環境可由 lock 重建。

## Owner-led

- [ ] 決定何時定案並跑 held-out test
  - **Decision needed:** 哪個 TIGER run 算定案；之後才對 test 跑一次，不用 test 調參。SASRec（`sasrec-beauty-v1`，CE 設定）與 FMRec（`fmrec-beauty-v1`）的 seed 42 已於 2026-11-19 跑過 test；其他 seed 與設定的 test 依 README 的統計規則。
  - **Acceptance:** 每個定案 run 只有一次 test，記在 Experiment-log，並與論文 test 數字並列。
- [ ] 確定推薦任務與成功指標
  - **Decision needed:** 下一項商品召回之外，是否也做評分預測或排序；目標使用情境與主要指標。
  - **Acceptance:** 根 INSIGHT 的 Discussion 寫下決定，Problems to solve 1 關閉。
- [ ] 跨 Recall 模型的候選輸出契約
  - **Decision needed:** 候選集的格式、數量與載入方式，供 Ranking 與 pipelines 共用。
  - **Acceptance:** `Recall/README.md` 記錄格式；至少一個 Recall 模型能輸出它。
- [ ] 評估協定的延伸
  - **Decision needed:** 是否加入固定 catalog 之外的 cold-start 協定、同日時間戳的敏感度分析、全域時間切分（根 INSIGHT H2）。
  - **Acceptance:** 採用的協定寫進 README 與根 INSIGHT，結果與現有切分分開報告。
- [ ] Ranking 的候選集、負例與標籤
  - **Decision needed:** 是否以 Recall 輸出作為候選集、負例怎麼抽、標籤怎麼定。
  - **Acceptance:** `Ranking/INSIGHT.md` 記錄決定，可以開始建立 Ranking 基線。

## Ideas to discuss

- 啟用 2023 `All_Beauty`：需要另訂切分與評估，不與 2014 混用。
- 實作 `sentence-t5_kmeans`、`sentence-t5_lsh` 比較 SID 方法（`Recall/TIGER/INSIGHT.md` H6）；等 TIGER 的訓練正常後才有意義。
- FMRec 以 50 件歷史對照論文設定（需另建 `--max-history 50` 的切分），優先度低。
- 真正需要不同實作時，新增 TIGER 變體（例如 `constrained-decoding/`），附自己的 Experiment-log 與測試。

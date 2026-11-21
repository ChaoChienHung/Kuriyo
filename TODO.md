# TODO

更新：2026-11-19。未完成的工作放這裡；已完成、等人審查的工作放 [REVIEW.md](REVIEW.md)；agent 目前停在哪裡見 [HANDOFF.md](HANDOFF.md)；研究證據與假設見 [INSIGHT.md](INSIGHT.md) 與各層的 INSIGHT。

## Task rules

- 每個 checkbox 一件工作，寫清楚結果與驗收方式。
- 只有影響實作時才寫範圍與相依。
- 需要 SoC 的工作由 agent 準備指令、使用者提交；agent 不自行提交、取消或修改 job。
- 長期構想留在 Ideas to discuss，接受、否決或完成後再移走。

## Agent-ready

依優先順序。

- [ ] Sports_and_Outdoors、Toys_and_Games 資料與三個模型
  - **Context:** TIGER 論文在三個 2014 分類上報告；只看 Beauty 的結論可能不穩。`run.sbatch` 已支援 `CATEGORY`。
  - **Scope:** `Sports_and_Outdoors` 與 `Toys_and_Games` 都已準備（README）；之後 SASRec、FMRec、TIGER（各自的 SID 版本）用預設 run 名稱訓練，多 seed 依 README。
  - **Acceptance:** 兩個分類的切分筆數、商品數與論文資料表對照；三個模型的 validation 分資料集記在各 Experiment-log 與 `Recall/INSIGHT.md`。
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

- [ ] TIGER 下一步：要不要繼續追與 item-ID 模型的差距
  - **Context:** TIGER validation Recall@10 停在約 0.080（SASRec／FMRec 約 0.10）。正則化只移動最佳點（H8，dropout 0.2）；參數比論文少不是原因（H4：transformer 本體與論文維度相同，13M 推算是大詞表 embedding）。TIGER test 0.0556，比論文低 14%。
  - **Decision needed:** (a) 試比論文更大的模型（例如 `--d-model 256`，偏離論文的設定比較，需新旗標）；(b) 補 TIGER 的 3 個 seed，把 0.080 定案後轉往 Sports／Toys；(c) 先做論文協定（抽樣 100 負例）確認 TIGER 與論文的差距。

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

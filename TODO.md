# TODO

更新：2026-11-19。未完成的工作放這裡；已完成、等人審查的工作放 [REVIEW.md](REVIEW.md)；agent 目前停在哪裡見 [HANDOFF.md](HANDOFF.md)；研究證據與假設見 [INSIGHT.md](INSIGHT.md) 與各層的 INSIGHT。

## Task rules

- 每個 checkbox 一件工作，寫清楚結果與驗收方式。
- 只有影響實作時才寫範圍與相依。
- 需要 SoC 的工作由 agent 準備指令、使用者提交；agent 不自行提交、取消或修改 job。
- 長期構想留在 Ideas to discuss，接受、否決或完成後再移走。

## Agent-ready

依優先順序。

- [ ] TIGER 3 個 seed（Beauty），然後 Sports_and_Outdoors 三個模型
  - **Context:** 使用者 2026-11-21 選定方向 (b)：先把 TIGER 約 0.080 的 Beauty 結果補到 3 個 seed，再看 TIGER 在 Sports 是否同樣落後 item-ID 模型。Expectation 都已事先寫在各 Experiment-log（2026-11-21，待執行）。
  - **Scope:** 使用者提交（可同時）：
    - TIGER seeds：`for s in 43 44; do SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-eval2k-s$s SEED=$s TRAIN_ARGS="--eval-every 2000 --steps 40000" STAGE=train sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch; done`
    - TIGER Sports：`CATEGORY=Sports_and_Outdoors SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-sports_and_outdoors-v1 TRAIN_ARGS="--eval-every 2000 --steps 60000" STAGE=all sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch`
    - SASRec Sports：`MODEL=sasrec CATEGORY=Sports_and_Outdoors STAGE=train sbatch --export=ALL --job-name=kuriyo-sasrec scripts/run.sbatch`，以及加 `RUN_NAME=sasrec-paperce-sports_and_outdoors-v1 TRAIN_ARGS="--loss ce"` 的同一指令
    - FMRec Sports：`MODEL=fmrec CATEGORY=Sports_and_Outdoors STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
  - **Acceptance:** 各條目填入 Results／Conclusion；TIGER 3 seed 以 `core.compare` 對 SASRec CE 與 FMRec（seed 42–44）；Sports 的跨家族表加進 `Recall/INSIGHT.md`。Toys_and_Games 之後另開。
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

- TIGER 的逐層溫度校準（Sequential Temperature Scaling，使用者 2026-11-23 提出，等 beam 寬度與描述 SID 兩個實驗之後）：beam 分數是 4 層條件機率的累積乘積，各層過度自信會在連乘後放大、讓 beam 錯剪目標前綴。由第 1 層往後逐層以一維網格搜尋溫度 T_j，最小化「前 j 層累積機率」對「目標前 j 碼是否仍在」的 ECE。注意：單層溫度保序，但不同層用不同 T_j 會改變完整 SID 的排序，所以在 TIGER 上它同時是重新排序，Recall 可能變好或變壞；校準集要從 validation 切出一半（另一半報告），不碰 test。beam 寬度實驗會先告訴我們搜尋誤差有多大。
- 啟用 2023 `All_Beauty`：需要另訂切分與評估，不與 2014 混用。
- 實作 `sentence-t5_kmeans`、`sentence-t5_lsh` 比較 SID 方法（`Recall/TIGER/INSIGHT.md` H6）；等 TIGER 的訓練正常後才有意義。
- FMRec 以 50 件歷史對照論文設定（需另建 `--max-history 50` 的切分），優先度低。
- 真正需要不同實作時，新增 TIGER 變體（例如 `constrained-decoding/`），附自己的 Experiment-log 與測試。

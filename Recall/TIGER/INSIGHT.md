# TIGER — Research Notes

TIGER 家族的研究筆記，適用於家族下所有變體。SID 本身的發現在 [Representations/SID/INSIGHT.md](../../Representations/SID/INSIGHT.md)；跨家族比較在 [Recall/INSIGHT.md](../INSIGHT.md)。

格式：`Insights`（從資料或實驗得到的觀察）、`Facts`（已確認的資料、實作與 run 結果）、`Limitations`（當前缺點）、`Hypothesis`（待驗證主張）、`Discussion`、`Problems to solve`；事實與假設分開。假設編號只在本檔內有效，跨檔引用時寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H3）。

*建立日期：2026-11-18（由根 INSIGHT 分出）。*

## Insights

- 完整 run 的十次 validation 中，最佳已記錄 NDCG@10 在 step 200,000（0.016787758）；不是嚴格單調上升，也尚無 test 結果。SID 三層使用碼數為 6/26/25，值得調查，但其與推薦效果的因果關係尚未驗證。
- SID 品質是目前最主要的嫌疑：舊 SID 訓練在第一個 epoch 就發散並塌縮，幾乎不帶商品語意（診斷見 [SID INSIGHT](../../Representations/SID/INSIGHT.md)）。「修好 SID 就能拉高 Recall」仍是假設（H3）。

## Facts

- 2014 `Beauty` 的 22,363 位使用者與 12,101 個商品，與 [TIGER 論文](https://arxiv.org/abs/2305.05065) 的 Beauty 資料表相符。
- **歷史長度上限 20 件**依 [TIGER 論文](https://arxiv.org/abs/2305.05065) Appendix C：論文寫訓練時使用者歷史最多 20 件；評估時的上限與訓練樣本如何由序列切出（只取最後一筆或展開所有前綴）論文都沒寫。Kuriyo 評估同樣截到 20 件，並把 train 段落的每個前綴展開成樣本，這兩點是 Kuriyo 的選擇。論文表 6 的 Beauty 平均序列長度 8.87、中位數 6，受上限影響的使用者應不多。附錄內容由網頁摘要取得，正式引用前需對照原文。
- `Representations/SID/` 已有商品內容向量、三層 RQ-VAE、碰撞處理及 SID 生成器；`Recall/TIGER/` 有 SID 序列 Transformer 與 validation 評估流程。試跑指令與模型限制見 [SID README](../../Representations/SID/README.md) 與 [TIGER baseline README](../../Recall/TIGER/baseline/README.md)。
- [TIGER 論文](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)報告的 Beauty 數值為 Recall@5 0.0454、NDCG@5 0.0321、Recall@10 0.0648、NDCG@10 0.0384。論文使用 Sentence-T5 向量、三層 RQ-VAE（每層 256 碼、20,000 epochs、Adagrad 0.4）與含 2,000 個 hashed user token 的 T5X 序列模型（200,000 steps）。這些數字是比較目標，尚非 Kuriyo 的結果。
- [`scripts/run.sbatch`](../../scripts/run.sbatch) 可分階段準備 Beauty 資料、訓練 SID、轉換樣本及訓練 PyTorch/Hugging Face T5 baseline；[`scripts/prepare_amazon_reviews_2014_beauty.py`](../../scripts/prepare_amazon_reviews_2014_beauty.py) 可驗證並重用已下載資料。SoC job 1153307 已執行至 200,000 steps 並印出 Saved；完整 log `slurm-kuriyo-tiger-1153307.out` 曾同步進 repo，已於 `2ad21c7` 移除，可用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回。
- 完整 run `tiger-paper-20261117-005232`（程式 `5f99955`）對 12,101 商品產生 SID，20,000 epochs 結束時使用碼數 6/26/25，碰撞群組 228。第一 epoch loss=8743993902.953560，最後 epoch loss=0.178358；尚無分項 loss 可定位原因。228 群組不能直接換算碰撞商品比例。
- step 200,000 的 validation（22,363 examples）：Recall@5=0.020569691、Recall@10=0.034208290、NDCG@5=0.012409267、NDCG@10=0.016787758；依已記錄 NDCG@10 為最佳，沒有最終 test。資料源為 job 1153307 log（見上一條取回方式），不是本機 smoke 指標。
- **新 SID 的 TIGER run `tiger-paper-v1`（訓練中）**：step 100,000 的 validation（22,363 examples）Recall@5=0.015964、Recall@10=0.028485、NDCG@5=0.010054、NDCG@10=0.014137，invalid_beam_fraction 0；同 step 的 train loss 1.793（舊 run 1.642）。舊 SID run 在 step 100,000 為 Recall@10 0.0310（job 1153307 log）。資料源是使用者貼出的 SoC log 片段，尚未取得完整 log 與 job ID。新 SID 第一層碼分布較平均，每個 token 的不確定性較高，所以兩個 run 的 loss 不可直接比較。
- 本地序列 smoke test 只取 128 筆 train、3 筆 validation、訓練 1 epoch；Recall@5/10 與 NDCG@5/10 均為 0。此數字僅驗證執行與指標流程，不能代表推薦品質；產物在 `Recall/TIGER/baseline/runs/smoke-sequence-from-final-sid/`。

## Limitations

- 現有完整 run `tiger-paper-20261117-005232` 使用舊 SID（三層使用率約 2.34%/10.16%/9.77%）；`stable` 配方的新 SID `tiger-paper-v1` 已產生（見 [SID INSIGHT](../../Representations/SID/INSIGHT.md)），但尚未 materialize 或訓練 TIGER，所以目前所有 TIGER 指標都來自舊 SID。
- 原有小型序列模型、訓練預算及限制商品目錄的解碼設定尚未對齊 TIGER 論文；序列 smoke test 只涵蓋極少量樣本，因此不能把目前的指標當作論文重現結果。
- HF T5 已在完整資料上跑到 200,000 steps；log 實測參數量為 5,587,968，取代早期約 470 萬的估值。與論文 T5X 的容量、hash、訓練細節及商品文句仍需逐項核對；未取得最終 test，不可宣稱數值復現。

## Hypothesis

- 可在固定切分與相同模型預算下，比較 RQ-VAE SID、隨機 ID 與其他內容編碼方式對召回品質的影響；尚未進行此比較。
- **H3 — SID 品質是與論文差距的主因（目前證據不支持）：**新 SID 在 100k step 的 Recall@10 0.0285，與舊 SID 同 step 的 0.0310 相當（標準誤約 0.0011，舊 run 自身在 60k–100k 間就波動 0.0287–0.0310）；同切分的 SASRec 達 0.0938（[Recall INSIGHT](../INSIGHT.md)）。差距較可能在 TIGER 的序列模型、訓練樣本或解碼。原假設：現行 SID 由發散後的 RQ-VAE 產生（PyTorch Adagrad accumulator 預設 0；論文框架的 Adagrad 通常預設 0.1），第一層只用 6 碼。用診斷中 E 或 F 設定（[SID INSIGHT](../../Representations/SID/INSIGHT.md) 的診斷表）重訓 SID、其他條件不變，預期 TIGER validation Recall@10 會明顯高於 0.0342。驗證方式：新 SID_VERSION、同一 T5 設定，先比較相同 step（如 20k／60k）的 validation，再決定是否跑滿 200k。
- **H4 — 模型容量不足是次要因素：**HF T5 實測 5,587,968 參數，論文約 1,300 萬。差異可能來自 T5X 不共享輸入輸出 embedding 或詞表大小，尚未核實；在 H3 驗證前不先調整。

## Discussion

- [TIGER 論文](https://arxiv.org/abs/2305.05065) 使用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games 資料，依使用者時間序列預測下一個商品；保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。目前下載的 2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。
- 早期 smoke trainer 使用 catalog-constrained beam search；完整 HF T5 run 使用 unconstrained beam search，十次 validation 的 invalid_beam_fraction 都為 0。兩種入口不可混為同一設定；後續調整仍先看 validation，模型定案後才看 test。

## Problems to solve

1. `tiger-paper-v1` 跑完後補上完整 validation 曲線；目前 100k step 已顯示新 SID 沒有改善（見 H3）。
2. 找出 TIGER 與同切分 SASRec（Recall@10 0.0938）差距的原因：逐項檢查訓練樣本、T5 設定與 token 排列、beam search 解碼；可先用隨機 SID 或小型 overfit 測試確認模型能學到 train 目標。
3. 保留已完成的完整 run 作為對照，做單因素比較（熱門基線見 [Recall INSIGHT](../INSIGHT.md)）；模型定案後再對 test 做一次最終評估。
4. 對齊論文的模型容量、訓練預算與解碼協定，確認指標的可比性。

# SID — Research Notes

與模型無關的商品 SID 產生（`Representations/SID/`）的研究筆記。事實與假設分開，每個結果都註明資料版本、設定、指標，以及來自哪一條 Experiment-log 或哪個 run。程式運作與陷阱見 [docs 的 sid-rqvae-training](../../docs/subsystems/sid-rqvae-training.md)；SID 對 TIGER 指標的影響記在 [TIGER INSIGHT](../../Recall/TIGER/INSIGHT.md)。

*建立：2026-11-18（由根 INSIGHT 分出）；最後更新：2026-11-19。Experiment log：[Experiment-log.md](Experiment-log.md)。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- PyTorch Adagrad 的 accumulator 預設 0，照抄論文 lr 0.4 會讓第一步發散、三層塌縮；即使 optimizer 穩定，Sentence-T5 向量的大共同分量仍讓 RQ-VAE 只學到平均 embedding。embedding 標準化加上重設未使用碼才同時解決兩者。證據：[Experiment-log](Experiment-log.md) 2026-11-18「RQ-VAE 塌縮診斷」。
- `stable` 配方的總 loss 隨訓練上升，原因是 latent 尺度擴大使量化誤差變大，重建仍持續下降；判斷 SID 好壞要看重建、碼使用數與碰撞，不能只看總 loss。證據：同檔 2026-11-18「`stable` 配方的 SoC 完整 SID」。
- 健康的 SID 不足以拉近 TIGER 與論文的差距（[TIGER INSIGHT](../../Recall/TIGER/INSIGHT.md) H3 不支持）；在 TIGER 的訓練修好之前，比較 SID 方法的意義有限。

## Facts

已確認的資料性質、實作細節與 run 結果。

- [`Representations/SID/scripts/generate_sids.py`](scripts/generate_sids.py) 以 Sentence-T5 與 RQ-VAE 產生 SID，輸出到 `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/<version>/`；`--recipe paper-pytorch|paper-tf-adagrad|stable` 選訓練配方。[`scripts/materialize_amazon_sids.py`](../../scripts/materialize_amazon_sids.py) 把完整商品 SID 對照表套用到三個 split。`sentence-t5_kmeans` 與 `sentence-t5_lsh` 只有預留目錄，沒有生成器。
- 舊 SID（job 1153307，論文設定的 PyTorch 實作）：三層使用 6/26/25 碼、碰撞群組 228，第一 epoch loss 約 8.7e9。新 SID `tiger-paper-v1`（job 1162430，`stable`，20,000 epochs）：256/256/256、碰撞群組 279。兩者的群組數不能直接比。證據：[Experiment-log](Experiment-log.md) 2026-11-17、2026-11-18。
- 本機 60 epochs（seed 42）下，三個配方的結果：`paper-pytorch` 1/1/1；`paper-tf-adagrad` 256/256/256 但重建只到「輸出平均 embedding」的水準；`stable` 256/256/256 且重建明顯更低。證據：同檔 2026-11-18。

## Limitations

會限制結論範圍的已知弱點。

- 商品目錄涵蓋三個 split；若要評估未見商品的歸納能力，需另訂只用 train 商品訓練 SID 的協定。
- 診斷只有單一 seed、60／300 epochs，遠少於論文的 20,000。
- `generate_sids` 的 `train.jsonl` 只記總 loss，分項 loss 要用 `diagnose_rqvae` 在本機重跑；SID 訓練沒有中途 checkpoint。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Representations/SID/INSIGHT.md` H[n]。

- 尚無。

## Discussion

仍在權衡的設計選擇。

- 尚無。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

1. 在 `train.jsonl` 記錄重建、碼本、commitment 分項 loss 與 latent 範數，讓 SoC run 不必在本機重現就能判讀。
2. SID 訓練沒有 checkpoint／resume，中斷就要重跑 20,000 epochs。

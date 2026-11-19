# TIGER — Research Notes

TIGER 家族的研究筆記，適用於家族下所有變體。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。SID 本身的發現在 [SID INSIGHT](../../Representations/SID/INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

*建立：2026-11-18（由根 INSIGHT 分出）；最後更新：2026-11-19。Experiment log：[baseline](baseline/Experiment-log.md)。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- **差距主要來自 Adafactor 步長，不是 SID：**全碼本使用的新 SID 在 200k 只比舊 SID 高 0.0026 Recall@10（0.0368 對 0.0342）；同一 SID 只把 Adafactor 改成 T5X 式縮放，20k 就從 0.0132 變成 0.0800，接近同切分 SASRec 與 FMRec（約 0.10）。照抄論文超參數時，要確認它在原框架裡的語意（相對或絕對步長）。證據：[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-18、2026-11-19。
- **舊設定到 200k 仍在上升：**兩個 `scale_parameter=False` 的完整 run 都沒有收斂，與步長過大造成的訓練噪音相符。比較變體時要比相同 step。

## Facts

已確認的資料性質、實作細節與 run 結果。

- 2014 `Beauty` 的 22,363 位使用者與 12,101 個商品，與 [TIGER 論文](https://arxiv.org/abs/2305.05065) 的 Beauty 資料表相符。
- **歷史長度上限 20 件**依 [TIGER 論文](https://arxiv.org/abs/2305.05065) Appendix C：論文寫訓練時使用者歷史最多 20 件；評估時的上限與訓練樣本如何由序列切出（只取最後一筆或展開所有前綴）論文都沒寫。Kuriyo 評估同樣截到 20 件，並把 train 段落的每個前綴展開成樣本，這兩點是 Kuriyo 的選擇。論文表 6 的 Beauty 平均序列長度 8.87、中位數 6，受上限影響的使用者應不多。附錄內容由網頁摘要取得，正式引用前需對照原文。
- [TIGER 論文](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)報告的 Beauty 數值為 Recall@5 0.0454、NDCG@5 0.0321、Recall@10 0.0648、NDCG@10 0.0384。論文使用 Sentence-T5 向量、三層 RQ-VAE（每層 256 碼、20,000 epochs、Adagrad 0.4）與含 2,000 個 hashed user token 的 T5X 序列模型（200,000 steps）。這些數字是比較目標，尚非 Kuriyo 的結果。
- 程式：`Representations/SID/` 產生 SID；[`scripts/run.sbatch`](../../scripts/run.sbatch) 分階段準備資料、訓練 SID、轉換樣本及訓練 HF T5 baseline（操作見 [baseline README](baseline/README.md)）。
- **完整 run 的 validation（2014 Beauty，22,363 examples，beam 10，seed 42）：**

  | run | SID | Recall@10（200k，另註者除外） | NDCG@10 | 來源 |
  | --- | --- | --- | --- | --- |
  | `tiger-paper-20261117-005232` | 舊（6/26/25） | 0.0342 | 0.0168 | job 1153307；[log](baseline/Experiment-log.md) 2026-11-17 |
  | `tiger-paper-v1` | `stable`（256/256/256） | 0.0368 | 0.0184 | job 1163052；[log](baseline/Experiment-log.md) 2026-11-18 |
  | `tiger-t5x-adafactor-v1` | `stable`，T5X 式 Adafactor | 0.0800（**20k**，訓練中） | 0.0434 | job 1165519；[log](baseline/Experiment-log.md) 2026-11-19 |

  兩者 invalid_beam_fraction 都是 0，都沒有跑 test。逐 step 曲線只記在 Experiment-log。
- **Adafactor 設定與論文框架不同**：`Recall/TIGER/baseline/train.py` 以 Hugging Face `Adafactor(lr=0.01, scale_parameter=False, relative_step=False)` 訓練。HF 在 `scale_parameter=False` 時每步更新量約為 lr 的絕對值（更新先以 RMS 正規化、clip 到 1）。T5X `t5x/adafactor.py` 的 `multiply_by_parameter_scale` 預設為 True，把 lr 乘上參數 RMS（下限 epsilon2=1e-3）；docstring 寫明 False 時 lr 是絕對步長。HF T5 初始化下，encoder `q.weight` 的 RMS 約 0.011、`k`／`v`／FFN `wi` 約 0.088、`o` 約 0.051、embedding 約 1.0。因此 Kuriyo 前 10k steps 每步約改動 `q` 權重自身大小的 90%，200k 時（lr 0.0022）仍約 20%；T5X 對同一權重的步長約 1e-4。`--adafactor-scale-parameter` 可改成 T5X 行為。
- T5 實測參數量：舊 SID 5,587,968、新 SID 4,815,488（詞表較小）。

## Limitations

會限制結論範圍的已知弱點。

- 每個設定只有一個 seed、只有 validation；沒有 test，不能和論文 test 直接比。
- HF T5 與論文 T5X 的容量、hash、訓練細節及商品文句尚未逐項核對，不可宣稱數值復現。
- 早期 smoke trainer 用 catalog-constrained beam search，完整 HF T5 run 用 unconstrained beam search；兩種入口不是同一設定。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/TIGER/INSIGHT.md` H[n]。

- **H3 — SID 品質是與論文差距的主因（不支持）：**原預期：用不塌縮的 SID 重訓，其他不變，validation Recall@10 會明顯高於 0.0342。結果 0.0368，只高約 2 個標準誤（約 0.0013），差距主因不在 SID。證據：[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-18。
- **H4 — 模型容量不足是次要因素：**HF T5 實測 4.8–5.6M 參數，論文約 1,300 萬。差異可能來自 T5X 不共享輸入輸出 embedding 或詞表大小，尚未核實；H5 修正後與 item-ID 模型仍差約 0.02，是下一個候選，等跑滿 200k 再決定。
- **H5 — Adafactor 步長是 TIGER 差距的主因（強烈支持）：**Kuriyo 以 `scale_parameter=False` 把 lr 0.01 當成絕對步長，小初始化的權重（如 `q`）每步被大幅改寫，訓練噪音大、收斂慢。兩個完整 run 到 200k 仍在上升，train loss 約 1.69／token（約 8.5 nats／商品），SASRec 約 7.0 nats 且 30 epochs 內收斂，與此相符。驗證：`ADAFACTOR_SCALE_PARAMETER=1`、SID `tiger-paper-v1`、新 RUN_NAME，比較 20k／40k validation 與 `tiger-paper-v1`（Recall@10 0.0132／0.0174）。結果：20k Recall@10 0.0800 對 0.0132（約 6 倍），已超過舊設定跑滿 200k 的 0.0368；[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-19（job 1165519）。40k 之後的曲線待補。
- **H6 — SID 方法影響召回品質：**在固定切分與相同模型預算下，RQ-VAE SID、隨機 ID 與其他內容編碼（`sentence-t5_kmeans`、`sentence-t5_lsh`）的召回品質不同。尚未比較，需先讓 TIGER 本身的訓練正常（H5）。

## Discussion

仍在權衡的設計選擇。

- **資料版本：**[TIGER 論文](https://arxiv.org/abs/2305.05065) 用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games，保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。目前只做 Beauty。
- **解碼：**完整 run 的 unconstrained beam search 沒有產生無效 SID（invalid_beam_fraction 0），暫不需要 constrained decoding；若之後出現無效 beam，再考慮 `constrained-decoding/` 變體。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

1. 讓 job 1165519 跑滿 200k，記錄最佳 step 與 validation；確認是否追上 SASRec／FMRec（約 0.10）。
2. 若仍落後，候選是容量（H4）、訓練樣本、token 排列與解碼。程式碼檢查已確認 token offset、user token、decoder start、label 格式、beam 對應與 validation 歷史沒有問題。
3. 對齊論文的模型容量、訓練預算與解碼協定，確認指標的可比性；模型定案後才對 test 做一次最終評估。

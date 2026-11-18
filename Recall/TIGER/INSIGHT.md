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
- **新 SID 的 TIGER run `tiger-paper-v1`**：SoC job 1163050（`STAGE=materialize`）與 job 1163052（`STAGE=train`），程式 `ea3fb5f`，SID `tiger-paper-v1`（`stable` 配方，256/256/256），T5 4,815,488 參數（新 SID 碰撞 token 較少，詞表較小）。log `slurm-kuriyo-tiger-1163052.out` 在 `00d4470` 進入 repo。validation（22,363 examples）Recall@10 依序為 20k 0.0132、40k 0.0174、60k 0.0231、80k 0.0271、100k 0.0285、120k 0.0310、140k 0.0304、160k 0.0324、180k 0.0361、200k 0.0368；step 200,000：Recall@5=0.021822、Recall@10=0.036801、NDCG@5=0.013492、NDCG@10=0.018354，invalid_beam_fraction 0，最終 train loss 1.694。舊 SID 同 step 為 Recall@10 0.0342、NDCG@10 0.0168。新 SID 第一層碼分布較平均，每個 token 的不確定性較高，所以兩個 run 的 loss 不可直接比較。尚未執行 test。
- **Adafactor 設定與論文框架不同**：`Recall/TIGER/baseline/train.py` 以 Hugging Face `Adafactor(lr=0.01, scale_parameter=False, relative_step=False)` 訓練。HF 在 `scale_parameter=False` 時每步更新量約為 lr 的絕對值（更新先以 RMS 正規化、clip 到 1）。T5X `t5x/adafactor.py` 的 `multiply_by_parameter_scale` 預設為 True，把 lr 乘上參數 RMS（下限 epsilon2=1e-3）；docstring 寫明 False 時 lr 是絕對步長。HF T5 初始化下，encoder `q.weight` 的 RMS 約 0.011、`k`／`v`／FFN `wi` 約 0.088、`o` 約 0.051、embedding 約 1.0。因此 Kuriyo 前 10k steps 每步約改動 `q` 權重自身大小的 90%，200k 時（lr 0.0022）仍約 20%；T5X 對同一權重的步長約 1e-4。
- 本地序列 smoke test 只取 128 筆 train、3 筆 validation、訓練 1 epoch；Recall@5/10 與 NDCG@5/10 均為 0。此數字僅驗證執行與指標流程，不能代表推薦品質；產物在 `Recall/TIGER/baseline/runs/smoke-sequence-from-final-sid/`。

## Limitations

- 現有完整 run `tiger-paper-20261117-005232` 使用舊 SID（三層使用率約 2.34%/10.16%/9.77%）；新 SID `tiger-paper-v1` 的完整 run 見 Facts。兩個完整 run 都用 `scale_parameter=False` 的 Adafactor（見 H5）。
- 原有小型序列模型、訓練預算及限制商品目錄的解碼設定尚未對齊 TIGER 論文；序列 smoke test 只涵蓋極少量樣本，因此不能把目前的指標當作論文重現結果。
- HF T5 已在完整資料上跑到 200,000 steps；log 實測參數量為 5,587,968，取代早期約 470 萬的估值。與論文 T5X 的容量、hash、訓練細節及商品文句仍需逐項核對；未取得最終 test，不可宣稱數值復現。

## Hypothesis

- 可在固定切分與相同模型預算下，比較 RQ-VAE SID、隨機 ID 與其他內容編碼方式對召回品質的影響；尚未進行此比較。
- **H3 — SID 品質是與論文差距的主因（目前證據不支持）：**新 SID 跑滿 200k 的 Recall@10 0.0368，只比舊 SID 的 0.0342 略高（標準誤約 0.0013）；同切分的 SASRec 達 0.0938（[Recall INSIGHT](../INSIGHT.md)）。差距較可能在 TIGER 的序列模型、訓練樣本或解碼。原假設：現行 SID 由發散後的 RQ-VAE 產生（PyTorch Adagrad accumulator 預設 0；論文框架的 Adagrad 通常預設 0.1），第一層只用 6 碼。用診斷中 E 或 F 設定（[SID INSIGHT](../../Representations/SID/INSIGHT.md) 的診斷表）重訓 SID、其他條件不變，預期 TIGER validation Recall@10 會明顯高於 0.0342。驗證方式：新 SID_VERSION、同一 T5 設定，先比較相同 step（如 20k／60k）的 validation，再決定是否跑滿 200k。
- **H4 — 模型容量不足是次要因素：**HF T5 實測 5,587,968 參數，論文約 1,300 萬。差異可能來自 T5X 不共享輸入輸出 embedding 或詞表大小，尚未核實；在 H3 驗證前不先調整。
- **H5 — Adafactor 步長是 TIGER 差距的主因：**Kuriyo 以 `scale_parameter=False` 把論文的 lr 0.01 當成絕對步長，小初始化的權重（如 `q`）每步被大幅改寫，訓練噪音大、收斂慢。兩個完整 run 的 validation 到 200k 仍在上升（約 390 個 epoch），train loss 約 1.69／token（約 8.5 nats／商品），SASRec 約 7.0 nats 且 30 epochs 內就收斂，與此相符。驗證方式：`ADAFACTOR_SCALE_PARAMETER=1`（`--adafactor-scale-parameter`，T5X 行為）、相同 SID `tiger-paper-v1` 與新 RUN_NAME，比較 20k／40k validation 與 `tiger-paper-v1`（Recall@10 0.0132／0.0174）。若明顯較高，H5 成立。

## Discussion

- [TIGER 論文](https://arxiv.org/abs/2305.05065) 使用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games 資料，依使用者時間序列預測下一個商品；保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。目前下載的 2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。
- 早期 smoke trainer 使用 catalog-constrained beam search；完整 HF T5 run 使用 unconstrained beam search，十次 validation 的 invalid_beam_fraction 都為 0。兩種入口不可混為同一設定；後續調整仍先看 validation，模型定案後才看 test。

## Problems to solve

1. 驗證 H5：SoC 以 `ADAFACTOR_SCALE_PARAMETER=1`、SID `tiger-paper-v1`、新 RUN_NAME 跑 `STAGE=train`，比較 20k／40k validation。
2. 若 H5 不成立，繼續找 TIGER 與同切分 SASRec（Recall@10 0.0938）的差距：小型 overfit 測試（本機負載過高未完成）、訓練樣本、token 排列、beam search 解碼。程式碼檢查已確認 token offset、user token、decoder start、label 格式、beam 對應與 validation 歷史沒有問題。
3. 保留已完成的完整 run 作為對照，做單因素比較（熱門基線見 [Recall INSIGHT](../INSIGHT.md)）；模型定案後再對 test 做一次最終評估。
4. 對齊論文的模型容量、訓練預算與解碼協定，確認指標的可比性。

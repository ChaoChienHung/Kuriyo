# SID — Research Notes

與模型無關的商品 SID 產生（`Representations/SID/`）的研究筆記。程式運作與陷阱見 [docs 的 sid-rqvae-training](../../docs/subsystems/sid-rqvae-training.md)；SID 對 TIGER 指標的影響記在 [TIGER INSIGHT](../../Recall/TIGER/INSIGHT.md)。

格式：`Insights`（從資料或實驗得到的觀察）、`Facts`（已確認的資料、實作與 run 結果）、`Limitations`（當前缺點）、`Hypothesis`（待驗證主張）、`Discussion`、`Problems to solve`；事實與假設分開。假設編號只在本檔內有效，跨檔引用時寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H3）。

*建立日期：2026-11-18（由根 INSIGHT 分出）。*

## Insights

- PyTorch Adagrad 的 accumulator 預設 0，照抄論文 lr 0.4 會讓第一步發散、三層塌縮；即使 optimizer 穩定，Sentence-T5 向量的大共同分量仍讓 RQ-VAE 只學到平均 embedding。embedding 標準化加上重設未使用碼才同時解決兩者（見 Facts）。
- `stable` 配方的總 loss 隨訓練上升，原因是 latent 尺度擴大使量化誤差變大，重建仍持續下降；判斷 SID 好壞要看重建、碼使用數與碰撞，不能只看總 loss。

## Facts

- [`Representations/SID/scripts/generate_sids.py`](../../Representations/SID/scripts/generate_sids.py) 可用 Sentence-T5 與 RQ-VAE 產生 SID，目標輸出為 `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/<version>/`。[`scripts/materialize_amazon_sids.py`](../../scripts/materialize_amazon_sids.py) 可將完整商品 SID 對照表套用到 train/validation/test，並按 SID 版本隔離樣本；目前 `sentence-t5_kmeans` 與 `sentence-t5_lsh` 只有預留目錄，尚無生成器。
- 本機已下載 Sentence-T5 base 並用 64 件商品、1 epoch 完成 Sentence-T5 + RQ-VAE smoke test；三層 codebook 使用碼數為 3、1、1（容量 16），顯示此短跑發生 collapse。產物僅在忽略目錄 `Recall/TIGER/baseline/runs/sentence-t5-smoke-64/`，沒有把這個不完整 SID 目錄當作正式 split 資料。
- 本地 TF-IDF/SVD smoke test 對完整 12,101 件 Beauty 商品訓練 RQ-VAE 1 epoch，三層 codebook 的使用碼數分別為 27、25、15（每層容量 32），546 組三碼 SID 發生碰撞；附加第四碼後每件商品的 SID 唯一。產物在本地忽略目錄 `Recall/TIGER/baseline/runs/smoke-beauty-final/`。
- **RQ-VAE 診斷實驗（2026-11-18，本機 CPU）：**以 Sentence-T5 base 重新編碼完整 12,101 件商品（L2 normalize；每維均方 0.0013，即恆輸出 0 的重建 MSE），沿用 `Representations/SID/rqvae.py` 與 `generate_sids.initialize_codebooks`（隨機 1,024 件 k-means 初始化）、batch 1,024、seed 42。指令為 `python -m Representations.SID.scripts.diagnose_rqvae`；程式版本為 `495076a` 之後的工作樹（`Representations/SID/rqvae.py` 未改）。結果：

  | 設定 | epochs | 重建 MSE | 三層使用碼數 | 不重複三碼 SID | 落入碰撞的商品 | 最大碰撞群 |
  |---|---|---|---|---|---|---|
  | 初始化後（未訓練） | 0 | 2.33e-3 | 256/256/256 | 11,957 | 278 | 5 |
  | A. PyTorch Adagrad lr 0.4、accumulator 0（現行 `--paper-baseline`） | 300 | 4.1e-3 | **1/1/1** | 1 | 12,101 | 12,101 |
  | B. Adagrad lr 0.4、accumulator 0.1 | 300 | 2.20e-4 | **1**/254/243 | 9,119 | 5,006 | 13 |
  | C. AdamW lr 1e-3 | 300 | 2.20e-4 | **1/1/1** | 1 | 12,101 | 12,101 |
  | D. B + embedding 逐維標準化 | 60 | 0.93（標準化尺度） | 55/200/228，持續下降 | 4,591 | 8,735 | 550 |
  | E. B + 每 epoch 重設未使用碼 | 60 | 2.21e-4 | 256/256/256 | 11,879 | 428 | 6 |
  | F. AdamW + 標準化 + 重設未使用碼 | 60 | 0.54（標準化尺度，仍在下降） | 256/256/256 | 11,928 | 317 | 6 |

  A 在第 1 個 epoch latent 範數由 0.32 變成 5,260，loss 5.1e6，與 SoC log 第一 epoch 8.7e9 一致。B、C 的重建 MSE 收斂到約 2.2e-4，等於所有商品都輸出平均 embedding 的水準，代表 decoder 沒有用到碼本資訊。只有 F 的重建誤差明顯低於「輸出平均值」（標準化尺度下 0.54 < 1.0），同時保持全碼本使用。各設定只跑一個 seed，60／300 epochs 遠少於論文 20,000 epochs。
- **SID 配方實作後的驗證（2026-11-18，本機，60 epochs，seed 42）：**`generate_sids` 新增 `--recipe paper-pytorch|paper-tf-adagrad|stable`，以 `diagnose_rqvae --recipe` 套用正式程式碼重跑，結果與上表一致：`paper-pytorch` 1/1/1、12,101 件碰撞；`paper-tf-adagrad` 256/256/256、346 件碰撞、最大群 5，重建 2.21e-4（仍只到平均值水準）；`stable` 256/256/256、313 件碰撞、最大群 6，標準化重建 0.551 且持續下降。尚未以 20,000 epochs 或 TIGER 訓練驗證。
- **`stable` 配方的 SoC 完整 SID（2026-11-18，job 1162430，commit `728f052`，SID_VERSION `tiger-paper-v1`）：**Log 顯示 20,000 epochs 完成，碼本使用 256/256/256、碰撞群組 279（舊 run 1153307 為 6/26/25、228 群組；舊 run 碼組合最多 6×26×25=3,900 種，群組數不能直接比）。總 loss 由 0.605（epoch 100）升至 0.829（epoch 20,000）。本機同配方 1,500 epochs（seed 42）分項：重建 0.483→0.387 持續下降，碼本／commitment 0.085→0.249 上升，latent 範數 3.80→6.35，碰撞 345→554 件、最大群 4→11；總 loss 上升來自 latent 尺度擴大而非重建變差。尚未在 TIGER 上驗證新 SID 的效果。

## Limitations

- `sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有預留目錄。商品目錄涵蓋三個 split；若要評估未見商品的歸納能力，需另訂只用 train 商品訓練 SID 的協定。
- 64 件商品、1 epoch 的 Sentence-T5 + RQ-VAE 試跑出現 codebook 使用率低的情況（三層各使用 3、1、1 個碼，容量各 16）；仍須在完整資料與足夠訓練步數下確認是否持續發生。
- `generate_sids` 的 `train.jsonl` 只記總 loss，分項 loss 要用 `diagnose_rqvae` 在本機重跑；SID 訓練沒有中途 checkpoint。

## Hypothesis

- 尚無。

## Discussion

- 尚無。

## Problems to solve

1. 在 `train.jsonl` 記錄重建、碼本、commitment 分項 loss 與 latent 範數，讓 SoC run 不必在本機重現就能判讀。
2. 加入 SID 訓練的 checkpoint／resume。

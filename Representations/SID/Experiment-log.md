# SID — Experiment Log

與模型無關的商品 SID 產生（`Representations/SID/`）的實驗日誌，最新的日期在最上面。超出單一 run 的結論整理到 [INSIGHT.md](INSIGHT.md)；SID 對 TIGER 指標的影響記在 [TIGER Experiment-log](../../Recall/TIGER/Experiment-log.md)。

所有實驗都用 2014 Beauty 的完整商品目錄（12,101 件，`catalog/Beauty/items.jsonl`），三層 RQ-VAE、每層 256 碼。2026-11-18 以前的條目是事後依 log 補記；Expectation 依當時的動機補寫，不參考後來的結果。

## 2026-11-18

### `stable` 配方的 SoC 完整 SID（`tiger-paper-v1`）

#### Motivation

本機診斷顯示 `stable` 配方（AdamW、embedding 標準化、重設未使用碼）能同時避免發散與塌縮。在論文的 20,000 epochs 下產生正式 SID，供 TIGER 檢驗 [TIGER INSIGHT](../../Recall/TIGER/INSIGHT.md) H3。

#### Details

- **Baseline:** job 1153307 的舊 SID（下方 2026-11-17 條目）。
- **Change:** `--recipe stable`；其餘（Sentence-T5 base、三層 256 碼、20,000 epochs）相同。
- **Data:** 2014 Beauty 完整商品目錄。
- **Environment:** SoC job 1162430，程式 `728f052`，SID_VERSION `tiger-paper-v1`。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 STAGE=sid sbatch --export=ALL scripts/run.sbatch`
- **Artifacts:** SoC 的 `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/tiger-paper-v1/`；log 用 `git show b7a4078:slurm-kuriyo-tiger-1162430.out` 取回。

#### Settings

- **架構：**Sentence-T5 base 向量，MLP 512／256／128，latent 32，三層各 256 碼，k-means 以隨機 1,024 件商品初始化；batch 1,024，seed 42。
- **配方 `stable`：**AdamW lr 1e-3，embedding 逐維標準化，每 epoch 重設未使用碼；20,000 epochs。

#### Expectation

三層全碼本使用，碰撞少，重建明顯低於「輸出平均 embedding」的水準。

#### Key Metrics

- 三層使用碼數、碰撞群組數、總 loss 走勢。

#### Results

| Run | 三層使用碼數 | 碰撞群組 | 總 loss | Notes |
| --- | --- | --- | --- | --- |
| 舊 SID | 6/26/25 | 228 | 8.7e9 → 0.178 | job 1153307 |
| `stable` | 256/256/256 | 279 | 0.605（epoch 100）→ 0.829（epoch 20,000） | job 1162430 |

- **Observations:** 舊 SID 的碼組合最多 6×26×25=3,900 種，群組數不能直接比。總 loss 上升；本機同配方 1,500 epochs（seed 42）的分項顯示重建 0.483→0.387 持續下降，碼本／commitment 0.085→0.249 上升，latent 範數 3.80→6.35，碰撞 345→554 件、最大群 4→11。
- **Hypotheses:** 總 loss 上升來自 latent 尺度擴大，不是重建變差。

#### Conclusion

採用 `tiger-paper-v1` 作為 TIGER 的新 SID。它對 TIGER 的效果見 TIGER Experiment-log 2026-11-18（H3 不被支持）。

### SID 配方實作後的驗證

#### Motivation

把診斷得到的修正寫成 `generate_sids --recipe paper-pytorch|paper-tf-adagrad|stable`，確認正式程式碼重現診斷結果。

#### Details

- **Baseline:** 同日「RQ-VAE 塌縮診斷」的 A、B、F 設定。
- **Change:** 改用正式程式碼的配方。
- **Data:** 2014 Beauty 完整商品目錄的 Sentence-T5 embedding。
- **Environment:** 本機，60 epochs，seed 42。
- **Command:** `python -m Representations.SID.scripts.diagnose_rqvae --emb <SID 目錄>/embeddings.npz --recipe <配方> --epochs 60 --report 20`
- **Artifacts:** 只有終端輸出，沒有保存產物。

#### Settings

- **架構：**Sentence-T5 base 向量，MLP 512／256／128，latent 32，三層各 256 碼，k-means 以隨機 1,024 件商品初始化；batch 1,024，seed 42。
- **配方：**`paper-pytorch`（Adagrad lr 0.4、accumulator 0）、`paper-tf-adagrad`（Adagrad lr 0.4、accumulator 0.1、重設未使用碼）、`stable`（同上一條）；各 60 epochs。

#### Expectation

三個配方分別重現 A、B（+accumulator）、F 的碼使用與重建。

#### Key Metrics

- 三層使用碼數、碰撞商品數、最大碰撞群、重建誤差。

#### Results

| 配方 | 三層使用碼數 | 碰撞商品 | 最大群 | 重建 |
| --- | --- | --- | --- | --- |
| `paper-pytorch` | 1/1/1 | 12,101 | 12,101 | — |
| `paper-tf-adagrad` | 256/256/256 | 346 | 5 | 2.21e-4（平均值水準） |
| `stable` | 256/256/256 | 313 | 6 | 0.551（標準化尺度，仍在下降） |

- **Observations:** 與診斷表一致。
- **Hypotheses:** 無。

#### Conclusion

配方實作正確；以 `stable` 送 SoC 完整訓練（上一條）。

### RQ-VAE 塌縮診斷

#### Motivation

job 1153307 的 SID 第一個 epoch loss 8.7e9，三層只用 6/26/25 碼。找出發散與塌縮的原因。

#### Details

- **Baseline:** 現行 `--paper-baseline`（PyTorch Adagrad lr 0.4、accumulator 0）。
- **Change:** 逐一改 optimizer、accumulator 初值、embedding 標準化、重設未使用碼（見結果表）。
- **Data:** 以 Sentence-T5 base 重新編碼完整 12,101 件商品（L2 normalize；每維均方 0.0013，即恆輸出 0 的重建 MSE）。
- **Environment:** 本機 CPU，seed 42，batch 1,024，`generate_sids.initialize_codebooks`（隨機 1,024 件 k-means 初始化）；程式為 `495076a` 之後的工作樹（`rqvae.py` 未改）。
- **Command:** `python -m Representations.SID.scripts.diagnose_rqvae`
- **Artifacts:** 只有終端輸出。

#### Settings

- **架構：**Sentence-T5 base 向量，MLP 512／256／128，latent 32，三層各 256 碼，k-means 以隨機 1,024 件商品初始化；batch 1,024，seed 42。
- **各設定：**optimizer、accumulator、標準化與重設未使用碼見結果表 A–F；A–C 跑 300 epochs，D–F 跑 60 epochs。

#### Expectation

A 應重現 SoC 的第一 epoch 發散與塌縮。若主因是 Adagrad accumulator 從 0 開始，B（accumulator 0.1）應消除發散並恢復三層碼使用；若 B 仍有某層塌縮，代表還有第二個原因，標準化（D）或重設未使用碼（E）應能補上。換成 AdamW（C）若同樣塌縮，問題就不只是 optimizer。

#### Key Metrics

- 重建 MSE 是否低於「所有商品輸出平均 embedding」的水準（約 2.2e-4）、三層使用碼數、碰撞。

#### Results

| 設定 | epochs | 重建 MSE | 三層使用碼數 | 不重複三碼 SID | 落入碰撞的商品 | 最大碰撞群 |
| --- | --- | --- | --- | --- | --- | --- |
| 初始化後（未訓練） | 0 | 2.33e-3 | 256/256/256 | 11,957 | 278 | 5 |
| A. PyTorch Adagrad lr 0.4、accumulator 0 | 300 | 4.1e-3 | **1/1/1** | 1 | 12,101 | 12,101 |
| B. Adagrad lr 0.4、accumulator 0.1 | 300 | 2.20e-4 | **1**/254/243 | 9,119 | 5,006 | 13 |
| C. AdamW lr 1e-3 | 300 | 2.20e-4 | **1/1/1** | 1 | 12,101 | 12,101 |
| D. B + embedding 逐維標準化 | 60 | 0.93（標準化尺度） | 55/200/228，持續下降 | 4,591 | 8,735 | 550 |
| E. B + 每 epoch 重設未使用碼 | 60 | 2.21e-4 | 256/256/256 | 11,879 | 428 | 6 |
| F. AdamW + 標準化 + 重設未使用碼 | 60 | 0.54（標準化尺度，仍在下降） | 256/256/256 | 11,928 | 317 | 6 |

- **Observations:** A 在第 1 個 epoch latent 範數由 0.32 變成 5,260，loss 5.1e6，與 SoC log 第一 epoch 8.7e9 一致。B、C 的重建收斂到約 2.2e-4，等於輸出平均 embedding，decoder 沒用到碼本資訊。只有 F 的重建明顯低於平均值水準（標準化尺度 0.54 < 1.0），同時全碼本使用。
- **Hypotheses:** 各設定只有一個 seed，60／300 epochs 遠少於 20,000；長訓練下是否維持未驗證。

#### Conclusion

原因有兩個：PyTorch Adagrad accumulator 預設 0 造成第一步發散；Sentence-T5 向量的大共同分量讓 RQ-VAE 只學到平均值。整理到 [SID INSIGHT](INSIGHT.md) Insights，並實作成 `--recipe`（上一條）。

## 2026-11-17

### 首次完整 SID（job 1153307）

#### Motivation

依論文設定（Sentence-T5、三層 256 碼、20,000 epochs、Adagrad 0.4）產生完整 SID，供第一個完整 TIGER run 使用。

#### Details

- **Baseline:** 無。
- **Change:** —
- **Data:** 2014 Beauty 完整商品目錄。
- **Environment:** SoC job 1153307，程式 `5f99955`（`STAGE=all` 的 SID 階段）。
- **Command:** 見 [TIGER Experiment-log](../../Recall/TIGER/Experiment-log.md) 2026-11-17。
- **Artifacts:** SoC 的 `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/tiger-paper-20261117-005232/`；log 用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回。

#### Settings

- **架構：**Sentence-T5 base 向量，MLP 512／256／128，latent 32，三層各 256 碼，k-means 以隨機 1,024 件商品初始化；batch 1,024，seed 42。
- **配方（相當於現在的 `paper-pytorch`）：**PyTorch Adagrad lr 0.4、accumulator 0，不標準化、不重設碼；20,000 epochs。

#### Expectation

依論文設定，12,101 件商品對三層各 256 碼應有足夠的多樣性，大部分碼都會被使用，三碼 SID 碰撞少；loss 從初始值平穩下降，不出現數量級的跳動。

#### Key Metrics

- 三層使用碼數、碰撞群組、loss。

#### Results

- **Observations:** 三層使用 6/26/25 碼（約 2.34%／10.16%／9.77%），碰撞群組 228（不能換算成碰撞商品比例）。第一 epoch loss 8,743,993,902.95，最後 epoch 0.178358；log 沒有分項 loss。
- **Hypotheses:** 第一 epoch 的巨大 loss 與低碼本使用可能有關（之後由 2026-11-18 的診斷確認）。

#### Conclusion

SID 塌縮，列為 TIGER 差距的主要嫌疑並安排診斷。

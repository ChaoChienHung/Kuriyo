# TIGER baseline — Experiment Log

`Recall/TIGER/baseline/` 的實驗日誌，最新的日期在最上面。超出單一 run 的結論整理到家族的 [INSIGHT.md](../INSIGHT.md)；SID 本身的實驗記在 [SID Experiment-log](../../../Representations/SID/Experiment-log.md)。

所有 run 都用 2014 Beauty 切分（131,413 train／22,363 validation，歷史上限 20）、seed 42（`train.py` 預設），validation 以全 catalog 的 unconstrained beam search（beam 10）評估。2026-11-19 以前的條目是事後依 log 補記；Expectation 依當時的動機補寫，不參考後來的結果。

## 2026-11-19

### T5X 式 Adafactor 步長（H5）

#### Motivation

同切分的 SASRec validation Recall@10 0.0938，TIGER 不論新舊 SID 都只有 0.034–0.037。程式碼檢查發現 Adafactor 以 `scale_parameter=False` 把 lr 0.01 當成絕對步長，與論文框架 T5X 不同。檢驗 [TIGER INSIGHT](../INSIGHT.md) H5。

#### Details

- **Baseline:** `tiger-paper-v1`（SoC job 1163052，程式 `ea3fb5f`），見下方 2026-11-18 條目。
- **Change:** 只改 `ADAFACTOR_SCALE_PARAMETER=1`（`--adafactor-scale-parameter`，lr 乘上參數 RMS）；SID、模型、資料、seed 都相同。
- **Data:** 2014 Beauty，SID `tiger-paper-v1`（`stable` 配方，256/256/256）。
- **Environment:** SoC job 1165519，節點 xgpf11，程式 `bb1088c`，seed 42，T5 4,815,488 參數。第一次提交的 job 1165195 落在 xgpe2，CUDA 初始化失敗，沒有訓練。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-v1 ADAFACTOR_SCALE_PARAMETER=1 STAGE=train sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-v1/`；log `slurm-kuriyo-tiger-1165519.out`（repo 根目錄）。

#### Expectation

若步長是主因，20k、40k 的 validation Recall@10 會明顯高於 baseline 的 0.0132、0.0174，而且收斂所需的 steps 明顯變少。

#### Key Metrics

- validation Recall@10、NDCG@10，step 20k 與 40k。
- 明顯高於 baseline 即支持 H5，接著跑滿 200k；與 baseline 相近則不支持，改查訓練樣本、token 排列與解碼。

#### Results

| Run | train loss @1k | train loss @1.5k | Recall@10 @20k | Recall@10 @40k | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline | 3.362 | 3.127 | 0.0132 | 0.0174 | `tiger-paper-v1`，job 1163052 |
| T5X Adafactor | 2.36 | 2.144 | 待定 | 待定 | `tiger-t5x-adafactor-v1`，job 1165519（訓練中） |

- **Observations:** 截至 log 的 step 1,500，train loss 比 baseline 同 step 低約 1.0。兩者 SID 相同，loss 可以直接比，但 train loss 下降快不代表 validation 會更好。
- **Hypotheses:** 尚無。

#### Conclusion

進行中，等 20k／40k validation。

## 2026-11-18

### 新 SID `tiger-paper-v1` 跑滿 200k（H3）

#### Motivation

舊 SID 在第一個 epoch 發散，三層只用到 6/26/25 個碼。檢驗 [TIGER INSIGHT](../INSIGHT.md) H3：SID 品質是與論文差距的主因。

#### Details

- **Baseline:** `tiger-paper-20261117-005232`（job 1153307，舊 SID），見 2026-11-17 條目。
- **Change:** 改用 `stable` 配方的新 SID `tiger-paper-v1`（[SID Experiment-log](../../../Representations/SID/Experiment-log.md) 2026-11-18，job 1162430）；T5 設定不變。新 SID 的碰撞 token 較少，詞表較小，T5 參數由 5,587,968 變成 4,815,488。
- **Data:** 2014 Beauty，SID `tiger-paper-v1`。
- **Environment:** SoC job 1163050（`STAGE=materialize`）與 job 1163052（`STAGE=train`），程式 `ea3fb5f`，seed 42。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-paper-v1 STAGE=train sbatch --export=ALL scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/TIGER/baseline/runs/tiger-paper-v1/`；log 已從工作目錄移除，用 `git show b7a4078:slurm-kuriyo-tiger-1163052.out` 取回。

#### Expectation

換成全碼本使用的 SID 後，validation Recall@10 會明顯高於舊 SID 的 0.0342。

#### Key Metrics

- validation Recall@10、NDCG@10，先比 20k／60k，再看 200k。
- 明顯高於 0.0342（超出標準誤約 0.0013 數倍）才算支持 H3。

#### Results

| step | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| 20k | 0.00939 | 0.01319 | 0.00593 | 0.00718 |
| 40k | 0.01105 | 0.01744 | 0.00696 | 0.00901 |
| 60k | 0.01391 | 0.02307 | 0.00872 | 0.01176 |
| 80k | 0.01614 | 0.02714 | 0.00979 | 0.01336 |
| 100k | 0.01596 | 0.02848 | 0.01005 | 0.01414 |
| 120k | 0.01672 | 0.03099 | 0.01058 | 0.01527 |
| 140k | 0.01851 | 0.03036 | 0.01134 | 0.01522 |
| 160k | 0.01887 | 0.03242 | 0.01179 | 0.01622 |
| 180k | 0.02066 | 0.03613 | 0.01266 | 0.01766 |
| **200k** | **0.021822** | **0.036801** | **0.013492** | **0.018354** |
| 舊 SID 200k | 0.020570 | 0.034208 | 0.012409 | 0.016788 |

- **Observations:** 200k 只比舊 SID 高 0.0026 Recall@10；曲線到 200k 仍在上升。invalid_beam_fraction 每次都是 0，最終 train loss 1.694。新 SID 第一層碼分布較平均，每個 token 不確定性較高，兩個 run 的 loss 不能直接比。
- **Hypotheses:** 差距較可能在序列模型的訓練（後來寫成 H5）、訓練樣本或解碼。

#### Conclusion

不支持 H3。保留 `tiger-paper-v1` 作為之後 TIGER 實驗的 baseline；下一步是 SASRec 同切分檢查與 H5。沒有跑 test。

## 2026-11-17

### 首次完整 paper-configured run（job 1153307）

#### Motivation

在完整 Beauty 上跑出第一個依論文公開設定的 TIGER，作為之後所有比較的起點。

#### Details

- **Baseline:** 無（首次完整 run）；比較目標是 TIGER 論文的 Beauty test 數字（Recall@10 0.0648）。
- **Change:** —
- **Data:** 2014 Beauty，SID 與 RUN_NAME 都是 `tiger-paper-20261117-005232`（`paper-pytorch` 行為，舊 SID）。
- **Environment:** SoC job 1153307，程式 `5f99955`，seed 42，T5 5,587,968 參數。
- **Command:** `SID_VERSION=tiger-paper-20261117-005232 RUN_NAME=tiger-paper-20261117-005232 STAGE=all sbatch --export=ALL scripts/run.sbatch`（依 log 記錄的設定重建；SID 20,000 epochs → materialize → TIGER 200,000 steps）
- **Artifacts:** SoC 舊路徑 `Recall/TIGER/runs/tiger-paper-20261117-005232/`（續訓或 test 前需搬到 `baseline/runs/`）；log 用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回；本機匯入紀錄在 `runs/tiger-paper-20261117-005232/imports/slurm-1153307/`（重建指令見 [HANDOFF](../../../HANDOFF.md)）。

#### Expectation

依論文公開設定實作，validation Recall@10 應與論文 Beauty 的 0.0648 同一量級（validation 與 test 同協定、只差最後一筆），至少明顯高於熱門商品基線。若遠低於論文，代表實作或設定和論文有落差，需要逐項排查。

#### Key Metrics

- validation Recall@5/10、NDCG@5/10，每 20k steps 一次，以 NDCG@10 選最佳。

#### Results

| Run | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Notes |
| --- | --- | --- | --- | --- | --- |
| step 200k（最佳） | 0.020570 | 0.034208 | 0.012409 | 0.016788 | `tiger-paper-20261117-005232` |
| 論文（test） | 0.0454 | 0.0648 | 0.0321 | 0.0384 | 不同 split，僅作參考 |

- **Observations:** 十次 validation 中最佳 NDCG@10 在最後一次（200k），曲線不是嚴格單調；invalid_beam_fraction 都是 0。log 最後印出 Saved；沒有 sacct 成功狀態或 checkpoint 校驗。SID 三層只用 6/26/25 碼（見 SID Experiment-log）。
- **Hypotheses:** SID 塌縮可能是差距主因（後來寫成 H3）；模型容量只有論文約 1,300 萬的一半以下（H4）。

#### Conclusion

取得第一個可比較的 baseline，validation Recall@10 0.0342，約為熱門商品基線（0.0170）的 2 倍，遠低於論文。下一步先查 SID。沒有跑 test。

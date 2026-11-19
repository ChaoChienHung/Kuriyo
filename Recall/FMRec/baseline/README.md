# FMRec baseline

本目錄是 FMRec 家族的 `baseline` 變體，擁有自己完整的程式、依賴、測試與 runs；不 import 其他變體或 `Recall/TIGER/`，只用 `Recall/core/`、`core/`。`baseline` 代表比較起點，**不代表已重現論文數值**。家族說明見 [FMRec 家族 README](../README.md)。

論文：Feng Liu et al., **[Flow Matching based Sequential Recommender Model](https://arxiv.org/abs/2505.16298)**，IJCAI 2025。作者程式：[FengLiu-1/FMRec](https://github.com/FengLiu-1/FMRec)（核對版本 commit `1c39b17`）。本目錄是 Kuriyo 的 PyTorch 重新實作，細節以作者程式為準；與作者程式不同之處列在下方「差異」。

目的：在 Kuriyo 的同一份 2014 Beauty leave-one-out 切分、同一個指標函式下，把 FMRec 與 [TIGER baseline](../../TIGER/baseline/README.md) 放在一起比較。FMRec 論文的 Beauty 結果（HR@5 5.84%、HR@10 8.27%、NDCG@5 4.16%、NDCG@10 4.95%）使用它自己的切分與最長 50 件歷史，也沒有和 TIGER 比較。

**目前狀態**：可執行；單元／整合測試與 CPU 小樣本 smoke 已通過。尚未做完整 GPU 訓練，沒有 validation 或 test 結果。

## 方法（對照作者程式）

- **商品表示**：可訓練的 item embedding（index 0 為 padding），經 dropout（0.3）與 LayerNorm；沒有位置 embedding（作者程式把位置 embedding 註解掉了）。`model.py` 的 `FMRec.encode_history`。
- **Flow**：z_t = t·x_target + (1−t)·z_noise，t=1 是資料、t=0 是雜訊（論文寫成 (1−t)x_c + t·x_n 的方向相反，作者程式用本方向）。網路做 **x₀ 預測**：直接輸出目標商品 embedding。
- **時間取樣**：mode sampling，t = 1 − u − s·(cos²(πu/2) − 1 + u)，再縮放到 [eps, 1]，eps = 0.001。時間以 t·(1/eps − 1) 進正弦 embedding。
- **條件注入**：x_t 與時間 embedding 的和，乘上逐元素權重 λ ~ N(0.001, 0.001)，加到每個歷史位置（作者稱 uncertainty injection）。
- **Decoder1／Decoder2**：同一個 Transformer stack 的前 2 層與後 2 層（作者 `num_blocks=4, last=2`），hidden 128、4 heads、dropout 0.1、pre-norm、GELU FFN 4×。讀最後一個位置：Decoder2 輸出經 LayerNorm 為 x₀ 預測；Decoder1 輸出接 MLP 128→512→2048→N（tanh）重建歷史商品集合。
- **Loss**：L_FM（x₀ 預測對目標 embedding 的 MSE）+ α·L_CE（x₀ 預測與所有 item embedding 內積的 softmax CE，α=0.2）+ β·L_MSE（歷史 multi-hot 的 balanced MSE，正例加同數量隨機負例，β=0.4）。
- **推論**：Euler ODE 30 步，從雜訊 z₀ 起，速度 = x₀ 預測 − z₀，得到 z₁；以 z₁ 與所有 item embedding 的內積做 full ranking。
- **最佳化**：Adam lr 1e-3、weight decay 0、StepLR 每 100 epochs ×0.1、batch 512、最多 500 epochs、每 20 epochs 做一次 validation、patience 5。

### 時間取樣 s 的選擇

論文 Appendix B.2 對 mode sampling 的 s 有 1.0 與 0.4 兩種說法。作者程式 `--s_modsamp` 預設 **1.0**，本實作 `--mode-scale` 預設也是 1.0，因為釋出的程式與預設值是唯一可以檢查的設定。要測試 0.4 時用 `--mode-scale 0.4` 開新 run；兩者都會記錄在 `config.json` 的 `model.mode_scale`。

## 與作者程式的差異

| 項目 | 作者程式 | 本實作 | 原因 |
| --- | --- | --- | --- |
| 歷史長度 | 最多 50 | 最多 20（`--max-history`） | 沿用 Kuriyo 切分（20 件上限來自 TIGER 論文 Appendix C），和 TIGER 可比 |
| 資料 | ICLRec／DuoRec 處理的 Beauty（22,361 users） | Kuriyo 2014 Beauty 切分（22,363 users、12,101 items） | 同一份切分才能比較 |
| attention mask | 用廣播把 padding 的 **query** 列遮掉，真實位置仍會 attend padding key | 遮掉 padding **key**；每個位置可 attend 自己 | 作者寫法讓輸出隨 padding 長度改變；本實作 padding invariant |
| 因果遮罩 | 無（雙向） | 預設雙向；`--attention causal` 可選 | 預設跟作者一致 |
| padding 商品 | 參與 CE 與排名 | CE 與排名都排除 index 0 | padding 不是可推薦商品 |
| balanced MSE 負例 | 從值為 0 的格子有放回抽樣 | 從所有格子有放回抽樣，抽到正例只是重複 | 避免每步建立 B×N 的 nonzero 索引；正例占比約 0.1%，影響可忽略 |
| 模型選擇 | 任一指標進步就保存；patience 以「沒有任何指標進步」計 | 以 validation NDCG@10 選 `best.pt`；patience 以 NDCG@10 計 | 與 TIGER baseline 一致 |
| 評估雜訊 | 未固定 | validation／test 用 `--seed` 建立的獨立 generator，不影響訓練 RNG | 讓評估可重現，續訓結果不受評估影響 |
| seed | 1997 | 42 | 與 Kuriyo 其他 run 一致；可用 `--seed 1997` |

兩者都不從排名中過濾歷史商品。

## 資料與指標

讀取 `data/amazon_reviews/2014/<train|validation|test>/raw/Beauty/examples.jsonl` 與 `catalog/Beauty/items.jsonl`（產生方式見根目錄 README）。商品 index 依 catalog 檔案順序，1..12,101。訓練用 train 的前綴展開樣本；validation 的歷史不含 validation 目標；test 的歷史包含 validation 商品。切分是 per-user 時間排序 leave-one-out，同時間戳以來源行序決定，歷史最多 20 件。

指標用共用的 `Recall/core/metrics.py`：每個請求只有一個目標，Recall@K 等於論文的 HR@K；NDCG@K 以 1/log2(rank+1) 計。排名是全 catalog（12,101 件），不抽負例。

## 執行

從 **repository 根目錄**執行，依賴只有根目錄共用的 PyTorch／NumPy：

```bash
python -m pip install -r Recall/FMRec/baseline/requirements.txt
```

本地小樣本 smoke（CPU；2026-11-18 在本機與完整測試同時跑，約 8.5 分鐘）：

```bash
python -m Recall.FMRec.baseline.train --run-name fmrec-smoke-$(date +%Y%m%d) \
  --epochs 2 --eval-every 1 --max-train-examples 4096 --max-validation-examples 1024
```

完整訓練（作者預設；需要 GPU。每 epoch 約 257 steps，最多 500 epochs）：

```bash
python -m Recall.FMRec.baseline.train --run-name fmrec-beauty-v1 --device cuda
```

在 SoC 上用 [`scripts/run.sbatch`](../../../scripts/run.sbatch) 的 `MODEL=fmrec` 提交，資源與 partition 先依 [SoC 操作說明](../../../skills/nus-soc-slurm/SKILL.md) 核對。`STAGE=train` 只檢查 Beauty 切分存在就訓練；run 目錄已有 `latest.pt` 時自動續訓；`STAGE=test` 做一次 held-out test：

```bash
MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch
```

續訓（從 `checkpoints/latest.pt`，每個 epoch 結束時保存）：

```bash
python -m Recall.FMRec.baseline.train --run-name fmrec-beauty-v1 --device cuda --resume
```

續訓時資料校驗值、seed、batch size、歷史長度與模型設定必須和 `settings.json` 相同，否則拒絕執行。可加大 `--epochs` 延長；已 early stop 的 run 不會再訓練。

模型定案後，只執行一次 held-out test（拒絕覆寫既有的 `test/<set>/metrics.json`）：

```bash
python -m Recall.FMRec.baseline.train --run-name fmrec-beauty-v1 --device cuda --eval-test-only
```

## 輸出

`Recall/FMRec/baseline/runs/<RUN_NAME>/`（Git 只追蹤 `settings.json`、`events.log`、`validation/`、`test/`）；run 目錄的版面與 `settings.json` 見 [`core/README.md`](../../../core/README.md#run-目錄)。`train/train.jsonl` 每 step 記 loss、loss_fm、loss_ce、loss_mse、learning rate 與 epoch；`settings.json` 的 `run.implementation` 記上游 commit。

模型約 2,840 萬參數，其中約 2,480 萬在歷史重建 MLP 的最後一層（2048×12,102）。

## 測試

```bash
python -m unittest discover -s Recall/FMRec/baseline/tests -t . -v
```

涵蓋：causal 模式不看後面位置、padding key 不被 attend、左側 padding 不改變預測、loss 與梯度有限且傳到各部分、取樣輸出形狀與固定 seed 可重現、mode 時間取樣落在 [eps, 1]、padding 不進排名、評估接到共用指標、訓練／拒絕覆寫／續訓設定不符時拒絕／續訓與不中斷訓練 loss 相同／test 只能跑一次。這些測試用 CPU 與合成小資料，不代表推薦品質。

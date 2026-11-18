# Kuriyo — Research Notes

本文件記錄 Kuriyo 推薦模型專案的研究筆記。`Insights` 記錄從資料或實驗得到的觀察，`Facts` 記錄已確認的資料與實作，`Limitations` 記錄當前資料與實作的缺點，`Hypothesis` 記錄待驗證主張。新增實驗結論時，附上資料版本、切分方式、模型設定、評估指標與結果位置。各節依 General、Recall、Ranking、TIGER 分類；TIGER 是 Recall 內的具體模型。

*建立日期：2026-10-25；狀態更新：2026-11-17。交接摘要見 [HANDOFF.md](HANDOFF.md)。*

## Insights

### General

- `All_Beauty` 的使用者互動極稀疏：631,986 位使用者中，583,553 位只有一筆評論。因此，若要做「每位使用者最後一筆作測試」的離線評估，大部分使用者無法同時出現在訓練與測試中。這是對資料分布的觀察，尚非模型效果結論。
- 2014 `Beauty` 的 9,719 位使用者在 validation 與 test 邊界有相同時間戳。此資料只能用來源行序固定當日內的順序，無法驗證真實互動先後；解讀時間序列結果時須保留這個限制。

### Recall

- 尚無經實驗驗證的 Recall 模型洞見。

### Ranking

- 尚無經實驗驗證的 Ranking 模型洞見。

### TIGER

- 完整 run 的十次 validation 中，最佳已記錄 NDCG@10 在 step 200,000（0.016787758）；不是嚴格單調上升，也尚無 test 結果。SID 三層使用碼數為 6/26/25，值得調查，但其與推薦效果的因果關係尚未驗證。
- **2026-11-18 本機 RQ-VAE 診斷（只看 SID，未重訓 TIGER）：**目前 `--paper-baseline` 的 SID 訓練在第一個 epoch 就發散，且即使改用穩定的 optimizer，第一層 codebook 仍會塌縮。兩者都會讓 SID 幾乎不帶商品語意，是與論文差距的首要嫌疑；但「修好 SID 就能拉高 Recall」仍是假設（見 Hypothesis H3）。

## Facts

### General

- 本 repository 主要用來存放推薦模型，以及相關的資料處理、訓練、評估與實驗紀錄。參見 [README.md](README.md)。
- Amazon Reviews 依年份先分 `2014/`、`2023/`；各年份有來源 `raw/`、共用商品目錄 `catalog/`，以及 `train/`、`validation/`、`test/`。共用商品 SID 目錄位於年份層的 `artifacts/sids/<method>/<category>/<version>/`；各 split 下以 `raw/` 保存 ASIN 樣本，以 `sids/<method>/<category>/<version>/` 保存轉換後的 SID 樣本。預留方法名稱為 `sentence-t5_rqvae`、`sentence-t5_kmeans`、`sentence-t5_lsh`；2023 尚未切分或建立 SID。2014 `Beauty` 的 ASIN 切分結果已在本地產生，資料檔由 Git 忽略。
- `core/`、`utils/`、`scripts/`、`Representations/`、`Recall/`、`Ranking/` 和 `pipelines/` 已建立。2014 與 2023 下載腳本分別位於 `scripts/download_amazon_reviews_2014.py` 和 `scripts/download_amazon_reviews_2023.py`。
- 已下載並驗證 Amazon Reviews 2023 `All_Beauty` 的 701,528 筆評論及 112,590 筆商品 metadata。評論涵蓋 631,986 位使用者、112,565 個商品；這些商品全部能以 `parent_asin` 對應到 metadata。其中 9,159 位使用者有至少三筆評論，共貢獻 39,427 筆評論。統計由本地 JSONL 逐行計算；來源版本和校驗值記在 `data/amazon_reviews/2023/raw/All_Beauty/manifest.json`。
- 已下載並驗證 Amazon Reviews 2014 `Beauty` 5-core 的 198,502 筆評論與 259,204 筆商品 metadata。評論包含 22,363 位使用者與 12,101 個商品；每位使用者至少有五筆評論，所有評論商品都能以 `asin` 對應到 metadata。來源及校驗值記在 `data/amazon_reviews/2014/raw/Beauty/manifest.json`。

### Recall

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。
- 熱門商品基線（2026-11-18，本機）：以每位使用者 train 段落中的所有互動（歷史 + train 目標）計數，固定推薦前 10 名。validation Recall@5/10 = 0.00984 / 0.01704、NDCG@5/10 = 0.00581 / 0.00811；test Recall@5/10 = 0.00729 / 0.01203、NDCG@5/10 = 0.00400 / 0.00552。validation 目標從未出現在該使用者的歷史中（重複率 0）。目前 TIGER validation Recall@10 0.0342 約為熱門基線的 2 倍。

### Ranking

- `Ranking/` 目前只有 [README](Ranking/README.md)。2014 `Beauty` 的現有序列樣本只有下一件商品正例，尚未建立 Ranking 用候選集、負例或排序標籤，也沒有訓練與評估程式。

### TIGER

- 2014 `Beauty` 的 22,363 位使用者與 12,101 個商品，與 [TIGER 論文](https://arxiv.org/abs/2305.05065) 的 Beauty 資料表相符。
- 2014 `Beauty` 切分方式：依使用者與 `unixReviewTime` 升冪排序；時間戳相同時保留原始檔行序。每位使用者的最後一筆作 test、倒數第二筆作 validation，其餘用於 train；訓練段落展開成前綴預測樣本，歷史最多 20 件。共產生 131,413 筆 train、22,363 筆 validation、22,363 筆 test，以及放在 `2014/catalog/Beauty/` 的 12,101 件商品內容資料。規則和來源 SHA-256 記在 `data/amazon_reviews/2014/train/raw/Beauty/split_manifest.json`。
- `Representations/SID/` 已有商品內容向量、三層 RQ-VAE、碰撞處理及 SID 生成器；`Recall/TIGER/` 有 SID 序列 Transformer 與 validation 評估流程。試跑指令與模型限制見 [SID README](Representations/SID/README.md) 與 [TIGER baseline README](Recall/TIGER/baseline/README.md)。
- [`Representations/SID/scripts/generate_sids.py`](Representations/SID/scripts/generate_sids.py) 可用 Sentence-T5 與 RQ-VAE 產生 SID，目標輸出為 `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/<version>/`。[`scripts/materialize_amazon_sids.py`](scripts/materialize_amazon_sids.py) 可將完整商品 SID 對照表套用到 train/validation/test，並按 SID 版本隔離樣本；目前 `sentence-t5_kmeans` 與 `sentence-t5_lsh` 只有預留目錄，尚無生成器。
- [TIGER 論文](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)報告的 Beauty 數值為 Recall@5 0.0454、NDCG@5 0.0321、Recall@10 0.0648、NDCG@10 0.0384。論文使用 Sentence-T5 向量、三層 RQ-VAE（每層 256 碼、20,000 epochs、Adagrad 0.4）與含 2,000 個 hashed user token 的 T5X 序列模型（200,000 steps）。這些數字是比較目標，尚非 Kuriyo 的結果。
- [`scripts/run.sbatch`](scripts/run.sbatch) 可分階段準備 Beauty 資料、訓練 SID、轉換樣本及訓練 PyTorch/Hugging Face T5 baseline；[`scripts/prepare_amazon_reviews_2014_beauty.py`](scripts/prepare_amazon_reviews_2014_beauty.py) 可驗證並重用已下載資料。SoC job 1153307 已執行至 200,000 steps 並印出 Saved；完整 log `slurm-kuriyo-tiger-1153307.out` 曾同步進 repo，已於 `2ad21c7` 移除，可用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回。
- 完整 run `tiger-paper-20261117-005232`（程式 `5f99955`）對 12,101 商品產生 SID，20,000 epochs 結束時使用碼數 6/26/25，碰撞群組 228。第一 epoch loss=8743993902.953560，最後 epoch loss=0.178358；尚無分項 loss 可定位原因。228 群組不能直接換算碰撞商品比例。
- step 200,000 的 validation（22,363 examples）：Recall@5=0.020569691、Recall@10=0.034208290、NDCG@5=0.012409267、NDCG@10=0.016787758；依已記錄 NDCG@10 為最佳，沒有最終 test。資料源為 job 1153307 log（見上一條取回方式），不是本機 smoke 指標。
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
- 本地序列 smoke test 只取 128 筆 train、3 筆 validation、訓練 1 epoch；Recall@5/10 與 NDCG@5/10 均為 0。此數字僅驗證執行與指標流程，不能代表推薦品質；產物在 `Recall/TIGER/baseline/runs/smoke-sequence-from-final-sid/`。

## Limitations

### General

- 2014 `Beauty` 的評論時間戳只精確到日期；9,719 位使用者的 validation 與 test 互動同日。來源行序能讓切分重現，但無法確認當日真實先後，時間序列評估可能受影響。
- 2023 `All_Beauty` 目前只有原始下載檔，尚無可供模型訓練與評估的切分或 SID 資料；2014 與 2023 的結果也不能直接視為同一資料集比較。

### Recall

- 熱門商品基線已在本機算出（見 Facts／Recall），但尚無 SASRec 等序列基線，無法判斷 TIGER 差距有多少來自模型本身。

### Ranking

- 尚未建立候選集、負例、排序標籤及 Ranking 訓練／評估程式，因此目前不能評估排序階段的品質。

### TIGER

- `sentence-t5_rqvae` 已在 SoC 對完整 12,101 件 Beauty 商品產生 SID，但三層使用率僅約 2.34%/10.16%/9.77%，仍需診斷；`sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有預留目錄。現有 SID smoke test 不能用作完整資料集的模型效果結論。商品目錄涵蓋三個 split；若要評估未見商品的歸納能力，仍需另訂只用 train 商品訓練 SID 的協定。
- 64 件商品、1 epoch 的 Sentence-T5 + RQ-VAE 試跑出現 codebook 使用率低的情況（三層各使用 3、1、1 個碼，容量各 16）；仍須在完整資料與足夠訓練步數下確認是否持續發生。
- 原有小型序列模型、訓練預算及限制商品目錄的解碼設定尚未對齊 TIGER 論文；序列 smoke test 只涵蓋極少量樣本，因此不能把目前的指標當作論文重現結果。
- HF T5 已在完整資料上跑到 200,000 steps；log 實測參數量為 5,587,968，取代早期約 470 萬的估值。與論文 T5X 的容量、hash、訓練細節及商品文句仍需逐項核對；未取得最終 test，不可宣稱數值復現。

## Hypothesis

### General

- **H2 — 時間切分更接近實際使用情境：**依互動時間切分資料，可能比隨機切分更能反映模型對未來互動的預測能力。需比較兩種切分的結果與資料洩漏風險。

### Recall

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。

### Ranking

- 尚未提出可測試的 Ranking 假設；需先確定候選集與標籤。

### TIGER

- 可在固定切分與相同模型預算下，比較 RQ-VAE SID、隨機 ID 與其他內容編碼方式對召回品質的影響；尚未進行此比較。
- **H3 — SID 品質是與論文差距的主因：**現行 SID 由發散後的 RQ-VAE 產生（PyTorch Adagrad accumulator 預設 0；論文框架的 Adagrad 通常預設 0.1），第一層只用 6 碼。用診斷中 E 或 F 設定重訓 SID、其他條件不變，預期 TIGER validation Recall@10 會明顯高於 0.0342。驗證方式：新 SID_VERSION、同一 T5 設定，先比較相同 step（如 20k／60k）的 validation，再決定是否跑滿 200k。
- **H4 — 模型容量不足是次要因素：**HF T5 實測 5,587,968 參數，論文約 1,300 萬。差異可能來自 T5X 不共享輸入輸出 embedding 或詞表大小，尚未核實；在 H3 驗證前不先調整。

## Discussion

### General

- **候選資料來源：**已為 [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 `Beauty` 5-core 與 [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) 的 `All_Beauty` 建立下載流程。2014 年版適合對照 TIGER 論文；2023 年版適合探索較新資料。正式用於訓練前須確認資料品質、規模與使用條件。
- **推薦任務：**需決定要預測評分、排序商品，還是做下一項商品推薦。任務不同，訓練標籤、負樣本與指標也會不同。
- **資料切分：**2014 `Beauty` 已使用每位使用者 leave-one-out 切分，並保留 `data/amazon_reviews/<year>/raw/` 原檔。全域時間切分、冷啟動評估與同日時間戳的敏感度分析仍值得另做；不同切分的結果需分開報告。

### Recall

- **首個基線：**可先用熱門商品推薦作為簡單基線，再與使用者—商品互動模型比較。這是待決定的實驗方向，尚非既定方案。

### Ranking

- 排序模型需要明確的候選集來源、負例抽樣及標籤；應先決定是否用 Recall 的輸出作為候選集，再制定離線評估協定。

### TIGER

- [TIGER 論文](https://arxiv.org/abs/2305.05065) 使用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games 資料，依使用者時間序列預測下一個商品；保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。目前下載的 2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。
- 早期 smoke trainer 使用 catalog-constrained beam search；完整 HF T5 run 使用 unconstrained beam search，十次 validation 的 invalid_beam_fraction 都為 0。兩種入口不可混為同一設定；後續調整仍先看 validation，模型定案後才看 test。

## Problems to solve

### General

1. 確定推薦任務、目標使用情境與成功指標。
2. 選定資料來源與分類，記錄取得方式、版本、欄位及使用條件。
3. 評估同日時間戳排序、冷啟動商品與不同資料切分對結果的影響。

### Recall

1. 建立首個可重現的候選商品召回基線及 Recall@K／NDCG@K 評估。
2. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。

### Ranking

1. 定義候選集來源、負例抽樣、排序標籤與離線評估方式。
2. 在固定候選集上建立 Ranking 基線，避免與 Recall 指標混淆。

### TIGER

1. 修正 SID 訓練（Adagrad accumulator、未使用碼重設、embedding 標準化擇一或組合），以新 SID 版本重訓並驗證 H3；診斷結果見 Facts／TIGER。
2. 保留已完成的完整 run，建立熱門商品基線與單因素對照；模型定案後再對 test 做一次最終評估。
3. 對齊論文的模型容量、訓練預算與解碼協定，確認指標的可比性。

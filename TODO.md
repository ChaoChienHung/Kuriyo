# TODO

本文件追蹤 Kuriyo 的實作工作；研究觀察、已知限制與假設記在 [INSIGHT.md](INSIGHT.md)。

## Repository architecture

- [x] 將共用 SID 生成器及 RQ-VAE 移至 `Representations/SID/`，讓 Recall 模型可引用同一份商品表示。
- [x] 建立 `Recall/core/` 的共用下一商品評估指標，並讓 TIGER 使用。
- [x] 將 SID 商品產物移到資料集層級，讓轉換後的 split 樣本依 SID 版本隔離。
- [x] 將完整商品 catalog 移到年份層，並建立可重跑的 Beauty 準備入口。
- [x] 將共用序列 batching、實驗紀錄與基本 Transformer 放進 `Recall/core/`，在 `Recall/tests/` 驗證。
- [x] 整理 `skills/nus-soc-slurm/SKILL.md`，建立 `scripts/run.sbatch` 作為 SoC 分階段入口。
- [ ] 定義跨 Recall 模型的候選商品輸出格式、模型載入與評估介面。
- [ ] 有實際 Ranking 或端到端模型後，定義 `pipelines/` 的跨階段資料契約。

## Data and representations

- [ ] 在完整 2014 Beauty 商品目錄上產生 `sentence-t5_rqvae` SID，檢查 codebook usage、碰撞率及模型設定。
- [ ] 在 SoC 驗證 `run.sbatch` 的當前 partition／GPU、安裝與下載權限，跑完整 Beauty SID 訓練。
- [ ] 實作 `sentence-t5_kmeans`、`sentence-t5_lsh` 的 SID 生成器，沿用同一商品對照表格式。
- [ ] 明定商品 metadata 的使用範圍，以及固定商品目錄與未見商品兩種評估協定。
- [ ] 若使用 2023 All_Beauty，設計並記錄獨立的資料切分與版本。

## Models and evaluation

- [ ] 建立熱門商品 Recall 基線，用相同的 Beauty validation 與 TIGER 比較。
- [ ] 在完整 train／validation 上訓練 TIGER；定案後才執行 test 評估。
- [ ] 用論文設定 T5 入口跑 200,000 steps，記錄完整 validation；逐項核對與論文 Beauty 四個指標的差距。
- [ ] 為 Ranking 定義候選集、負例、排序標籤和基線。
- [ ] 視模型發展加入可重用的 Recall → Ranking 或端到端 pipeline。

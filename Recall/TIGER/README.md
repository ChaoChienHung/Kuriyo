# TIGER

TIGER（Transformer Index for GEnerative Recommenders）是序列推薦的生成式召回模型：先把商品內容編碼成 Semantic ID（SID），再根據使用者的互動歷史，生成下一件商品的 SID 並查回商品。論文：Shashank Rajput et al., **[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)**，NeurIPS 2023。

`Recall/TIGER/` 是一個模型家族。**每個變體是一個自己完整的空間**：程式、依賴、說明、測試與 runs 都在自己的子目錄裡；家族層只放這份說明、研究筆記 [INSIGHT.md](INSIGHT.md) 與實驗日誌 [Experiment-log.md](Experiment-log.md)（兩者都涵蓋所有變體）。商品 SID 由獨立的 [`Representations/SID/`](../../Representations/SID/README.md) 產生，跨模型共用的元件在 [`Recall/core/`](../core/README.md)。

```text
Recall/TIGER/
├── README.md        # 本檔：家族說明與變體列表
├── INSIGHT.md       # 家族研究筆記
├── Experiment-log.md # 所有變體的實驗日誌（條目標明變體）
├── baseline/        # 目前在 SoC 執行的實作
│   ├── README.md, requirements.txt, train.py, ..., tests/
│   └── runs/<RUN_NAME>/
└── <variant-name>/  # 之後的變體，同樣自成一個完整空間
```

## 變體

| 變體 | 狀態 | 說明 |
| --- | --- | --- |
| [`baseline/`](baseline/README.md) | 已完成首輪 SoC 訓練（200,000 steps）與 validation；尚無最終 test | 依論文已公開設定的 HF T5；`scripts/run.sbatch` 使用此入口。完整操作指南見其 README |

## 外部實作對照

社群的 TIGER 實作只拿來對照數字，不 import、不複製程式。原封不動地在固定 commit 上執行（`scripts/external_tiger.sbatch`，clone、環境、資料與輸出都在 Git 忽略的 `third_party/<名稱>/`），結果記在 [Experiment-log.md](Experiment-log.md)，以 Slurm log 為證據。

| 實作 | 來源 | 授權 | 狀態 | 與 Kuriyo baseline 的主要差異 |
| --- | --- | --- | --- | --- |
| RQ-VAE-Recommender（Botta） | [GitHub](https://github.com/EdoardoBotta/RQ-VAE-Recommender)（`957d32b`），[Zenodo](https://doi.org/10.5281/zenodo.22076163) | MIT | `EXTERNAL=botta` 可執行 | P5 Beauty；sentence-t5-xxl 商品向量與附帶的 RQ-VAE checkpoint；d_model 384；AdamW 1e-3、10k warmup、10k iterations；限制在既有 SID 的生成；**每 1,000 步在 test 上評估**（沒有 validation 選模） |
| GRID（Snap） | [GitHub](https://github.com/snap-research/GRID)（`2fe3475`），論文 Ju et al., [Generative Recommendation with Semantic IDs: A Practitioner's Handbook](https://arxiv.org/abs/2507.22224)，CIKM 2025 | 自訂 LICENSE（GitHub 無法辨識），不複製程式 | 待加入 | P5 Beauty；Flan-T5-XL 向量、RQ-KMeans SID；dropout 0.15、無 user token、Adam 1e-3；beam 10 只保留有效前綴；以 validation Recall@5 選模、early stopping |
| tiger（NonameUntitled） | [GitHub](https://github.com/NonameUntitled/tiger) | MIT | 只讀程式 | LETTER 的資料處理、LLaMA-7B 向量；AdamW 3e-4 固定；beam 100 取 20、限制有效 SID 並排除歷史商品；自報 Beauty test Recall@10 0.0582 |

Kuriyo 的 2014 Beauty 切分與 [P5](https://arxiv.org/abs/2203.13366) 的 `sequential_data.txt` 逐使用者相同（`Recall/INSIGHT.md` Facts），所以 P5 資料上的數字可以和 Kuriyo 的切分直接比較；評估程式則各自不同。

## 新增變體

1. 只有設定不同時，用既有變體加參數跑新的 run，不開新目錄。
2. 變體有 run 記進 Experiment-log 後結果凍結（規則見根目錄 `AGENTS.md`）。要改預設值、換演算法或修正會改變結果的 bug 時，複製該變體成同層的新目錄，名稱描述差異（例如 `constrained-decoding/`），不用 `variant1/`、`v2/`；舊變體保留。
3. 變體之間不互相 import；只可 import 跨模型共用的 `Recall/core/`、`core/` 等。這樣修改 baseline 不會悄悄改變其他變體的結果。
4. 在新變體的 README 寫明與 baseline 的差異，並更新上表；runs 寫到自己的 `runs/`，測試放自己的 `tests/`（含 `__init__.py`），實驗記在家族的 `Experiment-log.md`，Details 寫 `Variant: <name>`。

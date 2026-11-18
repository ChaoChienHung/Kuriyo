# TIGER

TIGER（Transformer Index for GEnerative Recommenders）是序列推薦的生成式召回模型：先把商品內容編碼成 Semantic ID（SID），再根據使用者的互動歷史，生成下一件商品的 SID 並查回商品。論文：Shashank Rajput et al., **[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)**，NeurIPS 2023。

`Recall/TIGER/` 是一個模型家族。**每個變體是一個自己完整的空間**：程式、依賴、說明、測試與 runs 都在自己的子目錄裡；家族層只放這份說明與研究筆記 [INSIGHT.md](INSIGHT.md)（適用於所有變體）。商品 SID 由獨立的 [`Representations/SID/`](../../Representations/SID/README.md) 產生，跨模型共用的元件在 [`Recall/core/`](../core/README.md)。

```text
Recall/TIGER/
├── README.md        # 本檔：家族說明與變體列表
├── INSIGHT.md       # 家族研究筆記
├── baseline/        # 目前在 SoC 執行的實作
│   ├── README.md, requirements.txt, train.py, ..., tests/
│   └── runs/<RUN_NAME>/
└── <variant-name>/  # 之後的變體，同樣自成一個完整空間
```

## 變體

| 變體 | 狀態 | 說明 |
| --- | --- | --- |
| [`baseline/`](baseline/README.md) | 已完成首輪 SoC 訓練（200,000 steps）與 validation；尚無最終 test | 依論文已公開設定的 HF T5；`scripts/run.sbatch` 使用此入口。完整操作指南見其 README |

## 新增變體

1. 只有超參數不同時，在既有變體加具名設定，不開新目錄。
2. 需要不同實作時，複製 `baseline/` 成與它同層的新目錄，名稱描述差異（例如 `constrained-decoding/`），不用 `variant1/`、`v2/`。
3. 變體之間不互相 import；只可 import 跨模型共用的 `Recall/core/`、`core/` 等。這樣修改 baseline 不會悄悄改變其他變體的結果。
4. 在新變體的 README 寫明與 baseline 的差異，並更新上表；runs 寫到自己的 `runs/`，測試放自己的 `tests/`（含 `__init__.py`）。

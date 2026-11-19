# Recall core

不同召回模型共用的程式；測試位於 [`Recall/tests/`](../tests/)，跨階段的 run 目錄與 checkpoint 在 [`core/`](../../core/README.md)。

- [`data.py`](data.py)：2014 ASIN 切分的讀取與檢查（`AsinSplit`：catalog、三個 split、split manifest、歷史上限、寫進 `settings.json` 的資料與前處理摘要）、`load_catalog`、`read_examples`，以及 token padding。
- [`evaluation.py`](evaluation.py)：全部商品排名的評估（`rank_full_catalog`），SASRec 與 FMRec 共用；可帶分組標籤。
- [`metrics.py`](metrics.py)：下一商品任務的 Recall@K 與 NDCG@K；可另外依每筆請求的標籤分組。
- [`popularity.py`](popularity.py)：依目標在 train 當目標的次數（0、1–4、5 以上）分組，三個家族共用，所以分組一致。
- [`transformer.py`](transformer.py)：基本 encoder–decoder Transformer（TIGER 的小模型使用）。

候選商品輸出格式尚未定義。

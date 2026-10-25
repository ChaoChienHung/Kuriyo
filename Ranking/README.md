# Ranking

`Ranking/` 預留給候選商品排序模型。Ranking 接收使用者歷史及 Recall 產生的候選商品，對候選商品重新排序。目前尚未實作 Ranking 模型、候選集產生流程或 Ranking 專用的訓練資料。

## 準備可共用的互動資料

先從 repository 根目錄下載並切分 Amazon Reviews 2014 Beauty：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s tests -v
```

切分結果位於 `data/amazon_reviews/2014/{train,validation,test}/raw/Beauty/examples.jsonl`，每筆有 `user_id`、`history`、`target_asin`、`target_timestamp`。商品內容位於 `data/amazon_reviews/2014/train/raw/Beauty/items.jsonl`。這些資料表示下一件商品的正例，尚未包含候選商品、負例或排序標籤，因此目前**不能直接執行 Ranking 訓練或評估**。

建立 Ranking 流程時，需先固定候選集來源、負例抽樣、訓練標籤與評估協定；validation 和 test 的候選集也需只使用當時可取得的資訊。通用資料規則見根目錄 [README](../README.md)，Recall 規劃見 [Recall README](../Recall/README.md)。

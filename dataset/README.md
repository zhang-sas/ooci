# 预处理数据目录

此仓库不附带原始或预处理数据。将你现有实验使用的文件放入下面的路径，保持原文件内容与划分：

```text
dataset/
  Restaurants_corenlp/{train_with_amr.json,test_with_amr.json}
  Laptops_corenlp/{train_with_amr.json,test_with_amr.json}
  MAMS_corenlp/{train_with_amr.json,test_with_amr.json}
  Tweets_corenlp/{train_with_amr.json,test_with_amr.json}
```

数据须包含句子 token、依存 head/deprel、与文本对齐的 AMR token/edge，以及方面列表。
不能将没有词对齐的 AMR 概念节点索引当成文本词索引。

完整格式与检查指令见 [数据说明](../docs/DATA.md)。本仓库没有包含或自动运行 CoreNLP/AMR 解析器；
它接收已有、经过核验的 `*_with_amr.json`。不要用空图或线性链替代已有解析图来声称复现原实验。

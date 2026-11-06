# 数据格式与准备

本发布目录使用现有 `*_with_amr.json` 文件。训练入口不下载数据，不改动划分，不用测试标签生成图。

## 文件结构

顶层为记录列表，每条记录至少包含：

| 字段 | 含义 |
|---|---|
| `token` | 原句的词列表 |
| `head` | 与 token 等长；依存父节点为 1 起始索引，0 表示根 |
| `deprel` | 与 token 等长；原解析器的依存关系字符串 |
| `amr_tokens` | AMR 词对齐所对应的文本切词 |
| `amr_edges` | `[source_index, relation, target_index]`；端点为 amr_tokens 的 0 起始索引 |
| `aspects` | 方面列表，包含 `from`、`to`、`term`、`polarity` |

方面区间为原 token 的 `[from,to)`，`term` 为相应词列表；训练/测试的极性为
`positive`、`negative` 或 `neutral`。极性仅作为分类目标/评估标签，不作为图或路由输入。

`amr_tokens` 与 `token` 可以切词边界不同，但忽略空白、大小写后必须对应同一文本。
本实现通过字符区间对齐转换 AMR 端点，不支持将未对齐的概念图直接冒充词图。

同一文本和图的多条记录按原 loader 的键合并成句子，去重方面跨度并保留方面列表。
因此日志中的句子数不等于分类样本（方面）数。

## 使用原服务器文件

新仓库与原项目并列时，可用软链接复用资产：

```bash
cd /你的路径/DAGF-AEOR-RG
ln -s /HUBU-AI009/DAGF-main/dataset/Restaurants_corenlp dataset/Restaurants_corenlp
ln -s /HUBU-AI009/DAGF-main/dataset/Laptops_corenlp dataset/Laptops_corenlp
ln -s /HUBU-AI009/DAGF-main/dataset/MAMS_corenlp dataset/MAMS_corenlp
ln -s /HUBU-AI009/DAGF-main/dataset/Tweets_corenlp dataset/Tweets_corenlp
```

上面只适用于对应目标尚不存在时；也可以直接传 `train.py --train-file ... --test-file ...`。
模型和词典可使用 `--roberta-path`、`--lexicon-dir` 外部路径。
不要将这些本地资产路径提交到 GitHub。

```bash
# 仅检查四个数据集中指定的一集的输入数据
python -B scripts/check_inputs.py restaurant --data-only
# 检查一个自定义文件
python -B scripts/check_inputs.py --file /path/to/train_with_amr.json
```

检查输出包含原始记录数、合并后句子数和方面数。检查只覆盖字段、区间和文本对齐条件，
不证明依存/AMR 的语义正确，也不将已有数据描述成意见跨度金标。

## 原始数据的预处理

若只有原始 XML/文本，需要自行使用与实验一致的依存解析器、AMR 解析器和词对齐过程，
导出上述字段后再检查。本仓库未包含这些外部解析器及其模型，也未生成替代图。
复现实验应优先使用原实验文件，并核对哈希、划分和方面数。

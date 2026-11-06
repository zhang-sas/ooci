# DAGF-AEOR-RG

完整模型 **AEOR（方面—意见归属路由）＋RG（关系边重权）** 的独立代码目录。

本目录从现有完整模型实现中提取，保留冻结 v1 的编码、图分支、路由、共享交互、CLS 分类读出与训练目标。
不包含 v3 修复、多层路由、scope、TAPT 等独立实验。模型与训练后端源码按字节保留，来源和 SHA-256 见
[`source_provenance.json`](source_provenance.json)。[English guide](README_en.md)。

## 模型

- 以句子为样本保留多个方面，复用文本编码分支及 MLM 分支各自的句子计算。
- AEOR 使用独占方面通道、共享通道和背景通道，并通过方面成员门分配共享证据：`R[a,i] = E[a,i] + S[i] * M[a,i]`。
- 在图读出与融合前确定方面条件路由；方面间交互传递共享证据，使用共享重叠门。
- RG 在原有最后一层依存 GCN 的消息权重上进行关系条件重权，不增加额外的关系消息分支。
- 结构先验来自文本和解析图，不使用测试金标极性构造边。
- 损失保持句子平均 CE 与原路由正则；`baseline_regularization_weight=0`，原 KL 与辅助分类项不计入优化目标。
- 该版本没有意见跨度/共享成员抽取监督。路由属于潜在结构，其可解释性及冲突隔离效果需通过独立机制评估检验。

## 环境

沿用已有可工作的 `dagf` 环境即可。依赖版本保留在 [`requirements.txt`](requirements.txt)，对应 Python 3.8/3.10 环境。
新环境示例：

```bash
conda create -n aeor-rg python=3.8 -y
conda activate aeor-rg

# GPU 环境先安装与服务器兼容的 PyTorch；下面沿用原环境的 cu121 配置
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 准备输入

代码仓库不包含训练/测试数据、预训练权重、词典数据、模型检查点和运行日志。
将原实验使用的相同资产放入以下路径，或使用命令参数指定外部路径：

```text
DAGF-AEOR-RG/
  dataset/
    Restaurants_corenlp/{train_with_amr.json,test_with_amr.json}
    Laptops_corenlp/{train_with_amr.json,test_with_amr.json}
    MAMS_corenlp/{train_with_amr.json,test_with_amr.json}
    Tweets_corenlp/{train_with_amr.json,test_with_amr.json}
  roberta/{config.json,vocab.json,merges.txt,模型权重及原tokenizer配置}
  opinion-lexicon-English/{positive-words.txt,negative-words.txt}
```

数据格式和索引约定见 [`docs/DATA.md`](docs/DATA.md)。此版本接收已有的词对齐依存/AMR JSON，未附带原始文本解析流水线。
检查输入仅检查文件和字段结构，不运行模型，也不代表图语义或标注真值已获人工核验。

```bash
python -B verify_release.py
python -B scripts/check_inputs.py all
python -B scripts/check_inputs.py twitter
```

## 单次训练

在仓库根目录执行，每条指令只训练完整模型：

```bash
bash run.sh restaurant 1000
bash run.sh laptop 1000
bash run.sh mams 1000
bash run.sh twitter 1000
```

也可直接运行 Python。下例复用服务器原项目中的数据、模型和词典，不复制大文件：

```bash
python -B train.py --dataset restaurant --seed 1000 \
  --train-file /HUBU-AI009/DAGF-main/dataset/Restaurants_corenlp/train_with_amr.json \
  --test-file /HUBU-AI009/DAGF-main/dataset/Restaurants_corenlp/test_with_amr.json \
  --roberta-path /HUBU-AI009/DAGF-main/roberta \
  --lexicon-dir /HUBU-AI009/DAGF-main/opinion-lexicon-English
```

| 数据集 | epoch | 测试间隔（参数更新次数） | 句子 batch | lr | 最大子词长度 |
|---|---:|---:|---:|---:|---:|
| Restaurant | 50 | 20 | 16 | 2e-5 | 384 |
| Laptop | 30 | 10 | 16 | 2e-5 | 512 |
| MAMS | 30 | 20 | 16 | 2e-5 | 384 |
| Twitter | 30 | 10 | 16 | 1e-5 | 384 |

默认测试间隔沿用现有完整模型实验；每个 epoch 结束时也会测试。
MAMS 是本项目增加的实验配置，不是原 DAGF 论文报告的设置。
完整预设见 [`configs/`](configs/)。公开入口不读取全局 `EPOCHS`/`LEARNING_RATE` 环境变量，调整须显式传参。

```bash
# 只检查计划，不创建训练目录、不加载模型
python -B train.py --dataset restaurant --seed 1000 --dry-run

# 显式调整测试间隔；其余设置仍来自该数据集完整模型预设
bash run.sh restaurant 1000 --log-step 20
```

## 五个种子，一次运行

`all` 包含 Restaurant、Laptop、MAMS。Twitter 通过单集入口运行。
默认种子为 `1000 1001 1002 1003 1004`，在一张 GPU 上依次训练。

```bash
bash run_seeds.sh all
bash run_seeds.sh twitter

# 一次跑三个数据集，每集仅一个种子
bash run_seeds.sh all --seeds 1000

# 仅打印十五次训练的指令
bash run_seeds.sh all --dry-run
```

重复相同指令会核验并跳过本批已完整完成的任务。中断/失败任务从头训练到新的 attempt，
不恢复优化器或中断 epoch。变更数据、源码或设置后使用 `--fresh` 开始新的一批。

输出 `runs_suites/full_all_seeds_1000_1001_1002_1003_1004/` 包含：

- `per_seed.csv`：逐次训练的最高 Acc、最高 Macro-F1，以及 F1 最优检查点的 Acc。
- `mean_std.csv`：全部指定种子完成后的均值和样本标准差，`ddof=1`。
- `suite_manifest.json`、`summary.json`：配置、输入指纹、运行目录和完成状态。

缺少任何指定种子时不输出完整均值/标准差；只运行一个种子时标准差留空。

## 测试与检查点约定

**复现协议保持“完整训练集＋周期测试选模”，不新增验证集。**
这是与既有实验一致的 test-selection 协议，不能写成独立验证集选模。
Acc 与 Macro-F1 的最大值分别记录，可能来自不同检查点。

- `best_acc.pt`：测试 Acc 最优权重。
- `best_f1.pt`：测试 Macro-F1 最优权重。
- `best.pt`：原协议的兼容别名，指向 Acc 最优权重。
- `test_report.json`：默认最终报告对应 `best_acc.pt`；需要 F1 最优点的成对指标时评估 `best_f1.pt`。

```bash
python -B evaluate.py \
  --checkpoint runs/restaurant/seed_1000/实际运行目录/best_f1.pt \
  --data-file dataset/Restaurants_corenlp/test_with_amr.json \
  --output runs_eval/restaurant_best_f1 \
  --device cuda --export-routes
```

评估入口同时支持新 `paper full` 和旧冻结 `v1_relation_gate` 完整模型检查点，拒绝纯 v1、关闭模块或其他版本权重。
移动目录后可通过 `--roberta-path`、`--lexicon-dir` 指向相同资产。`--output` 必须是新目录。
最终路由导出包含 E/S/M/R 与背景权重，不自动将它们当作已校准的意见归属概率。

## 代码位置

| 文件 | 内容 |
|---|---|
| `train.py`、`run.sh` | 仅完整模型的训练入口 |
| `run_experiments.py`、`run_seeds.sh` | 多数据集、多种子顺序运行与汇总 |
| `evaluate.py` | 完整模型评估、路由导出 |
| `model.py` | 对外模型类及关键模块导入 |
| `innovations/v1_reference/model.py` | 原 AEOR、共享交互、句子 CE |
| `innovations/v1_relation_gate/{model,layers}.py` | v1＋关系边重权 RG |
| `innovations/paper_experiments/{model,train}.py` | 当前完整模型包装与原训练入口 |
| `dagf_complete/` | 独立运行所需的既有 DAGF 图分支和 AGF 核心 |
| `aeor_rg/runtime.py` | 路径解析、配置约束和完成结果核验 |
| `scripts/check_inputs.py` | 数据及资产结构检查 |

少量沿用历史名称的工具模块是原训练代码的导入依赖，并未启用同名实验。
原项目的基线、冻结 v1 和其他文件夹均未被此发布目录修改。

## 上传 GitHub

上传本目录中的代码文件，使用已包含的 `.gitignore`。本目录未初始化 `.git`，也未自动创建或推送远程仓库。

```bash
git init
git add .
git status --short
git commit -m "Release full AEOR+RG implementation"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

公开文件清单与 SHA-256 见 [`release_manifest.json`](release_manifest.json)。
上游来源与许可证状态见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；此整理未擅自为上游代码指定新许可证。
